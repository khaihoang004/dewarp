import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


# =========================================================
# CHARBONNIER
# =========================================================

class CharbonnierLoss(nn.Module):

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):

        loss = torch.sqrt(
            (pred - target).pow(2) +
            self.eps ** 2
        )

        return loss.mean()


# =========================================================
# FREQUENCY SEPARATOR
# =========================================================

class FrequencySeparator(nn.Module):

    def __init__(self, kernel_size=5):

        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x):

        low_freq = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding
        )

        high_freq = x - low_freq

        return low_freq, high_freq


# =========================================================
# STAGE 1 LOSS
# =========================================================

class Stage1Loss(nn.Module):
    """
    Stage 1:
    - Learn reconstruction
    - Stabilize refinement
    - Encourage balanced halting distribution
    """

    def __init__(self,

                 # =============================================
                 # ACT prior
                 # =============================================

                 prior_weight=0.003,

                 # =============================================
                 # frequency decomposition
                 # =============================================

                 freq_kernel=5,
                 high_weight=1.2,

                 # =============================================
                 # reconstruction weights
                 # =============================================

                 w_global=0.7,
                 w_freq=0.8,
                 w_msssim=0.2):

        super().__init__()

        self.prior_weight = prior_weight

        self.high_weight = high_weight

        self.w_global = w_global
        self.w_freq = w_freq
        self.w_msssim = w_msssim

        self.charbonnier = CharbonnierLoss()

        self.freq_separator = FrequencySeparator(
            kernel_size=freq_kernel
        )

    # =====================================================
    # RECONSTRUCTION LOSS
    # =====================================================

    def reconstruction_loss(self, pred, target):

        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        # -------------------------------------------------
        # global reconstruction
        # -------------------------------------------------

        global_loss = self.charbonnier(
            pred,
            target
        )

        # -------------------------------------------------
        # frequency decomposition
        # -------------------------------------------------

        pred_low, pred_high = self.freq_separator(pred)

        target_low, target_high = self.freq_separator(target)

        # low-frequency
        loss_low = self.charbonnier(
            pred_low,
            target_low
        )

        # high-frequency
        loss_high = self.charbonnier(
            pred_high,
            target_high
        )

        freq_loss = (
            loss_low +
            self.high_weight * loss_high
        )

        # -------------------------------------------------
        # perceptual structure
        # -------------------------------------------------

        ms_ssim_loss = 1.0 - ms_ssim(
            pred,
            target,
            data_range=1.0,
            size_average=True
        )

        # -------------------------------------------------
        # final reconstruction
        # -------------------------------------------------

        combined = (

            self.w_global * global_loss +

            self.w_freq * freq_loss +

            self.w_msssim * ms_ssim_loss
        )

        return (
            combined,
            global_loss,
            loss_low,
            loss_high,
            ms_ssim_loss
        )

    # =====================================================
    # FORWARD
    # =====================================================

    def forward(
        self,
        target,
        final_pred,
        intermediate_preds,
        halting_weights,
        **kwargs
    ):

        T = len(intermediate_preds)

        total_rec = 0.0

        total_global = 0.0
        total_low = 0.0
        total_high = 0.0
        total_msssim = 0.0

        # =================================================
        # reconstruction over all exits
        # =================================================

        for pred_t in intermediate_preds:

            (
                rec_t,
                global_t,
                low_t,
                high_t,
                msssim_t

            ) = self.reconstruction_loss(
                pred_t,
                target
            )

            total_rec += rec_t

            total_global += global_t
            total_low += low_t
            total_high += high_t
            total_msssim += msssim_t

        # =================================================
        # average losses
        # =================================================

        avg_rec = total_rec / T

        avg_global = total_global / T
        avg_low = total_low / T
        avg_high = total_high / T
        avg_msssim = total_msssim / T

        # =================================================
        # halting prior regularization
        # =================================================

        # expected shape:
        # halting_weights -> [T, B]

        halting_dist = halting_weights.mean(dim=1)

        prior = torch.full_like(
            halting_dist,
            1.0 / T
        )

        kl_loss = F.kl_div(
            torch.log(halting_dist + 1e-8),
            prior,
            reduction='batchmean'
        )

        # =================================================
        # final total loss
        # =================================================

        total_loss = (
            avg_rec +
            self.prior_weight * kl_loss
        )

        # =================================================
        # logging
        # =================================================

        loss_dict = {

            "loss/total": total_loss.item(),

            "loss/rec": avg_rec.item(),

            "loss/global": avg_global.item(),

            "loss/low_freq": avg_low.item(),

            "loss/high_freq": avg_high.item(),

            "loss/ms_ssim": avg_msssim.item(),

            "loss/kl_gate": kl_loss.item(),
        }

        return total_loss, loss_dict


# =========================================================
# STAGE 2 LOSS
# =========================================================

class Stage2Loss(nn.Module):
    """
    Stage 2:
    - Freeze backbone
    - Learn halting policy
    - Optimize stopping utility
    """

    def __init__(self,

                 gain_threshold=0.007,
                 gain_scale=6.0,

                 ponder_weight=0.01,

                 alpha=0.7):

        super().__init__()

        self.gain_threshold = gain_threshold
        self.gain_scale = gain_scale

        self.ponder_weight = ponder_weight

        self.alpha = alpha

        self.charbonnier = CharbonnierLoss()

    # =====================================================
    # reconstruction utility
    # =====================================================

    def reconstruction_loss(self, pred, target):

        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        charb = self.charbonnier(
            pred,
            target
        )

        msssim = 1.0 - ms_ssim(
            pred,
            target,
            data_range=1.0,
            size_average=True
        )

        combined = (
            self.alpha * charb +
            (1.0 - self.alpha) * msssim
        )

        return combined, charb, msssim

    # =====================================================
    # forward
    # =====================================================

    def forward(
        self,
        target,
        final_pred,
        intermediate_preds,
        halting_weights,
        halt_logits,
        **kwargs
    ):

        T = len(intermediate_preds)

        step_losses = []

        avg_charb = 0.0
        avg_msssim = 0.0

        # =================================================
        # evaluate reconstruction quality
        # =================================================

        with torch.no_grad():

            for pred_t in intermediate_preds:

                rec_t, charb_t, msssim_t = (
                    self.reconstruction_loss(
                        pred_t,
                        target
                    )
                )

                step_losses.append(rec_t)

                avg_charb += charb_t
                avg_msssim += msssim_t

        avg_charb /= T
        avg_msssim /= T

        # =================================================
        # gate supervision
        # =================================================

        gate_loss = 0.0

        prev_loss = None

        for t in range(T):

            curr_loss = step_losses[t]

            if prev_loss is None:

                target_prob = torch.zeros(
                    curr_loss.shape[0],
                    device=curr_loss.device
                )

            else:

                gain = prev_loss - curr_loss

                utility = torch.sigmoid(
                    self.gain_scale * (
                        gain - self.gain_threshold
                    )
                )

                target_prob = (
                    1.0 - utility
                ).detach()

            gate_loss += F.binary_cross_entropy_with_logits(
                halt_logits[t].view(-1),
                target_prob,
                reduction='mean'
            )

            prev_loss = curr_loss

        gate_loss /= T

        # =================================================
        # ponder cost
        # =================================================

        step_ids = torch.arange(
            1,
            T + 1,
            device=halting_weights.device
        ).float().view(-1, 1)

        expected_steps = (
            step_ids * halting_weights
        ).sum(dim=0).mean()

        # =================================================
        # final total loss
        # =================================================

        total_loss = (
            gate_loss +
            self.ponder_weight * expected_steps
        )

        # =================================================
        # logging
        # =================================================

        loss_dict = {

            "loss/total": total_loss.item(),

            "loss/gate": gate_loss.item(),

            "loss/ponder": (
                self.ponder_weight *
                expected_steps
            ).item(),

            "loss/expected_steps": expected_steps.item(),

            "eval/charbonnier": avg_charb.item(),

            "eval/ms_ssim": avg_msssim.item(),
        }

        return total_loss, loss_dict