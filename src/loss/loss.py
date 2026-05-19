import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps ** 2))

class Stage1Loss(nn.Module):

    def __init__(
        self,
        prior_weight=0.001,
        alpha=0.84
    ):
        super().__init__()

        self.prior_weight = prior_weight
        self.alpha = alpha

        self.charbonnier = CharbonnierLoss()

    def reconstruction_loss(
        self,
        pred,
        target
    ):

        charbonnier_loss = self.charbonnier(
            pred,
            target
        )

        ms_ssim_loss = 1.0 - ms_ssim(
            pred,
            target,
            data_range=1.0,
            size_average=True
        )

        return (
            self.alpha * charbonnier_loss
            + (1.0 - self.alpha) * ms_ssim_loss
        )

    def forward(
        self,
        target,
        layer_outputs,
        halting_weights,
        decoder
    ):

        T = len(layer_outputs)

        total_rec_loss = 0.0

        # =====================================
        # multi-step reconstruction
        # =====================================

        for t in range(T):

            pred_t = decoder(layer_outputs[t])

            rec_loss_t = self.reconstruction_loss(
                pred_t,
                target
            )

            total_rec_loss += (
                halting_weights[t].mean()
                * rec_loss_t
            )

        # =====================================
        # uniform prior KL
        # =====================================

        prior = torch.full_like(
            halting_weights,
            1.0 / T
        )

        kl_loss = F.kl_div(
            torch.log(halting_weights + 1e-8),
            prior,
            reduction='batchmean'
        )

        # =====================================
        # final
        # =====================================

        total_loss = (
            total_rec_loss
            + self.prior_weight * kl_loss
        )

        return total_loss

class Stage2Loss(nn.Module):

    def __init__(
        self,
        gain_threshold=0.005,
        gain_scale=10.0,
        ponder_weight=0.02,
        alpha=0.84
    ):
        super().__init__()

        self.gain_threshold = gain_threshold
        self.gain_scale = gain_scale
        self.ponder_weight = ponder_weight
        self.alpha = alpha

        self.charbonnier = CharbonnierLoss()

    def reconstruction_loss(
        self,
        pred,
        target
    ):

        charbonnier_loss = self.charbonnier(
            pred,
            target
        )

        ms_ssim_loss = 1.0 - ms_ssim(
            pred,
            target,
            data_range=1.0,
            size_average=True
        )

        return (
            self.alpha * charbonnier_loss
            + (1.0 - self.alpha) * ms_ssim_loss
        )

    def forward(
        self,
        target,
        layer_outputs,
        halt_logits,
        halting_weights,
        decoder
    ):

        T = len(layer_outputs)

        step_losses = []

        # =====================================
        # compute reconstruction losses
        # =====================================

        with torch.no_grad():

            for t in range(T):

                pred_t = decoder(layer_outputs[t])

                loss_t = self.reconstruction_loss(
                    pred_t,
                    target
                )

                step_losses.append(loss_t)

        # =====================================
        # gate supervision
        # =====================================

        gate_loss = 0.0

        prev_loss = None

        for t in range(T - 1):

            current_loss = step_losses[t]

            if prev_loss is None:

                gain_t = torch.tensor(
                    0.05,
                    device=current_loss.device
                )

            else:

                gain_t = prev_loss - current_loss

            prev_loss = current_loss

            utility = torch.sigmoid(
                self.gain_scale
                * (gain_t - self.gain_threshold)
            )

            target_exit = (1.0 - utility).detach()

            logits_t = halt_logits[t]

            target_tensor = torch.full_like(
                logits_t,
                target_exit
            )

            gate_loss += F.binary_cross_entropy_with_logits(
                logits_t,
                target_tensor
            )

        # =====================================
        # ponder regularization
        # =====================================

        expected_steps = 0.0

        for t in range(T):

            expected_steps += (
                (t + 1)
                * halting_weights[t].mean()
            )

        total_loss = (
            gate_loss
            + self.ponder_weight * expected_steps
        )

        return total_loss