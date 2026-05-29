import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from pytorch_ssim import ssim


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, reduction='mean'):
        loss = torch.sqrt((pred - target) ** 2 + self.eps ** 2)
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'none':
            return torch.mean(loss, dim=[1, 2, 3])

class FrequencySeparator(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x, grayscale_high=True):
        low_freq = F.avg_pool2d(
            x,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding
        )

        if grayscale_high:

            # luminance only
            gray = (
                0.299 * x[:, 0:1] +
                0.587 * x[:, 1:2] +
                0.114 * x[:, 2:3]
            )

            gray_low = F.avg_pool2d(
                gray,
                kernel_size=self.kernel_size,
                stride=1,
                padding=self.padding
            )

            high_freq = gray - gray_low

        else:
            high_freq = x - low_freq

        return low_freq, high_freq

class Stage1Loss(nn.Module):
    def __init__(
        self,
        prior_weight=0.01,
        freq_kernel=5,

        # frequency
        low_weight=0.5,
        high_weight=1.2,

        # global weights
        w_spatial=1.0,
        w_freq=1.0,
        w_ssim=0.05,
        ssim_warmup_epochs=5,
    ):
        super().__init__()

        self.prior_weight = prior_weight

        self.low_weight = low_weight
        self.high_weight = high_weight

        self.w_spatial = w_spatial
        self.w_freq = w_freq
        self.w_ssim = w_ssim

        self.ssim_warmup_epochs = ssim_warmup_epochs

        self.charbonnier = CharbonnierLoss()
        self.freq_separator = FrequencySeparator(kernel_size=freq_kernel)

    # SSIM Warmup
    def get_ssim_weight(self, current_epoch):

        if self.ssim_warmup_epochs <= 0:
            return self.w_ssim

        progress = min(current_epoch / self.ssim_warmup_epochs, 1.0)

        return progress * self.w_ssim

    def reconstruction_loss(self, pred, target, current_epoch=0):

        # Direct spatial reconstruction
        spatial_loss = self.charbonnier(pred, target)

        # Frequency decomposition
        pred_low, pred_high = self.freq_separator(pred)
        target_low, target_high = self.freq_separator(target)

        loss_low = self.charbonnier(pred_low, target_low)
        loss_high = self.charbonnier(pred_high, target_high)

        freq_loss = (
            self.low_weight * loss_low
            + self.high_weight * loss_high
        )

        # SSIM
        pred_safe = torch.clamp(pred, 0.0, 1.0)
        target_safe = torch.clamp(target, 0.0, 1.0)

        ssim_loss = 1.0 - ssim(
            pred_safe,
            target_safe,
            data_range=1.0,
            size_average=True
        )

        current_ssim_weight = self.get_ssim_weight(
            current_epoch
        )

        combined = (
            self.w_spatial * spatial_loss
            + self.w_freq * freq_loss
            + current_ssim_weight * ssim_loss
        )

        return (
            combined,
            spatial_loss,
            loss_low,
            loss_high,
            ssim_loss,
            current_ssim_weight,
        )

    def forward(
        self,
        target,
        final_pred,
        intermediate_preds,
        halting_weights,
        current_epoch=0,
        **kwargs
    ):

        T = len(intermediate_preds)

        total_rec = 0.0

        total_spatial = 0.0
        total_low = 0.0
        total_high = 0.0
        total_ssim = 0.0
        current_ssim_weight = 0.0

        for t in range(T):
            rec_t, spatial_t, low_t, high_t, ssim_t, current_ssim_weight = self.reconstruction_loss(intermediate_preds[t], target)
            total_rec += rec_t

            total_spatial += spatial_t
            total_low += low_t
            total_high += high_t

            total_ssim += ssim_t

        # Average over time steps
        avg_rec = total_rec / T

        avg_spatial = total_spatial / T
        avg_low = total_low / T
        avg_high = total_high / T

        avg_ssim = total_ssim / T

        # KL regularization → uniform halting
        hw_b = halting_weights.transpose(0, 1) # (T, B) -> (B, T)
        # KL(Halting || Uniform)
        kl_loss = torch.sum(hw_b * (torch.log(hw_b + 1e-8) - math.log(1.0 / T)), dim=1).mean()

        total_loss = avg_rec + self.prior_weight * kl_loss

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/rec": avg_rec.item(),
            "loss/spatial": avg_spatial.item(),
            "loss/low_freq": avg_low.item(),
            "loss/high_freq": avg_high.item(),
            "loss/ssim": avg_ssim.item(),
            "loss/ssim_weight": current_ssim_weight,
            "loss/kl_gate": kl_loss.item(),
        }

        return total_loss, loss_dict


class Stage2Loss(nn.Module):
    """Stage 2: Chỉ tune Gate + Ponder"""
    def __init__(self, gain_threshold=0.003, gain_scale=12.0, 
                 ponder_weight=0.015, alpha=0.7):
        super().__init__()
        self.gain_threshold = gain_threshold
        self.gain_scale = gain_scale
        self.ponder_weight = ponder_weight
        self.alpha = alpha
        self.charbonnier = CharbonnierLoss()

    def reconstruction_loss(self, pred, target):
        # Tính theo từng ảnh (Shape: (B,))
        charb = self.charbonnier(pred, target, reduction='none')
        pred_clamp = torch.clamp(pred, 0.0, 1.0)
        
        # size_average=False để trả về tensor (B,)
        ssim_val = ssim(pred_clamp, target, data_range=1.0, size_average=False)
        ssim = 1.0 - ssim_val
        
        combined = self.alpha * charb + (1.0 - self.alpha) * ssim
        return combined, charb, ssim

    def forward(self, target, final_pred, intermediate_preds, halting_weights, halt_logits, **kwargs):
        T = len(intermediate_preds)
        B = target.shape[0]

        step_losses = []
        avg_charb = 0.0
        avg_ssim = 0.0

        with torch.no_grad():
            for pred_t in intermediate_preds:
                rec_t, charb_t, ssim_t = self.reconstruction_loss(pred_t, target)
                step_losses.append(rec_t) # rec_t hiện có shape (B,)
                avg_charb += charb_t.mean()
                avg_ssim += ssim_t.mean()

        avg_charb /= T
        avg_ssim /= T

        gate_loss = 0.0
        prev_loss = None

        for t in range(T):
            curr_loss = step_losses[t]
            if prev_loss is None or t == 0:
                target_prob = torch.zeros(B, device=curr_loss.device) # (B,)
            else:
                gain = prev_loss - curr_loss # Shape: (B,) - Tính gain cho từng ảnh!
                utility = torch.sigmoid(self.gain_scale * (gain - self.gain_threshold))
                target_prob = (1.0 - utility).detach() # Shape: (B,)

            # Lúc này target_prob và halt_logits[t].view(-1) đều là (B,), rất chuẩn xác
            gate_loss += F.binary_cross_entropy_with_logits(
                halt_logits[t].view(-1), target_prob, reduction='mean'
            )
            prev_loss = curr_loss

        gate_loss /= T

        # Ponder loss 
        expected_steps = torch.sum((torch.arange(1, T+1, device=halting_weights.device).float().view(-1, 1) * halting_weights), dim=0).mean()

        total_loss = gate_loss + self.ponder_weight * expected_steps

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/gate": gate_loss.item(),
            "loss/ponder": (self.ponder_weight * expected_steps).item(),
            "loss/expected_steps": expected_steps.item(),
            "eval/charbonnier": avg_charb.item(),
            "eval/ssim": avg_ssim.item(),
        }

        return total_loss, loss_dict