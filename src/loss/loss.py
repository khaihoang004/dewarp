import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ms_ssim


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

    def forward(self, x):
        low_freq = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        high_freq = x - low_freq
        return low_freq, high_freq


class Stage1Loss(nn.Module):
    """Stage 1: Học refinement + phân bố halting đều"""
    def __init__(self, prior_weight=0.01, freq_kernel=5, high_weight=1.5, 
                 w_freq=1.0, w_msssim=0.8):
        super().__init__()
        self.prior_weight = prior_weight
        self.high_weight = high_weight
        self.w_freq = w_freq
        self.w_msssim = w_msssim
        
        self.charbonnier = CharbonnierLoss()
        self.freq_separator = FrequencySeparator(kernel_size=freq_kernel)

    def reconstruction_loss(self, pred, target):
        pred_low, pred_high = self.freq_separator(pred)
        target_low, target_high = self.freq_separator(target)

        loss_low = self.charbonnier(pred_low, target_low)
        loss_high = self.charbonnier(pred_high, target_high)
        
        freq_loss = loss_low + self.high_weight * loss_high

        # MS-SSIM
        pred_safe = torch.clamp(pred, 0.0, 1.0)
        ms_ssim_loss = 1.0 - ms_ssim(pred_safe, target, data_range=1.0, size_average=True)

        combined = self.w_freq * freq_loss + self.w_msssim * ms_ssim_loss
        return combined, loss_low, loss_high, ms_ssim_loss

    def forward(self, target, final_pred, intermediate_preds, halting_weights, **kwargs):
        T = len(intermediate_preds)
        B = target.shape[0]

        total_rec = 0.0
        total_low = 0.0
        total_high = 0.0
        total_msssim = 0.0

        for t in range(T):
            rec_t, low_t, high_t, msssim_t = self.reconstruction_loss(intermediate_preds[t], target)
            total_rec += rec_t
            total_low += low_t
            total_high += high_t
            total_msssim += msssim_t

        # Average over time steps
        avg_rec = total_rec / T
        avg_low = total_low / T
        avg_high = total_high / T
        avg_msssim = total_msssim / T

        # KL regularization → uniform halting
        halting_mean = halting_weights.mean(dim=1)                    # (T, B) -> (T,)
        prior = torch.full_like(halting_mean, 1.0 / T)
        kl_loss = F.kl_div(torch.log(halting_mean + 1e-8), prior, reduction='sum')

        total_loss = avg_rec + self.prior_weight * kl_loss

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/rec": avg_rec.item(),
            "loss/low_freq": avg_low.item(),
            "loss/high_freq": avg_high.item(),
            "loss/ms_ssim": avg_msssim.item(),
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
        msssim_val = ms_ssim(pred_clamp, target, data_range=1.0, size_average=False)
        msssim = 1.0 - msssim_val
        
        combined = self.alpha * charb + (1.0 - self.alpha) * msssim
        return combined, charb, msssim

    def forward(self, target, final_pred, intermediate_preds, halting_weights, halt_logits, **kwargs):
        T = len(intermediate_preds)
        B = target.shape[0]

        step_losses = []
        avg_charb = 0.0
        avg_msssim = 0.0

        with torch.no_grad():
            for pred_t in intermediate_preds:
                rec_t, charb_t, msssim_t = self.reconstruction_loss(pred_t, target)
                step_losses.append(rec_t) # rec_t hiện có shape (B,)
                avg_charb += charb_t.mean()
                avg_msssim += msssim_t.mean()

        avg_charb /= T
        avg_msssim /= T

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
            "eval/ms_ssim": avg_msssim.item(),
        }

        return total_loss, loss_dict