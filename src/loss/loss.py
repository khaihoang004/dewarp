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


class FrequencySeparator(nn.Module):
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def gaussian_blur(self, x):
        return F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)

    def forward(self, x):
        low_freq = self.gaussian_blur(x)
        high_freq = x - low_freq
        return low_freq, high_freq


class FocalFrequencyLoss(nn.Module):
    def __init__(self, alpha=1.2):
        super().__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        pred_fft = torch.fft.fft2(pred, norm='ortho')
        targ_fft = torch.fft.fft2(target, norm='ortho')

        pred_amp = torch.abs(pred_fft)
        targ_amp = torch.abs(targ_fft)

        freq_dist = (pred_amp - targ_amp) ** 2
        weight = (freq_dist / (freq_dist.mean(dim=[2, 3], keepdim=True) + 1e-8)) ** self.alpha
        loss = (weight * freq_dist).mean()
        return loss


class Stage1Loss(nn.Module):
    def __init__(self,
                 high_weight=1.9,      # Giảm nhẹ so với 2.0 gốc vì high_freq đã khá tốt
                 lambda_ffl=0.06,      # FFL nhỏ để hỗ trợ, tránh làm low_freq bị lấn át
                 w_charb=0.75,         # Tăng mạnh để giảm low_freq loss
                 w_freq=0.75,          # Giảm để nhường chỗ cho Charbonnier + MS-SSIM
                 w_msssim=1.05,        # Tăng mạnh (gốc là 0) để cải thiện perceptual & val MS-SSIM
                 prior_weight=0.006):  # Giảm nhẹ để reconstruction chiếm ưu thế
        super().__init__()

        self.high_weight = high_weight
        self.lambda_ffl = lambda_ffl
        self.w_charb = w_charb
        self.w_freq = w_freq
        self.w_msssim = w_msssim
        self.prior_weight = prior_weight

        self.charbonnier = CharbonnierLoss()
        self.freq_separator = FrequencySeparator(kernel_size=5)
        self.ffl = FocalFrequencyLoss(alpha=1.2)

    def reconstruction_loss(self, pred, target):
        pred = torch.clamp(pred, 0.0, 1.0)
        target = torch.clamp(target, 0.0, 1.0)

        # Global pixel accuracy
        charb_loss = self.charbonnier(pred, target)

        # Frequency separation
        pred_low, pred_high = self.freq_separator(pred)
        targ_low, targ_high = self.freq_separator(target)
        loss_low = self.charbonnier(pred_low, targ_low)
        loss_high = self.charbonnier(pred_high, targ_high)
        freq_loss = loss_low + self.high_weight * loss_high

        # Focal Frequency Loss
        ffl_loss = self.ffl(pred, target)

        # Perceptual
        ms_ssim_loss = 1.0 - ms_ssim(pred, target, data_range=1.0, size_average=True)

        # Combined loss
        combined = (self.w_charb * charb_loss +
                    self.w_freq * freq_loss +
                    self.lambda_ffl * ffl_loss +
                    self.w_msssim * ms_ssim_loss)

        return combined, charb_loss, loss_low, loss_high, ms_ssim_loss, ffl_loss

    def forward(self, target, final_pred, intermediate_preds, halting_weights, **kwargs):
        T = len(intermediate_preds)
        B = target.shape[0]

        total_rec = 0.0
        total_charb = 0.0
        total_low = 0.0
        total_high = 0.0
        total_msssim = 0.0
        total_ffl = 0.0

        for t in range(T):
            rec_t, charb_t, low_t, high_t, msssim_t, ffl_t = self.reconstruction_loss(
                intermediate_preds[t], target
            )
            total_rec += rec_t
            total_charb += charb_t
            total_low += low_t
            total_high += high_t
            total_msssim += msssim_t
            total_ffl += ffl_t

        avg_rec = total_rec / T
        avg_charb = total_charb / T
        avg_low = total_low / T
        avg_high = total_high / T
        avg_msssim = total_msssim / T
        avg_ffl = total_ffl / T

        # KL regularization
        halting_mean = halting_weights.mean(dim=1)
        prior = torch.full_like(halting_mean, 1.0 / T)
        kl_loss = F.kl_div(torch.log(halting_mean + 1e-8), prior, reduction='batchmean')

        total_loss = avg_rec + self.prior_weight * kl_loss

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/rec": avg_rec.item(),
            "loss/charb": avg_charb.item(),
            "loss/low_freq": avg_low.item(),
            "loss/high_freq": avg_high.item(),
            "loss/ffl": avg_ffl.item(),
            "loss/ms_ssim": avg_msssim.item(),
            "loss/kl_gate": kl_loss.item(),
        }

        return total_loss, loss_dict