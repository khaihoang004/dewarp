import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import torchvision.models as models


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


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        
        self.slice1 = vgg[:4]
        self.slice2 = vgg[4:9]
        self.slice3 = vgg[9:16]
        
        for p in self.parameters():
            p.requires_grad = False
            
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def feature_l1(self, a, b):
        return F.l1_loss(a, b, reduction='none').mean(dim=[1, 2, 3])

    def forward(self, pred, target):
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        h_pred = self.slice1(pred)
        h_target = self.slice1(target)
        loss = self.feature_l1(h_pred, h_target)

        h_pred = self.slice2(h_pred)
        h_target = self.slice2(h_target)
        loss += self.feature_l1(h_pred, h_target)

        h_pred = self.slice3(h_pred)
        h_target = self.slice3(h_target)
        loss += self.feature_l1(h_pred, h_target)

        return loss


class FrequencySeparator(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0, channels=3):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.channels = channels
        kernel = self._get_gaussian_kernel2d(kernel_size, sigma, channels)
        self.register_buffer('gaussian_kernel', kernel)

    def _get_gaussian_kernel2d(self, kernel_size, sigma, channels):
        k = torch.arange(kernel_size).float()
        mu = (kernel_size - 1) / 2.0
        gauss1d = torch.exp(-((k - mu) ** 2) / (2 * sigma ** 2))
        gauss2d = gauss1d.unsqueeze(1) * gauss1d.unsqueeze(0)
        gauss2d = gauss2d / gauss2d.sum()
        gauss2d = gauss2d.view(1, 1, kernel_size, kernel_size).repeat(channels, 1, 1, 1)
        return gauss2d

    def forward(self, x):
        x_padded = F.pad(x, (self.padding, self.padding, self.padding, self.padding), mode='reflect')
        low_freq = F.conv2d(x_padded, self.gaussian_kernel, padding=0, groups=self.channels)
        high_freq = x - low_freq
        return low_freq, high_freq


class FFTSeparator(nn.Module):
    def __init__(self, cutoff_freq=20):
        super().__init__()
        self.cutoff_freq = cutoff_freq

    def forward(self, x):
        _, _, H, W = x.shape
        
        fft_x = torch.fft.fft2(x)
        fft_x_shifted = torch.fft.fftshift(fft_x, dim=(-2, -1))
        
        Y, X = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
        center_y, center_x = H // 2, W // 2
        dist_from_center = torch.sqrt((Y - center_y)**2 + (X - center_x)**2).to(x.device)
        
        mask = torch.exp(-(dist_from_center**2) / (2 * (self.cutoff_freq**2)))
        mask = mask.unsqueeze(0).unsqueeze(0) # [1, 1, H, W]
        
        low_fft = fft_x_shifted * mask
        
        low_fft_unshifted = torch.fft.ifftshift(low_fft, dim=(-2, -1))
        low_freq = torch.fft.ifft2(low_fft_unshifted).real
        
        high_freq = x - low_freq
        
        return low_freq, high_freq


class Stage1Loss(nn.Module):
    def __init__(
        self,
        # Core reconstruction
        trajectory_weight=1.0,
        low_weight=0.6,
        high_weight=1.45,
        perceptual_weight=0.035,
        
        # Regularization
        refine_weight=0.10,      # Đảm bảo loop cải thiện
        kl_weight=0.06,          # Giữ KL để chia đều halting weights
    ):
        super().__init__()
        self.trajectory_weight = trajectory_weight
        self.low_weight = low_weight
        self.high_weight = high_weight
        self.perceptual_weight = perceptual_weight
        self.refine_weight = refine_weight
        self.kl_weight = kl_weight

        self.charbonnier = CharbonnierLoss()
        self.freq_separator = FFTSeparator(cutoff_freq=25)
        self.vgg_loss = VGGPerceptualLoss()

    def reconstruction_loss(self, pred, target):
        """Charbonnier + Frequency + VGG"""
        pred_low, pred_high = self.freq_separator(pred)
        target_low, target_high = self.freq_separator(target)
        
        low_loss = self.charbonnier(pred_low, target_low, reduction='none')
        high_loss = self.charbonnier(pred_high, target_high, reduction='none')
        
        freq_loss = self.low_weight * low_loss + self.high_weight * high_loss

        perceptual = self.vgg_loss(pred, target)

        combined = freq_loss + self.perceptual_weight * perceptual
        return combined, freq_loss, perceptual

    def forward(self, target, final_pred, intermediate_preds, 
                halting_weights=None, **kwargs):
        
        T = len(intermediate_preds)

        # 1. Trajectory Weighted Reconstruction (Charbonnier + Freq + VGG)
        rec_losses = []
        for pred_t in intermediate_preds:
            rec_t, _, _ = self.reconstruction_loss(pred_t, target)
            rec_losses.append(rec_t)

        rec_losses = torch.stack(rec_losses, dim=0)   # (T, B)

        if halting_weights is not None:
            q = halting_weights / (halting_weights.sum(dim=0, keepdim=True) + 1e-8)
            weighted_rec = (q * rec_losses).sum(dim=0).mean()
        else:
            weighted_rec = rec_losses.mean(dim=0).mean()

        # 2. Refine Loss - Đảm bảo loop sau cải thiện
        refine_loss = 0.0
        if T > 1 and self.refine_weight > 0:
            for t in range(1, T):
                improvement = rec_losses[t-1] - rec_losses[t]
                refine_loss += F.relu(improvement).mean()
            refine_loss = refine_loss / (T - 1)

        # 3. KL Loss (giữ lại để model chia đều weight)
        kl_loss = 0.0
        if halting_weights is not None and self.kl_weight > 0:
            q = halting_weights / (halting_weights.sum(dim=0, keepdim=True) + 1e-8)
            uniform_prior = torch.ones_like(q) / T
            kl_loss = F.kl_div(q.log(), uniform_prior, reduction='batchmean')

        # === Tổng loss ===
        total_loss = (
            self.trajectory_weight * weighted_rec +
            self.refine_weight * refine_loss +
            self.kl_weight * kl_loss
        )

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/rec_weighted": weighted_rec.item(),
            "loss/refine": float(refine_loss),
            "loss/kl": float(kl_loss),
        }

        return total_loss, loss_dict
        

class Stage2Loss(nn.Module):
    def __init__(self, rltt_weight=0.1, ponder_weight=0.015, alpha=0.7):
        super().__init__()
        self.rltt_weight = rltt_weight
        self.ponder_weight = ponder_weight
        self.alpha = alpha
        self.charbonnier = CharbonnierLoss()

    def reconstruction_loss(self, pred, target):
        charb = self.charbonnier(pred, target, reduction='none')
        pred_clamp = torch.clamp(pred, 0.0, 1.0)
        ssim_val = ssim(pred_clamp, target, data_range=1.0, size_average=False)
        ssim_loss = 1.0 - ssim_val
        combined = self.alpha * charb + (1.0 - self.alpha) * ssim_loss
        return combined, charb, ssim_loss

    def forward(self, target, final_pred, intermediate_preds, halting_weights, halt_logits, **kwargs):
        T = len(intermediate_preds)
        
        step_losses = []
        avg_charb = avg_ssim = 0.0

        with torch.no_grad():
            for pred_t in intermediate_preds:
                rec_t, charb_t, ssim_t = self.reconstruction_loss(pred_t, target)
                step_losses.append(rec_t)
                avg_charb += charb_t.mean()
                avg_ssim += ssim_t.mean()

        avg_charb /= T
        avg_ssim /= T

        # RLTT Policy Loss
        exit_probs = [torch.sigmoid(l.view(-1)) for l in halt_logits]
        policy_loss = 0.0
        
        for t in range(T):
            curr_loss = step_losses[t]
            
            # Advantage
            if t == 0:
                advantage_t = torch.zeros_like(curr_loss)
            else:
                advantage_t = (step_losses[t-1] - curr_loss).detach()

            p_t = exit_probs[t].clamp(1e-6, 1.0 - 1e-6) # (B,)
            
            pos_mask = (advantage_t > 0).float()
            neg_mask = (advantage_t <= 0).float()
            
            # RLTT Logic cốt lõi: 
            # - Có tiến bộ (Advantage > 0) -> phạt p_t (kích thích vòng lặp đi tiếp)
            # - Kém đi hoặc bằng (Advantage <= 0) -> phạt (1 - p_t) (kích thích vòng lặp dừng ngay lập tức)
            loss_pos = advantage_t * (-torch.log(1 - p_t)) * pos_mask
            loss_neg = (-advantage_t) * (-torch.log(p_t)) * neg_mask
            
            policy_loss += (loss_pos + loss_neg).mean()

        policy_loss /= max(T - 1, 1)

        # Ponder loss truyền thống (phạt số bước dự kiến)
        expected_steps = torch.sum(
            (torch.arange(1, T+1, device=halting_weights.device).float().view(-1, 1) * halting_weights), 
            dim=0
        ).mean()

        # Tổng hợp Loss cho Stage 2
        total_loss = self.rltt_weight * policy_loss + self.ponder_weight * expected_steps

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/rltt_policy": policy_loss.item(),
            "loss/ponder": (self.ponder_weight * expected_steps).item(),
            "loss/expected_steps": expected_steps.item(),
            "eval/charbonnier": avg_charb.item(),
            "eval/ssim": avg_ssim.item(),
        }

        return total_loss, loss_dict