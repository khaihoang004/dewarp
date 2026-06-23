import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import torchvision.models as models
import kornia.color as kc


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target, reduction='mean'):
        loss = torch.sqrt(torch.clamp((pred - target) ** 2 + self.eps ** 2, min=1e-8))
        if reduction == 'mean':
            return torch.mean(loss)
        elif reduction == 'batchmean':
            return torch.mean(loss, dim=[1, 2, 3])
        return loss


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
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]

        low_fft = fft_x_shifted * mask

        low_fft_unshifted = torch.fft.ifftshift(low_fft, dim=(-2, -1))
        low_freq = torch.fft.ifft2(low_fft_unshifted).real

        high_freq = x - low_freq

        return low_freq, high_freq


class LABColorLoss(nn.Module):
    def __init__(self, weight_l=1.0, weight_a=1.5, weight_b=2.0, alpha=10.0):
        super().__init__()
        self.weight_l = weight_l
        self.weight_a = weight_a
        self.weight_b = weight_b
        
        self.alpha = alpha 
        self.l1 = nn.L1Loss(reduction='none')

    def _to_lab(self, img):
        return kc.rgb_to_lab(img.clamp(min=1e-5, max=1.0))

    def forward(self, pred, input_img, target):
        pred_lab = self._to_lab(pred)
        input_lab = self._to_lab(input_img)
        target_lab = self._to_lab(target)

        diff_lab = torch.abs(input_lab - target_lab).mean(dim=1, keepdim=True) / 100.0

        focus_weight = 1.0 - torch.exp(-self.alpha * diff_lab).detach()

        loss_l = (self.l1(pred_lab[:, 0:1, :, :], target_lab[:, 0:1, :, :]) * focus_weight).mean()
        loss_a = (self.l1(pred_lab[:, 1:2, :, :], target_lab[:, 1:2, :, :]) * focus_weight).mean()
        loss_b = (self.l1(pred_lab[:, 2:3, :, :], target_lab[:, 2:3, :, :]) * focus_weight).mean()

        total_loss = (self.weight_l * loss_l + self.weight_a * loss_a + self.weight_b * loss_b) / 100.0

        return total_loss


class Stage1Loss(nn.Module):
    def __init__(
        self,
        trajectory_weight=1.0,
        low_weight=1.25,
        high_weight=0.65,
        perceptual_weight=0.035,
        color_weight=0.45,
        refine_weight=2.0,
        kl_weight=0.06,
    ):
        super().__init__()
        self.trajectory_weight = trajectory_weight
        self.low_weight = low_weight
        self.high_weight = high_weight
        self.perceptual_weight = perceptual_weight
        self.color_weight = color_weight
        self.refine_weight = refine_weight
        self.kl_weight = kl_weight

        self.charbonnier = CharbonnierLoss()
        self.freq_separator = FFTSeparator(cutoff_freq=25)
        self.vgg_loss = VGGPerceptualLoss()
        
        self.focused_color_loss_fn = LABColorLoss(
            weight_l=1.0,
            weight_a=1.5,
            weight_b=2.0,
            alpha=10.0
        )

    def reconstruction_loss(self, pred, input_img, target):
        pred_low, pred_high = self.freq_separator(pred)
        target_low, target_high = self.freq_separator(target)

        # 1. Frequency
        low_loss = self.charbonnier(pred_low, target_low, reduction='batchmean')
        high_loss = self.charbonnier(pred_high, target_high, reduction='batchmean')
        freq_loss = self.low_weight * low_loss + self.high_weight * high_loss
        
        # 2. VGG
        perceptual = self.vgg_loss(pred, target)
        structural_loss = freq_loss + self.perceptual_weight * perceptual 
        
        # 3.Color Loss
        color_loss = self.focused_color_loss_fn(pred, input_img, target)

        return structural_loss, low_loss, high_loss, color_loss

    def forward(self, input_img, target, final_pred, intermediate_preds, halting_weights=None, **kwargs):
        T = len(intermediate_preds)

        rec_losses = []
        low_losses = []
        high_losses = []
        color_losses = []

        for pred_t in intermediate_preds:
            rec_t, low_t, high_t, color_t = self.reconstruction_loss(pred_t, input_img, target)
            rec_losses.append(rec_t)
            low_losses.append(low_t)
            high_losses.append(high_t)
            color_losses.append(color_t)

        rec_losses     = torch.stack(rec_losses, dim=0)   # [T, B]
        low_losses     = torch.stack(low_losses, dim=0)
        high_losses    = torch.stack(high_losses, dim=0)
        color_losses   = torch.stack(color_losses, dim=0)

        if halting_weights is not None:
            if halting_weights.dim() == 2:
                q = halting_weights / (halting_weights.sum(dim=0, keepdim=True) + 1e-8)  # [T, B]
            else:
                q = halting_weights

            weighted_rec     = (q * rec_losses).sum(dim=0).mean()
            weighted_low     = (q * low_losses).sum(dim=0).mean()
            weighted_high    = (q * high_losses).sum(dim=0).mean()
            weighted_color   = (q * color_losses).sum(dim=0).mean()
        else:
            weighted_rec     = rec_losses.mean()
            weighted_low     = low_losses.mean()
            weighted_high    = high_losses.mean()
            weighted_color   = color_losses.mean()

        refine_loss = 0.0
        if T > 1 and self.refine_weight > 0:
            for t in range(1, T):
                improvement = rec_losses[t] - rec_losses[t - 1]
                refine_loss += F.relu(improvement).mean()
            refine_loss = refine_loss / (T - 1)

        kl_loss = 0.0
        if halting_weights is not None and self.kl_weight > 0:
            q = halting_weights / (halting_weights.sum(dim=0, keepdim=True) + 1e-8)
            q = q.clamp(min=1e-8)
            uniform_prior = torch.ones_like(q) / T
            kl_loss = F.kl_div(q.log(), uniform_prior, reduction='batchmean')

        total_loss = (
            self.trajectory_weight * weighted_rec +
            self.color_weight * weighted_color +
            self.refine_weight * refine_loss +
            self.kl_weight * kl_loss
        )

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/rec_structural": weighted_rec.item(),
            "loss/freq_low": weighted_low.item(),
            "loss/freq_high": weighted_high.item(),
            "loss/color_lab": weighted_color.item(),
            "loss/refine": float(refine_loss) if isinstance(refine_loss, float) else refine_loss.item(),
            "loss/kl": float(kl_loss) if isinstance(kl_loss, float) else kl_loss.item(),
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
        charb = self.charbonnier(pred, target, reduction='batchmean')
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
        exit_probs = [torch.sigmoid(halt_logits[t].view(-1)) for t in range(T)]
        policy_loss = 0.0

        for t in range(T):
            curr_loss = step_losses[t]

            if t == 0:
                advantage_t = torch.zeros_like(curr_loss)
            else:
                advantage_t = (step_losses[t - 1] - curr_loss).detach()

            p_t = exit_probs[t].clamp(1e-6, 1.0 - 1e-6)  # (B,)

            pos_mask = (advantage_t > 0).float()
            neg_mask = (advantage_t <= 0).float()

            loss_pos = advantage_t * (-torch.log(1 - p_t)) * pos_mask
            loss_neg = (-advantage_t) * (-torch.log(p_t)) * neg_mask

            policy_loss += (loss_pos + loss_neg).mean()

        policy_loss /= max(T - 1, 1)

        # Ponder loss
        expected_steps = torch.sum(
            (torch.arange(1, T + 1, device=halting_weights.device).float().view(-1, 1) * halting_weights),
            dim=0
        ).mean()

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