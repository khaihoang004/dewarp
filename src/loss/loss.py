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
        low_freq = F.conv2d(x, self.gaussian_kernel, padding=self.padding, groups=self.channels)
        high_freq = x - low_freq
        return low_freq, high_freq


class Stage1Loss(nn.Module):
    def __init__(
        self,
        rltt_trajectory_weight=1.0,
        consistency_weight=0.12,
        distillation_weight=0.25,
        freq_distill_weight=0.18,
        entropy_weight=0.035,
        kl_weight=0.04,
        rltt_policy_weight=0.06,
        low_weight=0.55,
        high_weight=1.45,
        w_freq=1.0,
        w_perceptual=0.03,
        w_ssim=0.07,
        ssim_warmup_epochs=6,
    ):
        super().__init__()
        self.rltt_trajectory_weight = rltt_trajectory_weight
        self.consistency_weight = consistency_weight
        self.distillation_weight = distillation_weight
        self.freq_distill_weight = freq_distill_weight
        self.entropy_weight = entropy_weight
        self.kl_weight = kl_weight
        self.rltt_policy_weight = rltt_policy_weight

        self.low_weight = low_weight
        self.high_weight = high_weight
        self.w_freq = w_freq
        self.w_perceptual = w_perceptual
        self.w_ssim = w_ssim
        self.ssim_warmup_epochs = ssim_warmup_epochs

        self.charbonnier = CharbonnierLoss()
        self.freq_separator = FrequencySeparator(kernel_size=5)
        self.vgg_loss = VGGPerceptualLoss()

    def get_ssim_weight(self, current_epoch):
        if self.ssim_warmup_epochs <= 0:
            return self.w_ssim
        progress = min(current_epoch / self.ssim_warmup_epochs, 1.0)
        return progress * self.w_ssim

    def reconstruction_loss(self, pred, target, current_epoch=0):
        pred_low, pred_high = self.freq_separator(pred)
        target_low, target_high = self.freq_separator(target)
        
        low_loss = self.charbonnier(pred_low, target_low, reduction='none')
        high_loss = self.charbonnier(pred_high, target_high, reduction='none')
        
        freq_loss = self.low_weight * low_loss + self.high_weight * high_loss

        perceptual = self.vgg_loss(pred, target)

        pred_safe = torch.clamp(pred, 0., 1.)
        target_safe = torch.clamp(target, 0., 1.)
        ssim_loss = 1.0 - ssim(pred_safe, target_safe, data_range=1.0, size_average=False)
        if ssim_loss.ndim > 1:
            ssim_loss = ssim_loss.mean(dim=tuple(range(1, ssim_loss.ndim)))

        current_ssim_w = self.get_ssim_weight(current_epoch)

        combined = (
            self.w_freq * freq_loss +
            self.w_perceptual * perceptual +
            current_ssim_w * ssim_loss
        )
        
        return combined, freq_loss, low_loss, high_loss, perceptual, ssim_loss

    def forward(self, target, final_pred, intermediate_preds, 
                halting_weights=None, exit_probs=None, current_epoch=0, **kwargs):
        
        T = len(intermediate_preds)

        # 1. Trajectory Weighted Reconstruction
        rec_losses = []
        for pred_t in intermediate_preds:
            rec_t, _, _, _, _, _ = self.reconstruction_loss(pred_t, target, current_epoch)
            rec_losses.append(rec_t)

        rec_losses = torch.stack(rec_losses, dim=0)   # (T, B)

        if halting_weights is not None:
            q = halting_weights / (halting_weights.sum(dim=0, keepdim=True) + 1e-8)
            weighted_rec = (q * rec_losses).sum(dim=0).mean()
        else:
            weighted_rec = rec_losses.mean(dim=0).mean()

        # 2. Consistency Loss
        consistency_loss = 0.0
        if T > 1 and self.consistency_weight > 0:
            for t in range(1, T):
                consistency_loss += F.mse_loss(intermediate_preds[t], intermediate_preds[t-1].detach())
            consistency_loss /= (T - 1)

        # 3. Progressive Distillation
        ilsd_loss = 0.0
        if T > 1 and self.distillation_weight > 0:
            for t in range(T - 1):
                stu = intermediate_preds[t]
                tea = intermediate_preds[t + 1].detach()
                spatial_d = F.l1_loss(stu, tea)
                _, sh = self.freq_separator(stu)
                _, th = self.freq_separator(tea)
                high_d = F.l1_loss(sh, th)
                weight = (t + 1) / T
                ilsd_loss += weight * (spatial_d + self.freq_distill_weight * high_d)
            ilsd_loss /= (T - 1)

        # 4. Entropy
        entropy_loss = 0.0
        if exit_probs is not None and self.entropy_weight > 0:
            p_mean = exit_probs.mean(dim=0)
            entropy_loss = - (p_mean * torch.log(p_mean + 1e-8)).mean()

        # 5. RLTT Policy
        rltt_policy_loss = 0.0
        if exit_probs is not None and self.rltt_policy_weight > 0 and T > 1:
            for t in range(1, T):
                adv = (rec_losses[t-1] - rec_losses[t]).detach()
                p_t = exit_probs[t].clamp(1e-6, 1.0 - 1e-6)
                loss_pos = adv * (-torch.log(1 - p_t)) * (adv > 0).float()
                loss_neg = (-adv) * (-torch.log(p_t)) * (adv <= 0).float()
                rltt_policy_loss += (loss_pos + loss_neg).mean()
            rltt_policy_loss /= (T - 1)

        # 6. KL
        kl_loss = 0.0
        if halting_weights is not None and self.kl_weight > 0:
            q = halting_weights / (halting_weights.sum(dim=0, keepdim=True) + 1e-8)
            uniform_prior = torch.ones_like(q) / T
            kl_loss = F.kl_div(q.log(), uniform_prior, reduction='batchmean')

        total_loss = (
            self.rltt_trajectory_weight * weighted_rec +
            self.consistency_weight * consistency_loss +
            self.distillation_weight * ilsd_loss +
            self.entropy_weight * entropy_loss +
            self.rltt_policy_weight * rltt_policy_loss +
            self.kl_weight * kl_loss
        )

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/rec_weighted": weighted_rec.item(),
            "loss/consistency": consistency_loss.item(),
            "loss/ilsd": ilsd_loss.item(),
            "loss/entropy": entropy_loss.item(),
            "loss/rltt_policy": rltt_policy_loss.item(),
            "loss/kl": kl_loss.item(),
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