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
        return F.l1_loss(
            a,
            b,
            reduction='none'
        ).mean(dim=[1, 2, 3])

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
    def __init__(self, kernel_size=5):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2

    def forward(self, x, grayscale_high=True):
        low_freq = F.avg_pool2d(x, kernel_size=self.kernel_size, stride=1, padding=self.padding)
        if grayscale_high:
            gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
            gray_low = F.avg_pool2d(gray, kernel_size=self.kernel_size, stride=1, padding=self.padding)
            high_freq = gray - gray_low
        else:
            high_freq = x - low_freq
        return low_freq, high_freq

class Stage1Loss(nn.Module):
    def __init__(
        self,
        entropy_weight=0.05,
        entropy_warmdown_epochs=10,

        refine_weight=0.05,
        distillation_weight=0.35,      
        freq_distill_weight=0.25,      
        freq_kernel=5,                 
        
        # frequency
        low_weight=0.5,
        high_weight=1.2,
        
        # global weights
        w_spatial=1.0,
        w_freq=1.0,
        w_ssim=0.05,
        w_perceptual=0.02,
        ssim_warmup_epochs=5,
    ):
        super().__init__()

        self.entropy_weight = entropy_weight
        self.entropy_warmdown_epochs = entropy_warmdown_epochs
        self.refine_weight = refine_weight
        self.distillation_weight = distillation_weight
        self.freq_distill_weight = freq_distill_weight

        self.low_weight = low_weight
        self.high_weight = high_weight

        self.w_spatial = w_spatial
        self.w_freq = w_freq
        self.w_ssim = w_ssim
        self.w_perceptual = w_perceptual

        self.ssim_warmup_epochs = ssim_warmup_epochs

        self.charbonnier = CharbonnierLoss()
        self.freq_separator = FrequencySeparator(kernel_size=freq_kernel)
        self.vgg_loss = VGGPerceptualLoss()

    def get_entropy_weight(self, current_epoch):
        min_weight = self.entropy_weight * 0.1
        if self.entropy_warmdown_epochs <= 0:
            return self.entropy_weight
            
        progress = min(current_epoch / self.entropy_warmdown_epochs, 1.0)
        return min_weight + (self.entropy_weight - min_weight) * (1.0 - progress)

    # SSIM Warmup
    def get_ssim_weight(self, current_epoch):
        if self.ssim_warmup_epochs <= 0:
            return self.w_ssim
        progress = min(current_epoch / self.ssim_warmup_epochs, 1.0)
        return progress * self.w_ssim

    def reconstruction_loss(self, pred, target, current_epoch=0):

        # Direct spatial reconstruction - Spatial Loss
        spatial_loss = self.charbonnier(pred, target, reduction='none')
        
        # Frequency Loss
        pred_low, pred_high = self.freq_separator(pred)
        target_low, target_high = self.freq_separator(target)
        loss_low = self.charbonnier(pred_low, target_low, reduction='none')
        loss_high = self.charbonnier(pred_high, target_high, reduction='none')
        freq_loss = (self.low_weight * loss_low + self.high_weight * loss_high)

        # VGG Perceptual Loss
        perceptual_loss = self.vgg_loss(pred, target)

        # SSIM Loss
        pred_safe = torch.clamp(pred, 0.0, 1.0)
        target_safe = torch.clamp(target, 0.0, 1.0)
        ssim_loss = 1.0 - ssim(pred_safe, target_safe, data_range=1.0, size_average=False)
        if ssim_loss.ndim > 1:
            ssim_loss = ssim_loss.mean(dim=tuple(range(1, ssim_loss.ndim)))

        current_ssim_weight = self.get_ssim_weight(current_epoch)

        combined = (
            self.w_spatial * spatial_loss
            + self.w_freq * freq_loss
            + self.w_perceptual * perceptual_loss
            + current_ssim_weight * ssim_loss
        )

        return (
            combined, 
            spatial_loss, 
            loss_low,
            loss_high, 
            perceptual_loss, 
            ssim_loss, 
            current_ssim_weight,
        )

    def compute_progressive_ilsd(self, intermediate_preds):
        if len(intermediate_preds) < 2:
            return 0.0
            
        distill_loss = 0.0
        T = len(intermediate_preds)
        
        for t in range(T - 1):
            student = intermediate_preds[t]
            teacher = intermediate_preds[t + 1].detach()
            
            spatial_dist = F.l1_loss(student, teacher)
            
            _, student_high = self.freq_separator(student)
            _, teacher_high = self.freq_separator(teacher)
            high_dist = F.l1_loss(student_high, teacher_high)
            
            weight = (t + 1) / T
            distill_loss += weight * (spatial_dist + self.freq_distill_weight * high_dist)
        
        return distill_loss / max(T - 1, 1)

    def forward(self, target, final_pred, intermediate_preds, halting_weights, current_epoch=0, **kwargs):
        T = len(intermediate_preds)

        rec_losses = []
        spatial_losses = []

        total_spatial = 0.0
        total_low = 0.0
        total_high = 0.0
        total_perceptual = 0.0
        total_ssim = 0.0
        current_ssim_weight = 0.0

        for t in range(T):
            (rec_t, spatial_t, low_t, high_t, perceptual_t, ssim_t, curr_ssim_w) = self.reconstruction_loss(
                intermediate_preds[t], target, current_epoch
            )
            rec_losses.append(rec_t)
            spatial_losses.append(spatial_t)

            total_spatial += spatial_t
            total_low += low_t
            total_high += high_t
            total_perceptual += perceptual_t
            total_ssim += ssim_t

        rec_losses = torch.stack(rec_losses, dim=0)   # (T, B)
        spatial_losses = torch.stack(spatial_losses, dim=0)   # (T, B)

        q = halting_weights / (halting_weights.sum(dim=0, keepdim=True) + 1e-8)
        weighted_rec = torch.sum(q * rec_losses, dim=0).mean()

        refine_loss = 0.0
        for t in range(T - 1):
            degradation = spatial_losses[t + 1] - spatial_losses[t] 
            refine_loss += F.relu(degradation).mean()
        refine_loss /= max(T - 1, 1)

        ilsd_loss = self.compute_progressive_ilsd(intermediate_preds)

        # KL Regularization for Ponder Gate
        # KL(Halting || Uniform)
        hw_b = q.transpose(0, 1) # (T, B) -> (B, T)
        
        kl_loss = torch.sum(
            hw_b * (torch.log(hw_b + 1e-8) - math.log(1.0 / T)), 
            dim=1
        ).mean()
        current_entropy_weight = self.get_entropy_weight(current_epoch)


        total_loss = (
            weighted_rec
            + self.refine_weight * refine_loss
            + self.distillation_weight * ilsd_loss
            + current_entropy_weight * kl_loss
        )

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/rec": weighted_rec.item(),
            "loss/refine": refine_loss.item(),
            "loss/ilsd": ilsd_loss.item(),
            "loss/kl_gate": kl_loss.item(),
            "loss/kl_weight": current_entropy_weight,
            
            "loss/spatial": (total_spatial / T).mean().item(),
            "loss/low_freq": (total_low / T).mean().item(),
            "loss/high_freq": (total_high / T).mean().item(),
            "loss/ssim": (total_ssim / T).mean().item(),
            
            "debug/q_mean": q.mean().item(),
            "debug/q_max": q.max().item(),
        }

        return total_loss, loss_dict


class Stage2Loss(nn.Module):
    def __init__(self, gain_threshold=0.003, gain_scale=12.0, ponder_weight=0.015, alpha=0.7):
        super().__init__()
        self.gain_threshold = gain_threshold
        self.gain_scale = gain_scale
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
        B = target.shape[0]

        step_losses = []
        avg_charb = 0.0
        avg_ssim = 0.0

        with torch.no_grad():
            for pred_t in intermediate_preds:
                rec_t, charb_t, ssim_t = self.reconstruction_loss(pred_t, target)
                step_losses.append(rec_t)
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
                gain = prev_loss - curr_loss # Shape: (B,)
                utility = torch.sigmoid(self.gain_scale * (gain - self.gain_threshold))
                target_prob = (1.0 - utility).detach() # Shape: (B,)

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