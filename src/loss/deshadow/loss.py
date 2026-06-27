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


class DeshadowLoss(nn.Module):
    def __init__(
        self,
        main_weight=1.0,        # Trọng số cho L1/Charbonnier trực tiếp trên ảnh
        freq_low_weight=1.25,   # Trọng số dải tần thấp
        freq_high_weight=0.65,  # Trọng số dải tần cao
        perceptual_weight=0.035,# Trọng số VGG
        color_weight=0.45,      # Trọng số LAB Color
        mask_weight=0.1,        # [MỚI] Trọng số cho Auxiliary Mask Loss của LoopViT
        trajectory_gamma=0.8,   # [MỚI] Hệ số suy giảm cho các bước lặp sớm (Discount factor)
    ):
        super().__init__()
        self.main_weight = main_weight
        self.freq_low_weight = freq_low_weight
        self.freq_high_weight = freq_high_weight
        self.perceptual_weight = perceptual_weight
        self.color_weight = color_weight
        self.mask_weight = mask_weight
        self.trajectory_gamma = trajectory_gamma

        self.charbonnier = CharbonnierLoss()
        self.freq_separator = FFTSeparator(cutoff_freq=25)
        self.vgg_loss = VGGPerceptualLoss()
        
        self.focused_color_loss_fn = LABColorLoss(
            weight_l=1.0, weight_a=1.5, weight_b=2.0, alpha=10.0
        )
        
        # Hàm loss cho phân loại bóng đổ tại Bottleneck
        self.mask_criterion = nn.CrossEntropyLoss()

    def image_reconstruction_loss(self, pred, input_img, target):
        """Tính tổng hợp các Loss khôi phục hình ảnh cho 1 bước dự đoán"""
        # 1. Main Spatial Loss (Mỏ neo chống sai lệch màu/sáng tổng thể)
        main_loss = self.charbonnier(pred, target, reduction='batchmean')

        # 2. Frequency Loss
        pred_low, pred_high = self.freq_separator(pred)
        target_low, target_high = self.freq_separator(target)
        low_loss = self.charbonnier(pred_low, target_low, reduction='batchmean')
        high_loss = self.charbonnier(pred_high, target_high, reduction='batchmean')
        
        # 3. VGG Perceptual Loss
        perceptual = self.vgg_loss(pred, target)
        
        # 4. Color Loss
        color_loss = self.focused_color_loss_fn(pred, input_img, target)

        total_rec_loss = (
            self.main_weight * main_loss +
            self.freq_low_weight * low_loss + 
            self.freq_high_weight * high_loss + 
            self.perceptual_weight * perceptual +
            self.color_weight * color_loss
        )
        
        return total_rec_loss, main_loss, color_loss

    def forward(self, input_img, target, intermediate_preds, bottleneck_logits_list):
        """
        Tham số mới:
        - intermediate_preds: List chứa ảnh dự đoán ở từng bước lặp
        - bottleneck_logits_list: List chứa (B, 2, H/8, W/8) xuất ra từ pred_head ở Bottleneck
        """
        T = len(intermediate_preds)
        
        # --- BƯỚC 1: TẠO PSEUDO MASK CHO BOTTLENECK ---
        with torch.no_grad():
            # Tính độ lệch giữa ảnh bóng và ảnh sạch gốc
            diff = torch.abs(input_img - target).mean(dim=1, keepdim=True)
            # Threshold > 0.05 coi là có bóng (Class 1), ngược lại là nền (Class 0)
            pseudo_mask = (diff > 0.05).float() 
            
            # Scale mask xuống kích thước của bottleneck (do Encoder dùng PixelUnshuffle)
            _, _, h_b, w_b = bottleneck_logits_list[0].shape
            target_mask = F.interpolate(pseudo_mask, size=(h_b, w_b), mode='nearest').long().squeeze(1)

        total_loss = 0.0
        acc_rec_loss = 0.0
        acc_mask_loss = 0.0
        
        # Biến theo dõi để log
        last_main = last_color = 0.0

        # --- BƯỚC 2: TÍNH LOSS TRÊN TOÀN BỘ QUỸ ĐẠO LẶP (TRAJECTORY) ---
        # Chúng ta dùng trọng số tăng dần (gamma^T-t) để nhấn mạnh vào các bước cuối
        weights = [self.trajectory_gamma ** (T - 1 - t) for t in range(T)]
        weight_sum = sum(weights)
        
        for t in range(T):
            w = weights[t] / weight_sum
            
            # A. Image Reconstruction Loss tại bước t
            rec_t, main_t, color_t = self.image_reconstruction_loss(intermediate_preds[t], input_img, target)
            
            # B. Auxiliary Mask Loss tại bước t
            mask_t = self.mask_criterion(bottleneck_logits_list[t], target_mask)
            
            total_loss += w * (rec_t + self.mask_weight * mask_t)
            
            acc_rec_loss += w * rec_t
            acc_mask_loss += w * mask_t
            
            if t == T - 1:
                last_main = main_t
                last_color = color_t

        loss_dict = {
            "loss/total": total_loss.item(),
            "loss/weighted_rec": acc_rec_loss.item(),
            "loss/weighted_mask": acc_mask_loss.item(),
            "eval/final_charbonnier": last_main.item(),
            "eval/final_color": last_color.item(),
        }

        return total_loss, loss_dict