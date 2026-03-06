# src/loss/loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim  # hoặc ms_ssim nếu bạn dùng multi-scale

def charbonnier_loss(pred, target, epsilon=1e-3):
    """Robust L1 loss (Charbonnier)"""
    diff = pred - target
    return torch.mean(torch.sqrt(diff**2 + epsilon**2))


def axis_aligned_geometric_loss(pred_bm, target_bm, weight=0.1):
    """
    Loss khuyến khích backward map giữ tính thẳng hàng theo trục (axis-aligned).
    Thường dùng trong dewarping để tránh méo mó lưới.
    """
    # Gradient theo x và y
    dx_pred = pred_bm[:, :, :, 1:] - pred_bm[:, :, :, :-1]
    dy_pred = pred_bm[:, :, 1:, :] - pred_bm[:, :, :-1, :]
    
    dx_tgt = target_bm[:, :, :, 1:] - target_bm[:, :, :, :-1]
    dy_tgt = target_bm[:, :, 1:, :] - target_bm[:, :, :-1, :]
    
    loss_x = F.l1_loss(dx_pred, dx_tgt)
    loss_y = F.l1_loss(dy_pred, dy_tgt)
    
    return weight * (loss_x + loss_y)


def ssim_loss(pred, target, data_range=1.0):
    """1 - SSIM loss"""
    return 1.0 - ssim(pred, target, data_range=data_range, size_average=True)


def unwarp(img, bm):
    """
    Unwarp ảnh warped bằng backward map bm.
    bm: (B, 2, H, W) - normalized grid [-1, 1] hoặc [0, 1] tùy dataset
    img: (B, 3, H, W)
    Trả về ảnh được warp ngược (straightened image)
    """
    # Tạo grid từ backward map
    grid = bm.permute(0, 2, 3, 1)  # (B, H, W, 2)
    
    # Nếu bm ở range [-1,1], giữ nguyên
    # Nếu bm ở range [0,1], scale về [-1,1]: grid = grid * 2 - 1
    
    # Giả sử bm đã ở [-1,1] (từ model tanh)
    return F.grid_sample(
        img,
        grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
class Doc3DDewarpLoss(nn.Module):
    def __init__(self, bm_recon_weight=1.0, axis_weight=0.1, re_weight=1.0, ssim_weight=0.1):
        super().__init__()
        self.bm_recon_weight = bm_recon_weight
        self.axis_weight = axis_weight
        self.re_weight = re_weight
        self.ssim_weight = ssim_weight

    def forward(self, pred_bm, bm, warped_img):
        """
        pred_bm: dự đoán backward map từ model (B, 2, H_img, W_img)
        bm: target backward map từ dataset (B, 2, H_bm, W_bm) → sẽ resize lên
        warped_img: ảnh input bị warp (B, 3, H_img, W_img)
        """
        # Resize target bm lên size của pred_bm
        bm_resized = F.interpolate(
            bm,
            size=pred_bm.shape[2:],
            mode='bilinear',
            align_corners=True
        )

        # BM loss
        bm_recon_loss = charbonnier_loss(pred_bm, bm_resized)
        bm_axis_loss = self.axis_weight * axis_aligned_geometric_loss(pred_bm, bm_resized)
        bm_loss = bm_recon_loss + bm_axis_loss

        # Restoration loss
        pred_unwarped = unwarp(warped_img, pred_bm)
        target_unwarped = unwarp(warped_img, bm_resized)
        
        img_recon_loss = charbonnier_loss(pred_unwarped, target_unwarped)
        img_ssim_loss = self.ssim_weight * ssim_loss(pred_unwarped, target_unwarped)
        
        re_loss = img_recon_loss + img_ssim_loss

        # Tổng loss (có thể điều chỉnh trọng số)
        total_loss = self.bm_recon_weight * bm_loss + self.re_weight * re_loss
        
        # Optional: trả về chi tiết để log
        return total_loss, {
            'bm_loss': bm_loss.item(),
            'bm_recon': bm_recon_loss.item(),
            'bm_axis': bm_axis_loss.item(),
            're_loss': re_loss.item(),
            'img_recon': img_recon_loss.item(),
            'img_ssim': img_ssim_loss.item()
        }