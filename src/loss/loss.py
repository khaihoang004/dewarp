import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from pytorch_msssim import ssim

class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=1e-3):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt(diff**2 + self.epsilon**2))

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



def curvature_consistency_loss(pred_bm, gt_bm, line_points, eps=1e-4):
    """
    Curvature loss như Eq.5-6.
    line_points: tensor (N, 2) hoặc list points trên line elements (giả sử đã project)
    Thực tế cần bilinear sample tại points → dùng grid_sample hoặc interpolate
    Ở đây giả lập đơn giản (thay bằng implement chính xác nếu có line_points thật)
    """
    # Giả lập: flatten và tính curvature trên toàn map (approximation cho test)
    # Thực tế: sample tại line points như paper
    def compute_curvature(bm):
        # Central difference cho derivative
        dx = bm[:, :, :, 1:] - bm[:, :, :, :-1]
        dy = bm[:, :, 1:, :] - bm[:, :, :-1, :]
        ddx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
        ddy = dy[:, :, 1:, :] - dy[:, :, :-1, :]

        # Pad để khớp size
        ddx = F.pad(ddx, (0, 1, 0, 0))
        ddy = F.pad(ddy, (0, 0, 0, 1))

        num = torch.abs(dx * ddy - dy * ddx)
        den = (dx**2 + dy**2).pow(1.5) + eps
        kappa = num / den
        return kappa.mean()

    kappa_pred = compute_curvature(pred_bm)
    kappa_gt = compute_curvature(gt_bm)
    return torch.abs(kappa_pred - kappa_gt)

def ssim_loss(pred, target, data_range=1.0):
    """1 - SSIM loss"""
    return 1.0 - ssim(pred, target, data_range=data_range, size_average=True)

def pde_biharmonic_residual(u):
    """
    Tính residual của biharmonic equation: ∇⁴u ≈ 0
    u: displacement field (B, 2, H, W) - backward map
    Trả về mean squared residual
    """
    def second_deriv(f, dim):
        # Central difference cho đạo hàm bậc 2
        if dim == 2:  # theo height
            d1 = f[:, :, 2:, :] - 2 * f[:, :, 1:-1, :] + f[:, :, :-2, :]
            d2 = f[:, :, :, 2:] - 2 * f[:, :, :, 1:-1] + f[:, :, :, :-2]
        else:  # theo width
            d1 = f[:, :, :, 2:] - 2 * f[:, :, :, 1:-1] + f[:, :, :, :-2]
            d2 = f[:, :, 2:, :] - 2 * f[:, :, 1:-1, :] + f[:, :, :-2, :]
        # Pad để khớp size
        pad = (0, 0, 0, 0) if dim == 2 else (1, 1, 0, 0)
        return F.pad(d1, pad), F.pad(d2, pad)

    ux = u[:, 0:1]  # dx channel
    uy = u[:, 1:2]  # dy channel

    # Laplacian ∇²u
    lap_ux_x, lap_ux_y = second_deriv(ux, 2), second_deriv(ux, 3)
    lap_uy_x, lap_uy_y = second_deriv(uy, 2), second_deriv(uy, 3)
    lap_ux = lap_ux_x + lap_ux_y
    lap_uy = lap_uy_x + lap_uy_y

    # ∇⁴u = ∇²(∇²u)
    laplap_ux_x, laplap_ux_y = second_deriv(lap_ux, 2), second_deriv(lap_ux, 3)
    laplap_uy_x, laplap_uy_y = second_deriv(lap_uy, 2), second_deriv(lap_uy, 3)
    laplap_ux = laplap_ux_x + laplap_ux_y
    laplap_uy = laplap_uy_x + laplap_uy_y

    residual = laplap_ux**2 + laplap_uy**2
    return torch.mean(residual)

class DewarpLoss(nn.Module):
    def __init__(
        self,
        lambda_recon=1.0,
        lambda_bm_recon=1.0,
        lambda_ssim=0.1,
        lambda_axis=0.1,
        lambda_curv=0.05,
        lambda_pde=0.01,

    ):
        super().__init__()
        
        self.lambda_recon = lambda_recon
        self.lambda_bm_recon = lambda_bm_recon
        self.lambda_ssim = lambda_ssim
        self.lambda_axis = lambda_axis
        self.lambda_curv = lambda_curv
        self.lambda_pde = lambda_pde

        self.charbonnier = CharbonnierLoss()

    def forward(
        self,
        pred_img,
        gt_img,
        pred_bm,
        gt_bm,
        line_points=None
    ):

        # Reconstruction loss
        l_recon = self.charbonnier(pred_img, gt_img)

        # SSIM loss
        l_ssim = 1 - ssim(pred_img, gt_img, data_range=1.0, size_average=True)

        # BM reconstruction
        gt_bm_up = F.interpolate(
            gt_bm,
            size=pred_bm.shape[2:],
            mode="bilinear",
            align_corners=True
        )

        l_bm = self.charbonnier(pred_bm, gt_bm_up)

        # Axis aligned
        l_axis = axis_aligned_geometric_loss(pred_bm, gt_bm_up)

        # PDE smoothness
        l_pde = pde_biharmonic_residual(pred_bm)

        # Curvature
        l_curv = 0
        if line_points is not None:
            l_curv = curvature_consistency_loss(pred_bm, gt_bm_up, line_points)

        # Total loss
        total = (
            self.lambda_recon * l_recon
            + self.lambda_ssim * l_ssim
            + self.lambda_bm_recon * l_bm
            + self.lambda_axis * l_axis
            + self.lambda_pde * l_pde
            + self.lambda_curv * l_curv
        )

        return total, {
            "recon": l_recon.item(),
            "ssim": l_ssim.item(),
            "bm": l_bm.item(),
            "axis": l_axis.item(),
            "pde": l_pde.item(),
            "curv": float(l_curv)
        }

def build_loss():
    return DewarpLoss(
        lambda_recon=1.0,
        lambda_ssim=0.1,
        lambda_bm_recon=1.0,
        lambda_axis=0.1,
        lambda_pde=0.01,
        lambda_curv=0.05
    )