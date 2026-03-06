import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import easyocr
import cv2
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

def extract_line_points(image_np, reader, num_points_per_line=10):
    try:
        results = reader.readtext(
            image_np,
            detail=1,
            paragraph=False,
            text_threshold=0.5,
            low_text=0.3
        )
        print(f"Debug: EasyOCR success - {len(results)} detections")
    except Exception as e:
        print(f"EasyOCR failed: {str(e)}")
        return None
    
    points = []
    for (bbox, text, prob) in results:
        if prob < 0.5: continue  # lọc nhiễu
        
        # Lấy 4 góc bounding box
        (tl, tr, br, bl) = bbox
        # Lấy baseline: từ tl → bl và tr → br, lấy midpoint
        mid_left = (tl[0] + bl[0])/2, (tl[1] + bl[1])/2
        mid_right = (tr[0] + br[0])/2, (tr[1] + br[1])/2
        
        # Sample num_points_per_line điểm đều trên baseline
        for t in np.linspace(0, 1, num_points_per_line):
            x = mid_left[0] + t * (mid_right[0] - mid_left[0])
            y = mid_left[1] + t * (mid_right[1] - mid_left[1])
            points.append([x / image_np.shape[1], y / image_np.shape[0]])  # normalize [0,1]

    if len(points) == 0:
        return None
    
    return torch.tensor(points, dtype=torch.float32)  # (N, 2)

def curvature_consistency_loss(
    pred_bm, gt_bm, 
    warped_img_np=None, 
    line_points=None, 
    ocr_reader=None, 
    eps=1e-4
):
    if line_points is None and warped_img_np is not None and ocr_reader is not None:
        line_points = extract_line_points(warped_img_np, reader=ocr_reader)

    if line_points is not None and len(line_points) > 10:
        # Sample tại sparse points (như code trước)
        grid = line_points.unsqueeze(0).unsqueeze(0).unsqueeze(0) * 2 - 1
        grid = grid.expand(pred_bm.shape[0], -1, -1, 2)

        pred_sample = F.grid_sample(pred_bm, grid, mode='bilinear', align_corners=True)
        gt_sample = F.grid_sample(gt_bm, grid, mode='bilinear', align_corners=True)

        # Curvature 1D
        def curv_1d(bm_seq):
            bm_seq = bm_seq.squeeze(2)  # (B, 2, N)
            dx = bm_seq[:, 0, 1:] - bm_seq[:, 0, :-1]
            dy = bm_seq[:, 1, 1:] - bm_seq[:, 1, :-1]
            ddx = dx[:, 1:] - dx[:, :-1]
            ddy = dy[:, 1:] - dy[:, :-1]
            ddx = F.pad(ddx, (0, 1))
            ddy = F.pad(ddy, (0, 1))
            num = torch.abs(dx * ddy - dy * ddx)
            den = (dx**2 + dy**2).pow(1.5) + eps
            return (num / den).mean(dim=-1).mean()

        return torch.abs(curv_1d(pred_sample) - curv_1d(gt_sample))
    
    # Fallback: approximation toàn map (như cũ)
    def compute_curvature(bm):
        dx = bm[:, :, :, 1:] - bm[:, :, :, :-1]
        dy = bm[:, :, 1:, :] - bm[:, :, :-1, :]
        ddx = dx[:, :, :, 1:] - dx[:, :, :, :-1]
        ddy = dy[:, :, 1:, :] - dy[:, :, :-1, :]
        ddx = F.pad(ddx, (0, 1, 0, 0))
        ddy = F.pad(ddy, (0, 0, 0, 1))
        num = torch.abs(dx * ddy - dy * ddx)
        den = (dx**2 + dy**2).pow(1.5) + eps
        return (num / den).mean()

    kappa_pred = compute_curvature(pred_bm)
    kappa_gt = compute_curvature(gt_bm)
    return torch.abs(kappa_pred - kappa_gt)

def ssim_loss(pred, target, data_range=1.0):
    """1 - SSIM loss"""
    return 1.0 - ssim(pred, target, data_range=data_range, size_average=True)

def pde_biharmonic_residual(u):
    def second_deriv(f, dim):
        # f: (B, C, H, W)
        if dim == 2:  # Đạo hàm bậc 2 theo Height (y)
            d = f[:, :, 2:, :] - 2 * f[:, :, 1:-1, :] + f[:, :, :-2, :]
            return F.pad(d, (0, 0, 1, 1)) # Pad 2 đầu Height
        elif dim == 3:  # Đạo hàm bậc 2 theo Width (x)
            d = f[:, :, :, 2:] - 2 * f[:, :, :, 1:-1] + f[:, :, :, :-2]
            return F.pad(d, (1, 1, 0, 0)) # Pad 2 đầu Width
        return 0

    ux = u[:, 0:1]
    uy = u[:, 1:2]

    # Laplacian ∇²u = u_xx + u_yy
    lap_ux = second_deriv(ux, 2) + second_deriv(ux, 3)
    lap_uy = second_deriv(uy, 2) + second_deriv(uy, 3)

    # Biharmonic ∇⁴u = ∇²(∇²u)
    laplap_ux = second_deriv(lap_ux, 2) + second_deriv(lap_ux, 3)
    laplap_uy = second_deriv(lap_uy, 2) + second_deriv(lap_uy, 3)

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
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        
    def forward(
        self,
        pred_img,
        gt_img,
        pred_bm,
        gt_bm,
        warped_img_np=None,
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
        l_curv = torch.tensor(0.0, device=pred_bm.device)
        if self.lambda_curv > 0:
            if line_points is not None:
                l_curv = curvature_consistency_loss(pred_bm, gt_bm_up, line_points=line_points)
            elif warped_img_np is not None:
                l_curv = curvature_consistency_loss(
                    pred_bm, gt_bm_up,
                    warped_img_np=warped_img_np,
                    ocr_reader=self.ocr_reader
                )

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