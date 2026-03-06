import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ms_ssim


class DewarpLoss(nn.Module):
    """
    Loss kết hợp:
    - L1 (pixel-wise)
    - 1 - MS-SSIM (structural similarity, rất tốt cho dewarping)
    """
    def __init__(self, alpha=1.0, beta=1.0, data_range=2.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.l1_loss = nn.L1Loss()
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        
        l1 = self.l1_loss(pred, target)
        
        msssim_val = ms_ssim(
            pred,
            target,
            data_range=self.data_range,
            size_average=True,
            win_size=11,
            channel=2
        )
        msssim_loss = 1.0 - msssim_val
        
        total_loss = self.alpha * l1 + self.beta * msssim_loss
        return total_loss


def build_loss(cfg):
    """
    Factory để tạo loss từ config.
    Bạn có thể thêm tham số alpha/beta vào config sau nếu muốn tune.
    """
    return DewarpLoss(
        alpha=getattr(cfg, 'loss_alpha', 1.0),
        beta=getattr(cfg, 'loss_beta', 1.0),
        data_range=2.0 if getattr(cfg, 'backward_map_range', 'tanh') == 'tanh' else 1.0
    )