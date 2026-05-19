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


class Stage1Loss(nn.Module):

    def __init__(
        self,
        prior_weight=0.001,
        alpha=0.84
    ):
        super().__init__()
        self.prior_weight = prior_weight
        self.alpha = alpha
        self.charbonnier = CharbonnierLoss()

    def reconstruction_loss(self, pred, target):
        charbonnier_loss = self.charbonnier(pred, target)
        ms_ssim_loss = 1.0 - ms_ssim(
            pred, target, data_range=1.0, size_average=True
        )
        return (
            self.alpha * charbonnier_loss
            + (1.0 - self.alpha) * ms_ssim_loss
        )

    def forward(
        self,
        target,
        final_pred,
        intermediate_preds,
        halting_weights,
        **kwargs
    ):
        T = len(intermediate_preds)
        total_rec_loss = 0.0

        # =====================================
        # multi-step reconstruction
        # =====================================
        for t in range(T):
            pred_t = intermediate_preds[t]
            rec_loss_t = self.reconstruction_loss(pred_t, target)
            
            # Indexing trực tiếp trên tensor halting_weights
            total_rec_loss += (
                halting_weights[t].mean() * rec_loss_t
            )

        # =====================================
        # uniform prior KL
        # =====================================
        # Tạo prior cùng shape với halting_weights
        prior = torch.full_like(
            halting_weights,
            1.0 / T
        )
        
        # Không cần dùng torch.stack nữa
        kl_loss = F.kl_div(
            torch.log(halting_weights + 1e-8),
            prior,
            reduction='batchmean'
        )

        # =====================================
        # final
        # =====================================
        total_loss = (
            total_rec_loss
            + self.prior_weight * kl_loss
        )

        return total_loss