import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return loss.mean()


class DocDeshadowLoss(nn.Module):
    def __init__(self, ssim_weight=0.4, loop_decay=0.9):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.ssim_weight = ssim_weight
        self.loop_decay = loop_decay

    def _image_loss(self, pred, target):
        """
        base reconstruction loss
        """
        l1 = self.charb(pred, target)
        ssim_loss = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
        return l1 + self.ssim_weight * ssim_loss

    def forward(self, final_pred, target, intermediate_preds=None, halting_weights=None):
        loss_final = self._image_loss(final_pred, target)

        loss_loop = 0.0

        if intermediate_preds is not None and len(intermediate_preds) > 0:
            T = len(intermediate_preds)

            losses = []
            for t, pred_t in enumerate(intermediate_preds):
                l = self._image_loss(pred_t, target)

                # decay theo timestep (giống idea loopLM nhẹ)
                weight = self.loop_decay ** t
                losses.append(weight * l)

            loss_loop = torch.stack(losses).mean()

        total_loss = loss_final + 0.5 * loss_loop

        loss_dict = {
            "loss_total": total_loss.item(),
            "loss_final": loss_final.item(),
            "loss_loop": float(loss_loop) if isinstance(loss_loop, float) else loss_loop.item(),
        }

        return total_loss, loss_dict