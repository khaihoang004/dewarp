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
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()


class DocDeshadowLoss(nn.Module):
    def __init__(
        self,
        ssim_weight=0.4,
        kl_weight=0.05,
        entropy_weight=0.01,
        loop_weight=0.5
    ):
        super().__init__()

        self.charb = CharbonnierLoss()
        self.ssim_weight = ssim_weight

        self.kl_weight = kl_weight
        self.entropy_weight = entropy_weight
        self.loop_weight = loop_weight

    # -------------------------------------------------
    def recon_loss(self, pred, target):
        l_charb = self.charb(pred, target)
        l_ssim = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)

        total = l_charb + self.ssim_weight * l_ssim

        return total, l_charb, l_ssim

    # -------------------------------------------------
    def forward(
        self,
        final_pred,
        target,
        intermediate_preds=None,
        exit_probs=None
    ):
        B = target.shape[0]

        # =========================
        # FINAL LOSS
        # =========================
        final_total, final_charb, final_ssim = self.recon_loss(final_pred, target)

        # =========================
        # LOOP LOSSES
        # =========================
        if intermediate_preds is not None and len(intermediate_preds) > 0:
            T = len(intermediate_preds)

            rec_list = []
            charb_list = []
            ssim_list = []

            for p in intermediate_preds:
                r, c, s = self.recon_loss(p, target)
                rec_list.append(r)
                charb_list.append(c)
                ssim_list.append(s)

            rec = torch.stack(rec_list, dim=0)
            charb = torch.stack(charb_list, dim=0)
            ssim_l = torch.stack(ssim_list, dim=0)

            # uniform aggregation (LoopLM base)
            weights = torch.ones_like(rec) / T

            loop_rec = (weights * rec).sum(dim=0).mean()
            loop_charb = (weights * charb).sum(dim=0).mean()
            loop_ssim = (weights * ssim_l).sum(dim=0).mean()
        else:
            loop_rec = torch.tensor(0.0, device=target.device)
            loop_charb = torch.tensor(0.0, device=target.device)
            loop_ssim = torch.tensor(0.0, device=target.device)

        # =========================
        # EXIT REGULARIZATION
        # =========================
        kl_loss = torch.tensor(0.0, device=target.device)
        entropy_loss = torch.tensor(0.0, device=target.device)

        if exit_probs is not None:
            # normalize distribution over loops
            p = exit_probs / (exit_probs.sum(dim=0, keepdim=True) + 1e-8)

            uniform = torch.ones_like(p) / p.shape[0]

            # KL(p || uniform)
            kl_loss = F.kl_div(p.log(), uniform, reduction="batchmean")

            # entropy penalty (prevent collapse / force distribution)
            entropy_loss = -(p * (p + 1e-8).log()).sum(dim=0).mean()

        # =========================
        # TOTAL
        # =========================
        loss = (
            final_total
            + self.loop_weight * loop_rec
            + self.kl_weight * kl_loss
            + self.entropy_weight * entropy_loss
        )

        loss_dict = {
            "loss/total": loss.item(),

            "loss/final_total": final_total.item(),
            "loss/final_charb": final_charb.item(),
            "loss/final_ssim": final_ssim.item(),

            "loss/loop_total": loop_rec.item(),
            "loss/loop_charb": loop_charb.item(),
            "loss/loop_ssim": loop_ssim.item(),

            "loss/kl": kl_loss.item(),
            "loss/entropy": entropy_loss.item(),
        }

        return loss, loss_dict