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


class DocDeshadowLossStage1(nn.Module):
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

    # ---------------------------
    def recon_loss(self, pred, target):
        l_charb = self.charb(pred, target)
        l_ssim = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)

        total = l_charb + self.ssim_weight * l_ssim
        return total, l_charb, l_ssim

    # ---------------------------
    def forward(
        self,
        final_pred,
        target,
        intermediate_preds=None,
        exit_probs=None,   # (T, B)
    ):
        B = target.shape[0]

        # =====================================================
        # 1. FINAL LOSS
        # =====================================================
        final_total, final_charb, final_ssim = self.recon_loss(final_pred, target)

        # =====================================================
        # 2. LOOP LOSS (uniform supervision)
        # =====================================================
        if intermediate_preds is not None and len(intermediate_preds) > 0:
            T = len(intermediate_preds)

            rec_stack = []
            charb_stack = []
            ssim_stack = []

            for p in intermediate_preds:
                r, c, s = self.recon_loss(p, target)
                rec_stack.append(r)
                charb_stack.append(c)
                ssim_stack.append(s)

            rec_stack = torch.stack(rec_stack, dim=0)     # (T, B)
            charb_stack = torch.stack(charb_stack, dim=0)
            ssim_stack = torch.stack(ssim_stack, dim=0)

            # uniform weights (Stage 1 assumption)
            h = torch.ones_like(rec_stack) / T

            loop_rec = (h * rec_stack).sum(dim=0).mean()
            loop_charb = (h * charb_stack).sum(dim=0).mean()
            loop_ssim = (h * ssim_stack).sum(dim=0).mean()
        else:
            loop_rec = torch.tensor(0.0, device=target.device)
            loop_charb = torch.tensor(0.0, device=target.device)
            loop_ssim = torch.tensor(0.0, device=target.device)

        # =====================================================
        # 3. EXIT DISTRIBUTION REGULARIZATION (Stage 1 only)
        # =====================================================
        kl_loss = torch.tensor(0.0, device=target.device)
        entropy_loss = torch.tensor(0.0, device=target.device)

        if exit_probs is not None:
            # exit_probs = p_t (T, B)
            p = exit_probs + 1e-8

            # normalize over time axis
            p = p / p.sum(dim=0, keepdim=True)

            T = p.shape[0]
            uniform = torch.ones_like(p) / T

            # KL(p || uniform)
            kl_loss = F.kl_div(p.log(), uniform, reduction="batchmean")

            # entropy = -Σ p log p
            entropy_loss = -(p * p.log()).sum(dim=0).mean()

        # =====================================================
        # 4. TOTAL LOSS
        # =====================================================
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

            "loss/loop_rec": loop_rec.item(),
            "loss/loop_charb": loop_charb.item(),
            "loss/loop_ssim": loop_ssim.item(),

            "loss/kl": kl_loss.item(),
            "loss/entropy": entropy_loss.item(),
        }

        return loss, loss_dict

class DocDeshadowLossStage2(nn.Module):
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

    # -------------------------
    def recon_loss(self, pred, target):
        l_charb = self.charb(pred, target)
        l_ssim = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)

        total = l_charb + self.ssim_weight * l_ssim
        return total, l_charb, l_ssim

    # -------------------------
    def forward(
        self,
        final_pred,
        target,
        intermediate_preds=None,
        exit_probs=None,     # (T,B)
        halting_probs=None,   # (T,B)
    ):

        # =========================
        # FINAL LOSS
        # =========================
        final_total, final_charb, final_ssim = self.recon_loss(final_pred, target)

        # =========================
        # LOOP LOSS
        # =========================
        T = len(intermediate_preds)

        rec_stack = []
        charb_stack = []
        ssim_stack = []

        for p in intermediate_preds:
            r, c, s = self.recon_loss(p, target)
            rec_stack.append(r)
            charb_stack.append(c)
            ssim_stack.append(s)

        rec_stack = torch.stack(rec_stack, dim=0)
        charb_stack = torch.stack(charb_stack, dim=0)
        ssim_stack = torch.stack(ssim_stack, dim=0)

        # normalize halting distribution
        h = halting_probs
        h = h / (h.sum(dim=0, keepdim=True) + 1e-8)

        # weighted reconstruction (CORE LoopLM)
        loop_rec = (h * rec_stack).sum(dim=0).mean()
        loop_charb = (h * charb_stack).sum(dim=0).mean()
        loop_ssim = (h * ssim_stack).sum(dim=0).mean()

        # =========================
        # HALTING ALIGNMENT (KEY)
        # =========================

        kl_loss = torch.tensor(0.0, device=target.device)
        entropy_loss = torch.tensor(0.0, device=target.device)

        if halting_probs is not None:

            # ---- quality distribution (detach to stabilize)
            quality = rec_stack.detach()
            quality = quality / (quality.sum(dim=0, keepdim=True) + 1e-8)

            # KL: h || quality
            kl_loss = F.kl_div(h.log(), quality, reduction="batchmean")

            # entropy (avoid collapse)
            entropy_loss = -(h * (h + 1e-8).log()).sum(dim=0).mean()

        # =========================
        # TOTAL LOSS
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