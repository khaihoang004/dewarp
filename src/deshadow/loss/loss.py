import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import kornia.color as kc

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.sqrt(diff * diff + self.eps * self.eps).mean()

class LABColorLoss(nn.Module):
    def __init__(self, weight_l=1.0, weight_a=1.5, weight_b=2.0, alpha=10.0):
        super().__init__()
        self.weight_l = weight_l
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.alpha = alpha 
        self.l1 = nn.L1Loss(reduction='none')

    def _to_lab(self, img):
        # Kornia yêu cầu input [0, 1]
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

        return (self.weight_l * loss_l + self.weight_a * loss_a + self.weight_b * loss_b) / 100.0


class LoopDocEnhanceLoss(nn.Module):
    def __init__(
        self,
        ssim_weight=0.4,
        color_weight=0.45,
        kl_weight=0.05,
        loop_weight=0.25
    ):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.color = LABColorLoss()
        
        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.kl_weight = kl_weight
        self.loop_weight = loop_weight

    def recon_loss(self, pred, input_img, target):
        """Tính toán tổng hợp các metric cho 1 output bất kỳ"""
        l_charb = self.charb(pred, target)
        l_ssim = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)
        l_color = self.color(pred, input_img, target)
        
        total = l_charb + (self.ssim_weight * l_ssim) + (self.color_weight * l_color)
        return total, l_charb, l_ssim, l_color

    def forward(
        self,
        input_img,
        target,
        intermediate_preds=None,
        halting=None,
    ):
        # 1. FINAL LOSS
        final_total, final_charb, final_ssim, final_color = self.recon_loss(final_pred, input_img, target)

        # 2. LOOP LOSS (Quỹ đạo trung gian)
        if intermediate_preds is not None and len(intermediate_preds) > 0:
            T = len(intermediate_preds)
            charb_stack, ssim_stack, color_stack = [], [], []

            for p in intermediate_preds:
                _, c, s, col = self.recon_loss(p, input_img, target)
                charb_stack.append(c)
                ssim_stack.append(s)
                color_stack.append(col)

            charb_stack = torch.stack(charb_stack, dim=0) 
            ssim_stack = torch.stack(ssim_stack, dim=0)
            color_stack = torch.stack(color_stack, dim=0)
            
            # Đánh trọng số tăng dần (ép các step sau phải nét hơn step trước)
            weights = torch.linspace(0.5, 1.0, steps=T, device=target.device)
            weights = weights / weights.sum()

            loop_charb = (weights * charb_stack).sum()
            loop_ssim = (weights * ssim_stack).sum()
            loop_color = (weights * color_stack).sum()
            
            loop_total = loop_charb + (self.ssim_weight * loop_ssim) + (self.color_weight * loop_color)
        else:
            zero_val = torch.tensor(0.0, device=target.device)
            loop_total = loop_charb = loop_ssim = loop_color = zero_val

        # 3. KL DIVERGENCE (Ép mô hình thoát sớm - Early Exit)
        kl_loss = torch.tensor(0.0, device=target.device)
        if halting is not None:
            h = halting.clamp(min=1e-8)
            T_val = h.shape[0]
            uniform = torch.full_like(h, 1.0 / T_val)
            # KL divergence chuẩn giữa phân phối halting thực tế và phân phối đều
            kl_loss = (h * (h.log() - uniform.log())).sum(dim=0).mean()

        # 4. TỔNG HỢP TOTAL LOSS
        loss = (
            final_total 
            + self.loop_weight * loop_total 
            + self.kl_weight * kl_loss
        )

        loss_dict = {
            "loss/total": loss.item(),
            "loss/final_total": final_total.item(),
            "loss/final_charb": final_charb.item(),
            "loss/final_ssim": final_ssim.item(),
            "loss/final_color": final_color.item(),
            "loss/loop_total": loop_total.item(),
            "loss/loop_charb": loop_charb.item(),
            "loss/loop_ssim": loop_ssim.item(),
            "loss/loop_color": loop_color.item(),
            "loss/kl": kl_loss.item(),
        }

        return loss, loss_dict