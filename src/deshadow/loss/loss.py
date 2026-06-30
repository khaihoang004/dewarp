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


class DocDeshadowLoss(nn.Module):
    def __init__(
        self,
        ssim_weight=0.0,
        color_weight=0.35,
        freq_weight=0.5,
        low_weight=1.25,
        high_weight=0.8,
        kl_weight=0.02,
        loop_weight=0.10,
    ):
        super().__init__()
        self.charb = CharbonnierLoss()
        self.freq_separator = FFTSeparator(cutoff_freq=25)
        self.color = LABColorLoss()

        self.freq_weight = freq_weight

        self.low_weight = low_weight
        self.high_weight = high_weight

        self.ssim_weight = ssim_weight
        self.color_weight = color_weight
        self.kl_weight = kl_weight
        self.loop_weight = loop_weight

    def recon_loss(self, pred, input_img, target):
        # 1. RGB (Charbonnier)
        l_charb = self.charb(pred, target)

        # 2. FFT frequency
        pred_low, pred_high = self.freq_separator(pred)
        tgt_low, tgt_high = self.freq_separator(target)

        l_low = self.charb(pred_low, tgt_low)
        l_high = self.charb(pred_high, tgt_high)

        l_freq = (
            self.low_weight * l_low +
            self.high_weight * l_high
        )

        # 4. Color (LAB)
        l_color = self.color(pred, input_img, target)

        # 5. SSIM Loss
        if self.ssim_weight > 0:
            ssim_val = ssim(pred, target, data_range=1.0, size_average=True)
            l_ssim = 1.0 - ssim_val
        else:
            l_ssim = torch.tensor(0.0, device=pred.device)

        # TOTAL 
        total = (
            l_charb
            + self.freq_weight * l_freq
            + self.color_weight * l_color
            + self.ssim_weight * l_ssim
        )

        return (
            total,
            l_charb,
            l_ssim,
            l_color,
            l_freq,
            l_low,
            l_high,
        )

    def forward(
        self,
        final_pred,
        input_img,
        target,
        intermediate_preds=None,
        halting=None,
    ):
        # 1. FINAL LOSS
        (
            final_total,
            final_charb,
            final_ssim,
            final_color,
            final_freq,
            final_low,
            final_high,
        ) = self.recon_loss(
            final_pred,
            input_img,
            target
        )
        
        # 2. LOOP LOSS (Quỹ đạo trung gian)
        if intermediate_preds is not None and len(intermediate_preds) > 0:
            T = len(intermediate_preds)
            charb_stack, ssim_stack, color_stack = [], [], []
            freq_stack, low_stack, high_stack = [], [], []

            for p in intermediate_preds:
                recon_outs = self.recon_loss(p, input_img, target)
                
                # Trích xuất giá trị theo đúng vị trí index trả về từ recon_loss
                charb_stack.append(recon_outs[1])
                ssim_stack.append(recon_outs[2])
                color_stack.append(recon_outs[3])
                freq_stack.append(recon_outs[4])     # l_freq ở index 4
                low_stack.append(recon_outs[5])      # l_low ở index 5
                high_stack.append(recon_outs[6])     # l_high ở index 6

            charb_stack = torch.stack(charb_stack, dim=0) 
            ssim_stack = torch.stack(ssim_stack, dim=0)
            color_stack = torch.stack(color_stack, dim=0)
            freq_stack = torch.stack(freq_stack, dim=0)
            low_stack = torch.stack(low_stack, dim=0)
            high_stack = torch.stack(high_stack, dim=0)
            
            # Đánh trọng số tăng dần theo thời gian/bước lặp
            weights = torch.linspace(0.5, 1.0, steps=T, device=target.device)
            weights = weights / weights.sum()

            loop_charb = (weights * charb_stack).sum()
            loop_ssim = (weights * ssim_stack).sum()
            loop_color = (weights * color_stack).sum()
            loop_freq = (weights * freq_stack).sum()
            loop_low = (weights * low_stack).sum()
            loop_high = (weights * high_stack).sum()

            # Tính tổng loop_total kèm theo hệ số scale của từng loss component giống final_loss
            loop_total = (
                loop_charb 
                + (self.ssim_weight * loop_ssim) 
                + (self.color_weight * loop_color)
                + (self.freq_weight * loop_freq)         # Đã thêm FFT vào loop
            )
        else:
            zero_val = torch.tensor(0.0, device=target.device)
            loop_total = loop_charb = loop_ssim = loop_color = loop_freq = loop_low = loop_high = zero_val

        # 3. KL DIVERGENCE (Early Exit)
        kl_loss = torch.tensor(0.0, device=target.device)
        if halting is not None:
            h = halting.clamp(min=1e-8)
            T_val = h.shape[0]
            uniform = torch.full_like(h, 1.0 / T_val)
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
            "loss/final_freq": final_freq.item(),
            "loss/final_low": final_low.item(),    # Thêm log final_low
            "loss/final_high": final_high.item(),  # Thêm log final_high
            "loss/loop_total": loop_total.item(),
            "loss/loop_charb": loop_charb.item(),
            "loss/loop_ssim": loop_ssim.item(),
            "loss/loop_color": loop_color.item(),
            "loss/loop_freq": loop_freq.item(),    # Thêm log thông số tần số trung gian
            "loss/loop_low": loop_low.item(),      # Thêm log loop_low
            "loss/loop_high": loop_high.item(),    # Thêm log loop_high
            "loss/kl": kl_loss.item(),
        }

        return loss, loss_dict