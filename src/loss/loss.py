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
    def __init__(self, prior_weight=0.005, alpha=0.84):
        super().__init__()
        self.prior_weight = prior_weight
        self.alpha = alpha
        self.charbonnier = CharbonnierLoss()

    def reconstruction_loss(self, pred, target):
        charbonnier_loss = self.charbonnier(pred, target)
        
        pred_safe = torch.clamp(pred, 0.0, 1.0).float()
        target_safe = target.float()

        ms_ssim_loss = 1.0 - ms_ssim(
            pred_safe, target_safe, data_range=1.0, size_average=True
        )
        
        combined_loss = self.alpha * charbonnier_loss + (1.0 - self.alpha) * ms_ssim_loss
        return combined_loss, charbonnier_loss, ms_ssim_loss

    def forward(self, target, final_pred, intermediate_preds, halting_weights, **kwargs):
        T = len(intermediate_preds)
        
        total_rec_loss = 0.0
        total_charb_loss = 0.0
        total_msssim_loss = 0.0

        # =====================================
        # multi-step reconstruction
        # =====================================
        for t in range(T):
            pred_t = intermediate_preds[t]
            rec_loss_t, charb_t, msssim_t = self.reconstruction_loss(pred_t, target)

            total_rec_loss += (rec_loss_t / T)
            total_charb_loss += (charb_t / T)
            total_msssim_loss += (msssim_t / T)

        # =====================================
        # uniform prior KL
        # =====================================
        prior = torch.full_like(halting_weights, 1.0 / T)
        
        kl_loss = F.kl_div(
            torch.log(halting_weights + 1e-6),
            prior,
            reduction='batchmean'
        )

        total_loss = total_rec_loss + self.prior_weight * kl_loss
        
        # =====================================
        # Bổ sung các thành phần vào dictionary
        # =====================================
        loss_dict = {
            "train/total_loss": total_loss.item(),
            "train/rec_loss_combined": total_rec_loss.item(),
            "train/rec_charbonnier": total_charb_loss.item(),
            "train/rec_ms_ssim": total_msssim_loss.item(),
            "train/kl_loss": kl_loss.item()
        }

        return total_loss, loss_dict


class Stage2Loss(nn.Module):
    def __init__(self, gain_threshold=0.005, gain_scale=10.0, ponder_weight=0.02, alpha=0.84):
        super().__init__()
        self.gain_threshold = gain_threshold
        self.gain_scale = gain_scale
        self.ponder_weight = ponder_weight
        self.alpha = alpha
        self.charbonnier = CharbonnierLoss()

    def reconstruction_loss(self, pred, target):
        charbonnier_loss = self.charbonnier(pred, target)
        
        pred_safe = torch.clamp(pred, 0.0, 1.0).float()
        target_safe = target.float()

        ms_ssim_loss = 1.0 - ms_ssim(
            pred_safe, target_safe, data_range=1.0, size_average=True
        )
        combined_loss = self.alpha * charbonnier_loss + (1.0 - self.alpha) * ms_ssim_loss
        return combined_loss, charbonnier_loss, ms_ssim_loss

    def forward(self, target, final_pred, intermediate_preds, halting_weights, halt_logits, **kwargs):
        T = len(intermediate_preds)
        step_losses = []
        
        avg_charb = 0.0
        avg_msssim = 0.0

        # =====================================
        # compute reconstruction losses
        # =====================================
        with torch.no_grad():
            for t in range(T):
                pred_t = intermediate_preds[t]
                rec_loss_t, charb_t, msssim_t = self.reconstruction_loss(pred_t, target)
                step_losses.append(rec_loss_t)
                
                avg_charb += charb_t
                avg_msssim += msssim_t
                
        # Trong Stage 2 backbone bị freeze, chỉ track giá trị trung bình để xem chất lượng
        avg_charb /= T
        avg_msssim /= T

        # =====================================
        # gate supervision
        # =====================================
        gate_loss = 0.0
        prev_loss = None

        for t in range(T - 1):
            current_loss = step_losses[t]
            if prev_loss is None:
                gain_t = torch.tensor(0.05, device=current_loss.device)
            else:
                gain_t = prev_loss - current_loss
            prev_loss = current_loss

            utility = torch.sigmoid(self.gain_scale * (gain_t - self.gain_threshold))
            target_exit = (1.0 - utility).detach()
            logits_t = halt_logits[t]
            target_tensor = torch.full_like(logits_t, target_exit)
            gate_loss += F.binary_cross_entropy_with_logits(logits_t, target_tensor)

        # =====================================
        # ponder regularization
        # =====================================
        expected_steps = 0.0
        for t in range(T):
            expected_steps += ((t + 1) * halting_weights[t].mean())

        total_loss = gate_loss + self.ponder_weight * expected_steps
        
        loss_dict = {
            "train/total_loss": total_loss.item(),
            "train/gate_loss": gate_loss.item() if isinstance(gate_loss, torch.Tensor) else gate_loss,
            "train/ponder_loss": (self.ponder_weight * expected_steps).item(),
            "train/expected_steps": expected_steps.item(),
            "train/eval_charbonnier": avg_charb.item(),
            "train/eval_ms_ssim": avg_msssim.item()
        }

        return total_loss, loss_dict