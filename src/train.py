import os
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
from pytorch_msssim import ms_ssim, ssim

# --- CÁC HÀM TIỆN ÍCH (Giữ nguyên) ---
@torch.no_grad()
def compute_psnr(pred, target):
    mse = torch.mean((pred - target) ** 2)
    if mse <= 1e-10:
        return 100.0
    return (20.0 * torch.log10(1.0 / torch.sqrt(mse))).item()

@torch.no_grad()
def compute_ssim(pred, target):
    if pred.dim() == 3: pred = pred.unsqueeze(0)
    if target.dim() == 3: target = target.unsqueeze(0)
    return ssim(pred, target, data_range=1.0, size_average=True).item()

@torch.no_grad()
def compute_ms_ssim(pred, target):
    if pred.dim() == 3: pred = pred.unsqueeze(0)
    if target.dim() == 3: target = target.unsqueeze(0)
    return ms_ssim(pred, target, data_range=1.0, size_average=True).item()

@torch.no_grad()
def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

def make_vis(inp, pred, gt):
    psnr = compute_psnr(pred, gt)
    metrics = {
        "psnr": psnr,
        "ssim": compute_ssim(pred, gt),
        "ms_ssim": compute_ms_ssim(pred, gt),
        "rmse": compute_rmse(pred, gt)
    }

    def to_np(x):
        x = x.detach().float().cpu()
        if x.dim() == 4:
            x = x.squeeze(0)
        x = torch.clamp(x, 0, 1)
        return x.permute(1, 2, 0).numpy()

    vis = np.concatenate([to_np(inp), to_np(pred), to_np(gt)], axis=1)
    return vis, metrics


# --- 1. HÀM TRAIN 1 EPOCH ---
def train_one_epoch(
    model, loader, optimizer, scaler, criterion, device, epoch, cfg,
    aug=None, global_step=0, accumulation_steps=4, max_steps_per_epoch=500
):
    model.train()
    running_loss = 0.0
    total_steps = min(len(loader), max_steps_per_epoch)

    # Thêm log_freq (bao nhiêu batch thì log lên wandb 1 lần, mặc định 50)
    log_freq = getattr(cfg, "log_freq", 50)
    grad_clip = getattr(cfg, "grad_clip", 1.0)

    pbar = tqdm(loader, total=total_steps, desc=f"Train Epoch {epoch}")
    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= total_steps:
            break

        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        if aug is not None:
            inp, gt = aug(inp, gt)

        with autocast(device_type="cuda", enabled=getattr(cfg, 'use_amp', True)):
            out = model(inp, calc_loss=True)

            loss, loss_dict = criterion(
                final_pred=out["final"],
                target=gt,
                intermediate_preds=out["intermediate"],
                halting=out["halting"],
            )

        loss_raw = loss.item()
        loss_scaled = loss / accumulation_steps
        scaler.scale(loss_scaled).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss_raw

        # LOG THEO TẦN SUẤT (Tránh việc đồ thị bị thiếu)
        if (batch_idx + 1) % log_freq == 0 or (batch_idx + 1) == total_steps:
            
            # --- TÍNH KỲ VỌNG SỐ VÒNG LẶP (EXPECTED STEPS) ---
            h = out["halting"] # Shape: (T, B)
            h_mean = h.mean(dim=1) # Trung bình theo batch, ra shape (T,)

            # --- IN RA XÁC SUẤT TẠI CÁC LOOP ---
            print(f"\n[Step {global_step}] Halting Probabilities per Loop:")
            probs_dict = {f"Loop {t+1}": f"{val:.4f}" for t, val in enumerate(h_mean)}
            print(probs_dict)
            T_steps = h.shape[0]
            
            # Tạo vector [1, 2, ..., T]
            step_indices = torch.arange(1, T_steps + 1, device=h.device).view(T_steps, 1).float()
            
            # Expected steps = sum(h_t * t)
            expected_steps = (h * step_indices).sum(dim=0).mean().item()
            # ---------------------------------------------------

            clean_loss_dict = {f"train/{k.replace('loss/', '')}": v for k, v in loss_dict.items()}
            
            wandb.log({
                "train/loss": loss_raw,
                "train/expected_steps": expected_steps, # LOG THÊM VÀO ĐÂY
                "lr": optimizer.param_groups[0]["lr"],
                **clean_loss_dict
            }, step=global_step)

        global_step += 1
        pbar.set_postfix(loss=f"{loss_raw:.4f}")

    return running_loss / total_steps, global_step


# --- 2. HÀM VALIDATE ---
@torch.no_grad()
def validate(model, loader, device, cfg, global_step=0, log_images=True, max_log_images=4):
    model.eval()

    psnr_sum = ssim_sum = ms_ssim_sum = rmse_sum = 0.0
    total_images = 0
    total_exit_steps = 0.0
    logged_images_count = 0

    val_images_log = {} # MỚI: Gom ảnh lại log 1 lần

    for i, batch in enumerate(tqdm(loader, desc="Validating")):
        inp = batch["inp"].to(device)
        gt = batch["gt"].to(device)

        with autocast(device_type="cuda", enabled=getattr(cfg, 'use_amp', True)):
            out = model(
                inp,
                halt_threshold=getattr(cfg, "halt_threshold", 0.8),
                return_steps=True, # Suy luận: bật tính năng check Early Exit
            )

            pred = out["final"]
            exit_steps = out["exit_steps"]

        batch_size = inp.size(0)
        total_images += batch_size
        total_exit_steps += exit_steps.sum().item()

        for b in range(batch_size):
            psnr_sum += compute_psnr(pred[b:b+1], gt[b:b+1])
            ssim_sum += compute_ssim(pred[b:b+1], gt[b:b+1])
            ms_ssim_sum += compute_ms_ssim(pred[b:b+1], gt[b:b+1])
            rmse_sum += compute_rmse(pred[b:b+1], gt[b:b+1])

            if log_images and logged_images_count < max_log_images:
                vis, metrics = make_vis(inp[b:b+1], pred[b:b+1], gt[b:b+1])
                # MỚI: Chỉ lưu vào dict, không đẩy lên wandb ngay
                val_images_log[f"val/sample_{logged_images_count}"] = wandb.Image(
                    vis, caption=f"PSNR={metrics['psnr']:.2f}"
                )
                logged_images_count += 1

    # Trả về toàn bộ metrics VÀ images trong 1 Dictionary duy nhất
    metrics_dict = {
        "val/psnr": psnr_sum / total_images,
        "val/ssim": ssim_sum / total_images,
        "val/ms_ssim": ms_ssim_sum / total_images,
        "val/rmse": rmse_sum / total_images,
        "val/avg_exit_steps": total_exit_steps / total_images,
    }
    metrics_dict.update(val_images_log)
    return metrics_dict


# --- 3. MAIN TRAINING LOOP ---
def train_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, cfg, aug=None):
    # SỬA LỖI: Thêm tham số device "cuda" cho GradScaler tương thích mọi ver PyTorch
    scaler = GradScaler("cuda", enabled=getattr(cfg, 'use_amp', True)) 
    
    best_psnr = 0.0
    global_step = 0
    start_epoch = 0

    checkpoint_path = getattr(cfg, "resume_checkpoint", None)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Load checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        global_step = checkpoint['global_step']
        best_psnr = checkpoint.get('best_psnr', 0.0)

        print(f"Continue training from epoch {start_epoch}, best_psnr={best_psnr:.3f}.")
    else:
        print("Training from scratch.")

    accumulation_steps = getattr(cfg, "accumulation_steps", 4)
    max_steps_per_epoch = getattr(cfg, "max_steps_per_epoch", 500)

    for epoch in range(start_epoch, cfg.epochs):
        train_loss, global_step = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            criterion=criterion,
            device=device,
            epoch=epoch,
            cfg=cfg,
            aug=aug,
            global_step=global_step,
            accumulation_steps=accumulation_steps,
            max_steps_per_epoch=max_steps_per_epoch
        )

        val_metrics = validate(model, val_loader, device, cfg, global_step=global_step)
        
        wandb.log({
            "epoch": epoch,
            "train/loss_epoch": train_loss,
            **val_metrics,
        }, step=global_step)

        scheduler.step()

        current_psnr = val_metrics["val/psnr"]
        avg_exit = val_metrics["val/avg_exit_steps"]

        state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_model_name = "best_model.pth"
            torch.save(state_to_save, best_model_name)
            print(f"→ Saved best model as '{best_model_name}' (PSNR: {best_psnr:.3f} | Avg Exit: {avg_exit:.2f})")

        checkpoint_state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': state_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_psnr': best_psnr
        }
        torch.save(checkpoint_state, "latest_checkpoint.pth")

        print(f"[{epoch:3d}] Loss: {train_loss:.4f} | PSNR: {current_psnr:.3f} | Avg Exit Steps: {avg_exit:.2f}")

        if (epoch + 1) >= 5 and current_psnr < 10.0:
            print(f"Stop! At {epoch}: PSNR ({current_psnr:.3f}) < 15.0. Model is likely dying.")
            break

    print("Training completed!")
    return model