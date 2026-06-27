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


# --- 1. HÀM TRAIN 1 EPOCH (SẠCH SẼ VÀ TỐI ƯU CHO LOOPVIT) ---
def train_one_epoch(
    model, loader, optimizer, scaler, criterion, device, epoch, cfg,
    aug=None, global_step=0, accumulation_steps=4, max_steps_per_epoch=500
):
    model.train()
    running_loss = 0.0
    total_steps = min(len(loader), max_steps_per_epoch)

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
            # BẮT BUỘC return_all=True khi Training để chạy đủ T bước
            outputs, intermediate_preds, bottleneck_logits_list = model(inp, tau=0.05, return_all=True)

            # Truyền List vào Criterion
            loss, loss_dict = criterion(
                input_img=inp,
                target=gt,
                intermediate_preds=intermediate_preds,
                bottleneck_logits_list=bottleneck_logits_list
            )

        loss_raw = loss.item()
        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        # Cập nhật Gradient
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss_raw

        # Log WandB
        if batch_idx == total_steps - 1:
            wandb.log({
                "train/loss": loss_raw,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train/{k}": (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
            }, step=global_step)

        global_step += 1
        pbar.set_postfix(loss=f"{loss_raw:.4f}")

    return running_loss / total_steps, global_step


# --- 2. HÀM VALIDATE (KIỂM TRA DỪNG ĐỘNG - EARLY EXIT) ---
@torch.no_grad()
def validate(model, loader, device, cfg, global_step=0, log_images=True, max_log_images=4):
    model.eval()

    psnr_sum = ssim_sum = ms_ssim_sum = rmse_sum = 0.0
    total_images = 0
    total_exit_steps = 0.0  # Biến mới để theo dõi LoopViT tiết kiệm được bao nhiêu vòng lặp
    logged_images_count = 0

    for i, batch in enumerate(tqdm(loader, desc="Validating")):
        inp = batch["inp"].to(device)
        gt = batch["gt"].to(device)

        with autocast(device_type="cuda", enabled=getattr(cfg, 'use_amp', True)):
            # Inference Mode: return_all=False để kích hoạt Entropy Hard-Exit
            pred, exit_steps = model(inp, tau=0.05, return_all=False)
            
        batch_size = inp.size(0)
        total_images += batch_size
        
        # Cộng dồn số bước đã exit để tính trung bình
        total_exit_steps += exit_steps.float().sum().item()

        for b in range(batch_size):
            g_img = gt[b:b+1]
            pred_img = pred[b:b+1]
            
            psnr_sum += compute_psnr(pred_img, g_img)
            ssim_sum += compute_ssim(pred_img, g_img)
            ms_ssim_sum += compute_ms_ssim(pred_img, g_img)
            rmse_sum += compute_rmse(pred_img, g_img)

            # Log ảnh mẫu trực tiếp lên WandB
            if log_images and logged_images_count < max_log_images:
                vis, metrics = make_vis(inp[b:b+1], pred_img, g_img)
                # Kèm theo số vòng lặp thực tế mà sample này cần
                caption = f"PSNR={metrics['psnr']:.2f} | Exit at Loop: {exit_steps[b].item()}"
                
                wandb.log({
                    f"val/sample_{logged_images_count}": wandb.Image(vis, caption=caption)
                }, step=global_step)
                logged_images_count += 1

    # Tính toán metrics cuối cùng
    val_metrics = {
        "val/psnr": psnr_sum / total_images,
        "val/ssim": ssim_sum / total_images,
        "val/ms_ssim": ms_ssim_sum / total_images,
        "val/rmse": rmse_sum / total_images,
        "val/avg_exit_steps": total_exit_steps / total_images, # <--- METRIC QUAN TRỌNG NHẤT!
    }

    return val_metrics


# --- 3. MAIN TRAINING LOOP ---
def train_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, cfg, aug=None):
    scaler = GradScaler(enabled=cfg.use_amp)
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
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "train/loss_epoch": train_loss,
            **val_metrics,
            "global_step": global_step
        }, step=global_step)

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

        if (epoch + 1) >= 5 and current_psnr < 15.0:
            print(f"Stop! At {epoch}: PSNR ({current_psnr:.3f}) < 15.0. Model is likely dying.")
            break

    print("Training completed!")
    return model