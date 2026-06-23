import os
import random
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
from pytorch_msssim import ms_ssim, ssim
import glob


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
    ssim_score = compute_ssim(pred, gt)
    ms_ssim_score = compute_ms_ssim(pred, gt)
    rmse = compute_rmse(pred, gt)

    def to_np(x):
        x = x.detach().float().cpu()
        if x.dim() == 4:
            x = x.squeeze(0)
        x = torch.clamp(x, 0, 1)
        return x.permute(1, 2, 0).numpy()

    vis = np.concatenate([to_np(inp), to_np(pred), to_np(gt)], axis=1)

    metrics = {
        "psnr": psnr,
        "ssim": ssim_score,
        "ms_ssim": ms_ssim_score,
        "rmse": rmse
    }
    return vis, metrics


def get_loop_steps_logs(inp, gt, intermediate_preds, halting_weights, halt_logits, max_samples=4, start_idx=0):
    T = len(intermediate_preds)
    B = min(inp.size(0), max_samples)
    log_dict = {}

    for b in range(B):
        loop_images = []
        for t in range(T):
            inp_b = inp[b:b+1]
            pred_t = intermediate_preds[t][b:b+1]
            gt_b = gt[b:b+1]

            vis, metrics = make_vis(inp_b, pred_t, gt_b)

            exit_prob = torch.sigmoid(halt_logits[t][b]).mean().item() if halt_logits is not None else 0.0
            act_w = halting_weights[t, b].item() if halting_weights is not None else 0.0

            caption = (f"{t+1}/{T} | Exit: {exit_prob*100:.1f}% | "
                       f"Weight: {act_w:.3f} | PSNR={metrics['psnr']:.2f} | "
                       f"RMSE={metrics['rmse']:.4f} | "
                       f"SSIM={metrics['ssim']:.4f} | MS-SSIM={metrics['ms_ssim']:.4f}")

            loop_images.append(wandb.Image(vis, caption=caption))

        # Sử dụng start_idx để không bị ghi đè index trên WandB khi khác batch
        log_dict[f"val/sample_{start_idx + b}_progress"] = loop_images

    return log_dict


def train_one_epoch(
    model, loader, optimizer, scaler, criterion, device, epoch, cfg,
    aug=None, stage=1, global_step=0,
    accumulation_steps=4, max_steps_per_epoch=500
):
    model.train()
    running_loss = 0.0

    total_steps = min(len(loader), max_steps_per_epoch)

    pbar = tqdm(
        loader,
        total=total_steps,
        desc=f"Train Epoch {epoch} | Stage {stage}"
    )

    optimizer.zero_grad(set_to_none=True)

    for batch_idx, batch in enumerate(pbar):
        if batch_idx >= total_steps:
            break

        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        if aug is not None:
            inp, gt = aug(inp, gt)

        with autocast(device_type="cuda", enabled=getattr(cfg, 'use_amp', True)):
            outputs = model(inp, return_all=True)
            pred, intermediate_preds, halting_weights, gate_logits, exit_probs = outputs

            if halting_weights is not None and halting_weights.shape[0] != len(intermediate_preds):
                T = len(intermediate_preds)
                num_gpus = halting_weights.shape[0] // T
                hw = halting_weights.view(num_gpus, T, -1).permute(1, 0, 2)
                halting_weights = hw.reshape(T, -1)

            if stage == 1:
                loss, loss_dict = criterion(
                    input_img=inp,
                    target=gt,
                    final_pred=pred,
                    intermediate_preds=intermediate_preds,
                    halting_weights=halting_weights,
                    current_epoch=epoch
                )
            else:
                loss, loss_dict = criterion(
                    target=gt,
                    final_pred=pred,
                    intermediate_preds=intermediate_preds,
                    halting_weights=halting_weights,
                    halt_logits=gate_logits
                )

        loss_raw = loss.item()

        loss = loss / accumulation_steps
        scaler.scale(loss).backward()

        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == total_steps:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss_raw

        T_steps = halting_weights.shape[0] if halting_weights is not None else len(intermediate_preds)
        
        if halting_weights is not None:
            step_ids = torch.arange(1, T_steps + 1, device=halting_weights.device).float().view(-1, 1)
            expected_steps = (step_ids * halting_weights).sum(dim=0).mean().item()
            hard_steps = halting_weights.argmax(dim=0).float().mean().item() + 1
        else:
            expected_steps = T_steps
            hard_steps = T_steps

        if batch_idx == total_steps - 1:
            wandb.log({
                "train/loss": loss_raw,
                "train/expected_steps": expected_steps,
                "train/hard_steps": hard_steps,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train/{k}": (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
            }, step=global_step)

        global_step += 1
        pbar.set_postfix(loss=f"{loss_raw:.4f}", steps=f"{expected_steps:.2f}")

    return running_loss / total_steps, global_step


# VALIDATION (PHIÊN BẢN LOG THẲNG - SIÊU NHẸ RAM)
@torch.no_grad()
def validate(model, loader, device, cfg, global_step=0, log_images=True, max_log_images=4):
    model.eval()

    loop_psnr_sums = None
    loop_ssim_sums = None

    psnr_sum = 0.0
    ssim_sum = 0.0
    ms_ssim_sum = 0.0
    rmse_sum = 0.0
    total_images = 0
    
    logged_images_count = 0

    for i, batch in enumerate(tqdm(loader, desc="Validating Progressively")):
        inp = batch["inp"].to(device)
        gt = batch["gt"].to(device)

        with autocast(device_type="cuda", enabled=getattr(cfg, 'use_amp', True)):
            outputs = model(inp, return_all=True)

        pred, inter_preds, halt_w, halt_logits, _ = outputs
        
        # SỬA LỖI DATAPARALLEL (Giữ nguyên phần này)
        if halt_w is not None and halt_w.shape[0] != len(inter_preds):
            T = len(inter_preds)
            num_gpus = halt_w.shape[0] // T
            hw = halt_w.view(num_gpus, T, -1).permute(1, 0, 2)
            halt_w = hw.reshape(T, -1)
            
            if halt_logits is not None:
                hl = halt_logits.view(num_gpus, T, -1).permute(1, 0, 2)
                halt_logits = hl.reshape(T, -1)
        
        T = len(inter_preds)
        batch_size = inp.size(0)

        if loop_psnr_sums is None:
            loop_psnr_sums = [0.0] * T
            loop_ssim_sums = [0.0] * T

        for b in range(batch_size):
            g_img = gt[b:b+1]
            psnr_sum += compute_psnr(pred[b:b+1], g_img)
            ssim_sum += compute_ssim(pred[b:b+1], g_img)
            ms_ssim_sum += compute_ms_ssim(pred[b:b+1], g_img)
            rmse_sum += compute_rmse(pred[b:b+1], g_img)

            for t in range(T):
                loop_psnr_sums[t] += compute_psnr(inter_preds[t][b:b+1], g_img)
                loop_ssim_sums[t] += compute_ssim(inter_preds[t][b:b+1], g_img)
            total_images += 1

            # LOG TRỰC TIẾP LÊN WANDB (KHÔNG GOM BIẾN)
            if log_images and logged_images_count < max_log_images:
                # Chỉ lấy dữ liệu của 1 ảnh tại index b
                img_inp = inp[b:b+1]
                img_pred = pred[b:b+1]
                img_gt = g_img
                
                # Tạo log đơn lẻ
                vis, metrics = make_vis(img_inp, img_pred, img_gt)
                wandb.log({
                    f"val/sample_{logged_images_count}_final": wandb.Image(vis, caption=f"PSNR={metrics['psnr']:.2f}")
                }, step=global_step)
                
                logged_images_count += 1

    # Tính toán metrics cuối cùng
    val_metrics = {
        "val/psnr": psnr_sum / total_images,
        "val/ssim": ssim_sum / total_images,
        "val/ms_ssim": ms_ssim_sum / total_images,
        "val/rmse": rmse_sum / total_images,
    }

    for t in range(T):
        val_metrics[f"val_loops/loop_{t+1}_psnr"] = loop_psnr_sums[t] / total_images
        val_metrics[f"val_loops/loop_{t+1}_ssim"] = loop_ssim_sums[t] / total_images

    return val_metrics


# MAIN TRAINING LOOP
def train_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, cfg, aug=None, stage=1):
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
        print("No valid checkpoint found or resume_checkpoint is empty. Training from scratch.")

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
            stage=stage,
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

        state_to_save = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()

        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_model_name = f"best_model_stage{stage}.pth"
            torch.save(state_to_save, best_model_name)
            print(f"→ Saved best model as '{best_model_name}' (PSNR: {best_psnr:.3f})")

        checkpoint_state = {
            'epoch': epoch,
            'global_step': global_step,
            'model_state_dict': state_to_save,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_psnr': best_psnr
        }
        torch.save(checkpoint_state, f"latest_checkpoint_stage{stage}.pth")

        print(f"[{epoch:3d}] Loss: {train_loss:.4f} | PSNR: {current_psnr:.3f} | SSIM: {val_metrics['val/ssim']:.4f}")

        if (epoch + 1) >= 5 and current_psnr < 15.0:
            print(f"Stop! At {epoch}: PSNR ({current_psnr:.3f}) < 15.0. Model is likely dying.")
            break

    print("Training completed!")
    return model