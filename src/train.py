import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
from pytorch_msssim import ms_ssim, ssim   # ← import cả hai


# =========================================================
# METRICS
# =========================================================

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


# =========================================================
# VISUAL + LOGGING
# =========================================================

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


def log_loop_steps(inp, gt, intermediate_preds, halting_weights, halt_logits, global_step, max_samples=2):
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
            act_w = halting_weights[t][b].mean().item()

            caption = (f"Loop {t+1}/{T} | Exit: {exit_prob*100:.1f}% | "
                      f"Weight: {act_w:.3f} | PSNR={metrics['psnr']:.2f} | "
                      f"SSIM={metrics['ssim']:.4f} | MS-SSIM={metrics['ms_ssim']:.4f}")

            loop_images.append(wandb.Image(vis, caption=caption))

        log_dict[f"val/sample_{b}_progress"] = loop_images

    wandb.log(log_dict, step=global_step)


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, cfg, aug=None, stage=1, global_step=0):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Train Epoch {epoch} | Stage {stage}")

    for batch_idx, batch in enumerate(pbar):
        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        if aug is not None:
            inp, gt = aug(inp, gt)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=cfg.use_amp):
            # Model trả về 5 giá trị khi return_all=True
            outputs = model(inp, return_all=True)
            pred, intermediate_preds, halting_weights, gate_logits, exit_probs = outputs

            if stage == 1:
                loss, loss_dict = criterion(
                    target=gt, 
                    final_pred=pred,
                    intermediate_preds=intermediate_preds, 
                    halting_weights=halting_weights
                )
            else:
                loss, loss_dict = criterion(
                    target=gt, 
                    final_pred=pred,
                    intermediate_preds=intermediate_preds, 
                    halting_weights=halting_weights,
                    halt_logits=gate_logits
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Log nhanh
        if (batch_idx + 1) % 20 == 0 or batch_idx == len(loader) - 1:
            expected_steps = (torch.arange(1, len(intermediate_preds)+1, device=halting_weights.device).float().view(-1, 1) * halting_weights).sum(dim=0).mean().item()

            wandb.log({
                "train/loss": loss.item(),
                "train/expected_steps": expected_steps,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train/{k}": v for k, v in loss_dict.items()}
            }, step=global_step)

        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", steps=f"{expected_steps:.2f}")

    return running_loss / len(loader), global_step


# =========================================================
# VALIDATION
# =========================================================

@torch.no_grad()
def validate(model, loader, device, global_step=0, log_images=True):
    model.eval()
    psnr_sum = ssim_sum = ms_ssim_sum = 0.0
    total = 0

    for i, batch in enumerate(tqdm(loader, desc="Validating")):
        inp = batch["inp"].to(device)
        gt = batch["gt"].to(device)

        # Chỉ log loop steps cho batch đầu tiên
        if i == 0 and log_images:
            outputs = model(inp, return_all=True)
            pred, inter_preds, halt_w, halt_logits, _ = outputs
            log_loop_steps(inp, gt, inter_preds, halt_w, halt_logits, global_step)
        else:
            pred = model(inp)  # inference mode

        # Metrics
        psnr_sum += compute_psnr(pred, gt)
        ssim_sum += compute_ssim(pred, gt)
        ms_ssim_sum += compute_ms_ssim(pred, gt)
        total += 1

    return {
        "val/psnr": psnr_sum / total,
        "val/ssim": ssim_sum / total,
        "val/ms_ssim": ms_ssim_sum / total,
    }


# =========================================================
# MAIN TRAINING LOOP
# =========================================================

def train_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, cfg, aug=None, stage=1):
    scaler = GradScaler(enabled=cfg.use_amp)
    best_psnr = 0.0
    global_step = 0

    for epoch in range(cfg.epochs):
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
            global_step=global_step
        )

        val_metrics = validate(model, val_loader, device, global_step=global_step)

        scheduler.step()

        # Log epoch summary
        wandb.log({
            "epoch": epoch,
            "train/loss_epoch": train_loss,
            **val_metrics,
            "global_step": global_step
        }, step=global_step)

        current_psnr = val_metrics["val/psnr"]
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            torch.save(model.state_dict(), f"best_model_stage{stage}.pth")
            print(f"→ Saved best model at epoch {epoch} (PSNR: {best_psnr:.3f})")

        print(f"[{epoch:3d}] Loss: {train_loss:.4f} | "
              f"PSNR: {current_psnr:.3f} | "
              f"SSIM: {val_metrics['val/ssim']:.4f} | "
              f"MS-SSIM: {val_metrics['val/ms_ssim']:.4f}")

    print("Training completed!")
    return model