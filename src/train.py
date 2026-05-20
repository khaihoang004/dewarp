import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
from pytorch_msssim import ssim

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
    # Fallback an toàn (Mặc dù đã xử lý slicing ở dưới)
    if pred.dim() == 3: pred = pred.unsqueeze(0)
    if target.dim() == 3: target = target.unsqueeze(0)
    return ssim(pred, target, data_range=1.0, size_average=True).item()

@torch.no_grad()
def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()

# =========================================================
# VISUAL
# =========================================================

def make_vis(inp, pred, gt):
    psnr = compute_psnr(pred, gt)
    ssim_score = compute_ssim(pred, gt)
    rmse = compute_rmse(pred, gt)

    def to_np(x):
        x = x.detach().float().cpu()
        if x.dim() == 4: x = x.squeeze(0) # Lột vỏ Batch để numpy vẽ
        x = torch.clamp(x, 0, 1)
        return x.permute(1, 2, 0).numpy()

    vis = np.concatenate([to_np(inp), to_np(pred), to_np(gt)], axis=1)
    return vis, {"psnr": psnr, "ssim": ssim_score, "rmse": rmse}

# =========================================================
# SLIDER LOGGER (CHẠY LÚC VALIDATION)
# =========================================================

def log_loop_steps(inp, gt, intermediate_preds, halting_weights, halt_logits, global_step, max_samples=2):
    T = len(intermediate_preds)
    B = min(inp.size(0), max_samples) 
    log_dict = {}

    for b in range(B):
        loop_images = [] 
        for t in range(T):
            # Slicing [b:b+1] giữ form 4D chuẩn
            inp_b = inp[b:b+1]
            pred_t = intermediate_preds[t][b:b+1]
            gt_b = gt[b:b+1]

            vis, metrics = make_vis(inp_b, pred_t, gt_b)

            exit_prob = torch.sigmoid(halt_logits[t][b]).mean().item()
            act_w = halting_weights[t][b].mean().item()

            caption = (f"Loop {t+1}/{T} | Exit Prob: {exit_prob*100:.1f}% | "
                       f"ACT W: {act_w:.3f} | P={metrics['psnr']:.2f} | S={metrics['ssim']:.3f}")
            loop_images.append(wandb.Image(vis, caption=caption))

        # Gom lại thành list để W&B tạo Slider
        log_dict[f"train/sample_{b}_loops"] = loop_images
    wandb.log(log_dict, step=global_step)

# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, cfg, aug=None, stage=1, global_step=0):
    model.train()
    running_loss = 0.0
    pbar = tqdm(loader, desc=f"Train {epoch}")

    for batch_idx, batch in enumerate(pbar):
        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        if aug is not None:
            inp, gt = aug(inp, gt)

        optimizer.zero_grad(set_to_none=True)

        with autocast(device_type="cuda", enabled=cfg.use_amp):
            pred, intermediate_preds, halting_weights, halt_logits = model(inp, return_all=True)

            if stage == 1:
                loss, loss_dict = criterion(
                    target=gt, final_pred=pred, 
                    intermediate_preds=intermediate_preds, halting_weights=halting_weights
                )
            else:
                loss, loss_dict = criterion(
                    target=gt, final_pred=pred, 
                    intermediate_preds=intermediate_preds, halt_logits=halt_logits, halting_weights=halting_weights
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Expected steps
        expected_steps = 0.0
        for t in range(halting_weights.size(0)):
            expected_steps += (t + 1) * halting_weights[t].mean()

        # Log Final Image mỗi 20 steps
        if (batch_idx + 1) % 20 == 0:
            model.eval()
            B_current = min(inp.size(0), 4) 
            imgs = []
            metrics_sum = {"psnr": 0, "ssim": 0, "rmse": 0}

            for i in range(B_current):
                vis, m = make_vis(inp[i:i+1], pred[i:i+1], gt[i:i+1])
                imgs.append(wandb.Image(vis, caption=f"FINAL | P={m['psnr']:.2f} | S={m['ssim']:.3f} | R={m['rmse']:.4f}"))
                for k in metrics_sum: metrics_sum[k] += m[k]

            for k in metrics_sum: metrics_sum[k] /= B_current

            wandb.log({
                "train/images_final": imgs,
                "train/loss": loss.item(),
                "train/psnr": metrics_sum["psnr"],
                "train/ssim": metrics_sum["ssim"],
                "train/rmse": metrics_sum["rmse"],
                "train/expected_steps": expected_steps.item(),
                "lr": optimizer.param_groups[0]["lr"],
            }, step=global_step)
            model.train()

        global_step += 1
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "steps": f"{expected_steps.item():.2f}"})

    return running_loss / len(loader), global_step

# =========================================================
# VALIDATION
# =========================================================

@torch.no_grad()
def validate(model, loader, device, epoch=0):
    model.eval()
    psnr_sum, ssim_sum = 0, 0

    for i, batch in enumerate(tqdm(loader, desc="Val")):
        inp = batch["inp"].to(device)
        gt = batch["gt"].to(device)

        if i == 0:
            outputs = model(inp, return_all=True)
            if len(outputs) == 4:
                pred, inter_preds, halt_w, halt_logits = outputs
                log_loop_steps(
                    inp=inp, gt=gt, intermediate_preds=inter_preds, 
                    halting_weights=halt_w, halt_logits=halt_logits, 
                    global_step=epoch, max_samples=2
                )
            else:
                pred = outputs[0] if isinstance(outputs, tuple) else outputs
        else:
            outputs = model(inp)
            pred = outputs[0] if isinstance(outputs, tuple) else outputs

        psnr_sum += compute_psnr(pred, gt)
        ssim_sum += compute_ssim(pred, gt)

    return psnr_sum / len(loader), ssim_sum / len(loader)

# =========================================================
# TRAIN LOOP CHÍNH
# =========================================================

def train_loop(model, train_loader, val_loader, optimizer, scheduler, criterion, device, cfg, aug=None, stage=1):
    scaler = GradScaler(enabled=cfg.use_amp)
    best_psnr = 0
    global_step = 0

    for epoch in range(cfg.epochs):
        train_loss, global_step = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion,
            device, epoch, cfg, aug=aug, stage=stage, global_step=global_step
        )

        val_psnr, val_ssim = validate(model, val_loader, device, epoch=epoch)
        scheduler.step()

        wandb.log({
            "epoch": epoch,
            "val/psnr": val_psnr,
            "val/ssim": val_ssim,
            "train/loss_epoch": train_loss,
            "global_step": global_step
        }, step=global_step)

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save(model.state_dict(), "best_model.pth")
            print("Saved best model")

        print(f"[{epoch}] loss={train_loss:.4f} psnr={val_psnr:.2f} ssim={val_ssim:.4f}")

    print("Done")