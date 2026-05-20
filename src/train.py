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
    if pred.dim() == 3:
        pred = pred.unsqueeze(0)
    if target.dim() == 3:
        target = target.unsqueeze(0)
        
    return ssim(pred, target, data_range=1.0, size_average=True).item()


@torch.no_grad()
def compute_rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2)).item()


# =========================================================
# VISUAL + METRICS
# =========================================================

def make_vis(inp, pred, gt):
    psnr = compute_psnr(pred, gt)
    ssim_score = compute_ssim(pred, gt)
    rmse = compute_rmse(pred, gt)

    def to_np(x):
        x = x.detach().float().cpu()
        x = torch.clamp(x, 0, 1)
        return x.permute(1, 2, 0).numpy()

    vis = np.concatenate(
        [to_np(inp), to_np(pred), to_np(gt)],
        axis=1
    )

    return vis, {
        "psnr": psnr,
        "ssim": ssim_score,
        "rmse": rmse
    }


# =========================================================
# LOOP STEP LOGGER (CORE PART)
# =========================================================

def log_loop_steps(inp, gt, intermediate_preds, halting_weights, global_step, max_samples=2):
    """
    Log full loop trajectory per sample
    """
    T = len(intermediate_preds)
    B = min(inp.size(0), max_samples)

    log_dict = {}

    for b in range(B):

        imgs = []
        psnr_curve = []
        ssim_curve = []
        rmse_curve = []
        halt_curve = []

        for t in range(T):

            pred_t = intermediate_preds[t][b]
            vis, metrics = make_vis(inp[b], pred_t, gt[b])

            halt = halting_weights[t][b].mean().item()

            imgs.append(
                wandb.Image(
                    vis,
                    caption=f"t={t} | "
                            f"P={metrics['psnr']:.2f} | "
                            f"S={metrics['ssim']:.3f} | "
                            f"R={metrics['rmse']:.4f} | "
                            f"H={halt:.3f}"
                )
            )

            psnr_curve.append(metrics["psnr"])
            ssim_curve.append(metrics["ssim"])
            rmse_curve.append(metrics["rmse"])
            halt_curve.append(halt)

        log_dict[f"loop/sample_{b}/images"] = imgs
        log_dict[f"loop/sample_{b}/psnr"] = psnr_curve
        log_dict[f"loop/sample_{b}/ssim"] = ssim_curve
        log_dict[f"loop/sample_{b}/rmse"] = rmse_curve
        log_dict[f"loop/sample_{b}/halt"] = halt_curve

    wandb.log(log_dict, step=global_step)


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(
    model, loader, optimizer, scaler, criterion,
    device, epoch, cfg, aug=None, stage=1, global_step=0
):

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

            (
                pred,
                intermediate_preds,
                halting_weights,
                halt_logits
            ) = model(inp, return_all=True)

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
                    halt_logits=halt_logits,
                    halting_weights=halting_weights
                )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # =========================================================
        # EXPECTED STEPS (halting insight)
        # =========================================================
        expected_steps = 0.0
        for t in range(halting_weights.size(0)):
            expected_steps += (t + 1) * halting_weights[t].mean()

        # =========================================================
        # LOG EVERY 20 STEPS
        # =========================================================
        if (batch_idx + 1) % 20 == 0:

            model.eval()

            # ---- loop-level logging (IMPORTANT) ----
            log_loop_steps(
                inp=inp,
                gt=gt,
                intermediate_preds=intermediate_preds,
                halting_weights=halting_weights,
                global_step=global_step,
                max_samples=2
            )

            # ---- final prediction logging ----
            B = inp.size(0)
            imgs = []
            metrics_sum = {"psnr": 0, "ssim": 0, "rmse": 0}

            for i in range(B):
                vis, m = make_vis(inp[i], pred[i], gt[i])

                imgs.append(
                    wandb.Image(
                        vis,
                        caption=f"FINAL | P={m['psnr']:.2f} | S={m['ssim']:.3f} | R={m['rmse']:.4f}"
                    )
                )

                for k in metrics_sum:
                    metrics_sum[k] += m[k]

            for k in metrics_sum:
                metrics_sum[k] /= B

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

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "steps": f"{expected_steps.item():.2f}"
        })

    return running_loss / len(loader), global_step


# =========================================================
# VALIDATION
# =========================================================

@torch.no_grad()
def validate(model, loader, device):

    model.eval()

    psnr_sum, ssim_sum = 0, 0
    sample = None

    for i, batch in enumerate(tqdm(loader, desc="Val")):

        inp = batch["inp"].to(device)
        gt = batch["gt"].to(device)

        pred = model(inp)

        psnr_sum += compute_psnr(pred, gt)
        ssim_sum += compute_ssim(pred, gt)

        if i == 0:
            sample = {
                "input": inp[:2].cpu(),
                "pred": pred[:2].cpu(),
                "gt": gt[:2].cpu()
            }

    return psnr_sum / len(loader), ssim_sum / len(loader), sample


# =========================================================
# TRAIN LOOP
# =========================================================

def train_loop(
    model, train_loader, val_loader,
    optimizer, scheduler, criterion,
    device, cfg, aug=None, stage=1
):

    scaler = GradScaler(enabled=cfg.use_amp)
    best_psnr = 0
    global_step = 0

    for epoch in range(cfg.epochs):

        train_loss, global_step = train_one_epoch(
            model, train_loader,
            optimizer, scaler, criterion,
            device, epoch, cfg,
            aug=aug, stage=stage,
            global_step=global_step
        )

        val_psnr, val_ssim, sample = validate(model, val_loader, device)
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