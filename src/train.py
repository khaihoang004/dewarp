import torch
import wandb
from tqdm import tqdm
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
    return ssim(
        pred,
        target,
        data_range=1.0,
        size_average=True
    ).item()

# =========================================================
# TRAIN
# =========================================================

def train_one_epoch(
    model, loader, optimizer, scaler, criterion, device, epoch, cfg, aug=None, stage=1, global_step=0
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

            # Đón cả total_loss và loss_dict từ hàm loss đã cập nhật
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

        # Tính expected steps để hiển thị tqdm và log
        expected_steps = 0.0
        for t in range(halting_weights.size(0)):
            expected_steps += ((t + 1) * halting_weights[t].mean())

        # ====================================================
        # LOGGING MỖI 20 BATCH (STEPS) VÀO WANDB
        # ====================================================
        if (batch_idx + 1) % 20 == 0:
            log_data = loss_dict  # Chứa rec_loss, kl_loss, gate_loss... tùy stage
            log_data["train/step_expected_steps"] = expected_steps.item()
            log_data["global_step"] = global_step
            log_data["lr"] = optimizer.param_groups[0]['lr']
            
            wandb.log(log_data, step=global_step)

        global_step += 1  # Tăng biến đếm tổng

        pbar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "steps": f"{expected_steps.item():.2f}"
        })

    return running_loss / len(loader), global_step

# =========================================================
# VALIDATE
# =========================================================

@torch.no_grad()
def validate(model, loader, device):
    model.eval()

    total_psnr = 0.0
    total_ssim = 0.0
    sample_images = None

    pbar = tqdm(loader, desc="Validate")

    for i, batch in enumerate(pbar):
        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        # Lúc eval, model tự động clamp giá trị về [0, 1] rồi
        pred = model(inp)

        psnr = compute_psnr(pred, gt)
        ssim_score = compute_ssim(pred, gt)

        total_psnr += psnr
        total_ssim += ssim_score

        pbar.set_postfix({
            "PSNR": f"{psnr:.2f}",
            "SSIM": f"{ssim_score:.4f}"
        })

        if i == 0:
            sample_images = {
                "input": inp[:2].cpu(),
                "pred": pred[:2].cpu(),
                "gt": gt[:2].cpu()
            }

    return total_psnr / len(loader), total_ssim / len(loader), sample_images

# =========================================================
# IMAGE LOGGING & TRAIN LOOP 
# (Phần này giữ nguyên logic của bạn, chỉ làm gọn)
# =========================================================

def log_images(sample_images, epoch):
    imgs = []
    inp = sample_images["input"]
    pred = sample_images["pred"]
    gt = sample_images["gt"]

    B = inp.size(0)
    for i in range(B):
        vis = torch.cat([inp[i], pred[i], gt[i]], dim=2)
        imgs.append(wandb.Image(vis, caption=f"Epoch {epoch}"))

    wandb.log({"samples": imgs})


def train_loop(
    model, train_loader, val_loader, optimizer, scheduler, criterion, device, cfg, aug=None, stage=1
):
    scaler = GradScaler(enabled=cfg.use_amp)
    best_psnr = 0.0
    
    global_step = 0  # Khởi tạo global_step để theo dõi xuyên suốt các epochs

    for epoch in range(cfg.epochs):
        
        # Nhận lại global_step từ epoch này để truyền tiếp cho epoch sau
        train_loss, global_step = train_one_epoch(
            model=model, loader=train_loader, optimizer=optimizer, scaler=scaler,
            criterion=criterion, device=device, epoch=epoch, cfg=cfg, aug=aug, stage=stage,
            global_step=global_step
        )

        val_psnr, val_ssim, sample_images = validate(model, val_loader, device)
        scheduler.step()

        # Log ở cấp độ Epoch
        wandb.log({
            "epoch": epoch,
            "epoch/train_loss": train_loss,
            "val/psnr": val_psnr,
            "val/ssim": val_ssim,
            "global_step": global_step
        }, step=global_step)

        if sample_images is not None:
            log_images(sample_images, epoch)

        print(
            f"[{epoch}] Loss: {train_loss:.4f} | "
            f"PSNR: {val_psnr:.2f} | SSIM: {val_ssim:.4f}"
        )

        if val_psnr > best_psnr:
            best_psnr = val_psnr
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "psnr": val_psnr
            }, "best_model.pth")
            print(">>> Saved best model")

    print("Training finished")