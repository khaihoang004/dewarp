import torch
import wandb
from tqdm import tqdm
import numpy as np
from torch.amp import autocast, GradScaler
from pytorch_msssim import ms_ssim, ssim


# =========================================================
# METRICS (TÍNH TOÁN THEO TỪNG ẢNH ĐƠN LẺ)
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
# VISUAL + LOGGING WANDB
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


def log_loop_steps(inp, gt, intermediate_preds, halting_weights, halt_logits, global_step, max_samples=4):
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
                      f"SSIM={metrics['ssim']:.4f} | MS-SIM={metrics['ms_ssim']:.4f}")

            loop_images.append(wandb.Image(vis, caption=caption))

        log_dict[f"val/sample_{b}_progress"] = loop_images

    wandb.log(log_dict, step=global_step)


# =========================================================
# 🛑 HỆ THỐNG CHẨN ĐOÁN SỐ HỌC TERMINAL
# =========================================================

def inspect_loop_dynamics(intermediate_preds, halting_weights, halt_logits):
    """Kiểm tra sự tiến hóa số học giữa các vòng lặp để cô lập lỗi hiển thị."""
    print("\n" + "="*30 + " TRÌNH CHẨN ĐOÁN LÕI LẶP ĐỘNG " + "="*30)
    T = len(intermediate_preds)
    print(f"[*] Tổng số vòng lặp nhận được từ Model: {T}")
    
    if T > 1:
        diff_1_2 = (intermediate_preds[1] - intermediate_preds[0]).abs().max().item()
        diff_last_first = (intermediate_preds[-1] - intermediate_preds[0]).abs().max().item()
        
        print(f"[*] Sai lệch toán học Max (|Loop 2 - Loop 1|): {diff_1_2:.8f}")
        print(f"[*] Sai lệch toán học Max (|Loop Cuối - Loop 1|): {diff_last_first:.8f}")
        
        p_min = intermediate_preds[0].min().item()
        p_max = intermediate_preds[0].max().item()
        print(f"[-] Dải giá trị thô của Loop 1 (Chưa clamp) : Lớn nhất = {p_max:.4f} | Nhỏ nhất = {p_min:.4f}")

        if diff_last_first == 0.0:
            print("\n[🚨 KẾT LUẬN]: CÁC TENSOR TRÙNG NHAU 100% VỀ MẶT TOÁN HỌC!")
            print("   -> Lõi Bottleneck đang bị đóng băng hoàn toàn qua các vòng.")
        elif diff_last_first < 1e-4:
            print("\n[🚨 KẾT LUẬN]: CÓ SỰ KHÁC BIỆT NHƯNG SIÊU NHỎ (< 0.0001)!")
            print("   -> Đặc trưng bị 'nuốt chửng' bởi nhánh Skip Connection quá lớn từ Encoder gửi sang.")
        else:
            print("\n[✅ KẾT LUẬN]: VỀ MẶT TOÁN HỌC CÁC TENSOR KHÁC NHAU RÕ RÀNG!")
            print("   -> Đồ thị mạng lặp chạy chuẩn. Kiểm tra lại giá trị khởi tạo LayerScale hoặc cờ AMP.")
            
    if halting_weights is not None:
        print(f"[-] Phân phối Trọng số Dừng (Halt Weights Avg): {[f'{w.mean().item():.3f}' for w in halting_weights]}")
    print("="*88 + "\n")


def inspect_gradient_flow(model):
    """Kiểm tra độ lớn Đạo hàm (Grad) để xác định xem tầng Bottleneck có được tối ưu không."""
    print("="*30 + " KIỂM TRA ĐỒ THỊ GRADIENT " + "="*30)
    has_bottleneck_grad = False
    
    for name, param in model.named_parameters():
        if "bottleneck" in name:
            if param.grad is not None:
                grad_norm = param.grad.abs().sum().item()
                print(f"[GRAD] Tầng lặp: {name:<50} | Tổng Grad tích lũy = {grad_norm:.8f}")
                if grad_norm > 0:
                    has_bottleneck_grad = True
            else:
                print(f"[GRAD] Tầng lặp: {name:<50} | Không tìm thấy Đạo hàm (Grad là None)!")
                
    if not has_bottleneck_grad:
        print("\n[🚨 NGUY HIỂM]: Toàn bộ khối Bottleneck nhận Đạo hàm bằng 0 hoặc None!")
    else:
        print("\n[✅ ỔN ĐỊNH]: Khối lặp có nhận được tín hiệu điều chỉnh đạo hàm từ hàm Loss.")
    print("="*88 + "\n")


# =========================================================
# TRAIN ONE EPOCH
# =========================================================

def train_one_epoch(model, loader, optimizer, scaler, criterion, device, epoch, cfg, aug=None, stage=1, global_step=0):
    model.train()
    running_loss = 0.0

    pbar = tqdm(
        loader,
        desc=f"Train Epoch {epoch} | Stage {stage}"
    )

    for batch_idx, batch in enumerate(pbar):
        inp = batch["inp"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        if aug is not None:
            inp, gt = aug(inp, gt)

        optimizer.zero_grad(set_to_none=True)

        with autocast(
            device_type="cuda",
            enabled=cfg.use_amp
        ):
            outputs = model(inp, return_all=True)
            pred, intermediate_preds, halting_weights, gate_logits, exit_probs = outputs

            # Kiểm tra đột biến toán học ở Batch đầu tiên của Epoch
            if batch_idx == 0:
                inspect_loop_dynamics(intermediate_preds, halting_weights, gate_logits)

            if stage == 1:
                loss, loss_dict = criterion(
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

        scaler.scale(loss).backward()
        
        # Kiểm tra dòng chảy đạo hàm trước khi Optimizer bước đi
        if batch_idx == 0:
            inspect_gradient_flow(model)

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        # Tính toán các chỉ số vòng lặp để log
        halt_stack = torch.stack([
            h.mean(dim=tuple(range(1, h.dim())))
            for h in halting_weights
        ], dim=0)

        step_ids = torch.arange(
            1,
            len(intermediate_preds) + 1,
            device=halt_stack.device
        ).float().view(-1, 1)

        expected_steps = (step_ids * halt_stack).sum(dim=0).mean().item()
        hard_steps = halt_stack.argmax(dim=0).float().mean().item() + 1

        if batch_idx == len(loader) - 1:
            wandb.log({
                "train/loss": loss.item(),
                "train/expected_steps": expected_steps,
                "train/hard_steps": hard_steps,
                "lr": optimizer.param_groups[0]["lr"],
                **{f"train/{k}": (v.item() if torch.is_tensor(v) else v) for k, v in loss_dict.items()}
            }, step=global_step)

        global_step += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}", steps=f"{expected_steps:.2f}")

    return running_loss / len(loader), global_step


# =========================================================
# VALIDATION (ĐÃ SỬA: THEO DÕI CHỈ SỐ TỪNG VÒNG LẶP TOÀN TẬP VAL)
# =========================================================

@torch.no_grad()
def validate(model, loader, device, global_step=0, log_images=True):
    model.eval()

    loop_psnr_sums = None
    loop_ssim_sums = None
    
    psnr_sum = 0.0
    ssim_sum = 0.0
    ms_ssim_sum = 0.0
    rmse_sum = 0.0
    total_images = 0

    for i, batch in enumerate(tqdm(loader, desc="Validating Progressively")):
        inp = batch["inp"].to(device)
        gt = batch["gt"].to(device)

        # Ép mô hình trả về toàn bộ các vòng lặp để tính toán chỉ số đối chứng
        outputs = model(inp, return_all=True)
        pred, inter_preds, halt_w, halt_logits, _ = outputs

        T = len(inter_preds)
        batch_size = inp.size(0)

        # Khởi tạo mảng tích lũy động theo số lượng loop thực tế
        if loop_psnr_sums is None:
            loop_psnr_sums = [0.0] * T
            loop_ssim_sums = [0.0] * T

        # Phân rã batch tính toán độc lập từng ảnh (Tránh hiện tượng lệch trung bình log)
        for b in range(batch_size):
            g_img = gt[b:b+1]
            
            psnr_sum += compute_psnr(pred[b:b+1], g_img)
            ssim_sum += compute_ssim(pred[b:b+1], g_img)
            ms_ssim_sum += compute_ms_ssim(pred[b:b+1], g_img)
            rmse_sum += compute_rmse(pred[b:b+1], g_img)
            
            # Tính toán chỉ số riêng cho từng bước sửa đổi tịnh tiến
            for t in range(T):
                loop_psnr_sums[t] += compute_psnr(inter_preds[t][b:b+1], g_img)
                loop_ssim_sums[t] += compute_ssim(inter_preds[t][b:b+1], g_img)
                
            total_images += 1

        # Chỉ log ảnh mẫu cho batch đầu tiên lên giao diện Wandb
        if i == 0 and log_images:
            log_loop_steps(inp, gt, inter_preds, halt_w, halt_logits, global_step)

    # Đóng gói chỉ số trung bình tổng của trạng thái Final
    val_metrics = {
        "val/psnr": psnr_sum / total_images,
        "val/ssim": ssim_sum / total_images,
        "val/ms_ssim": ms_ssim_sum / total_images,
        "val/rmse": rmse_sum / total_images,
    }
    
    # Bung chỉ số của từng Loop lên hệ thống biểu đồ Wandb để theo dõi xu hướng
    for t in range(T):
        val_metrics[f"val_loops/loop_{t+1}_psnr"] = loop_psnr_sums[t] / total_images
        val_metrics[f"val_loops/loop_{t+1}_ssim"] = loop_ssim_sums[t] / total_images

    return val_metrics


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

        # Log epoch summary (Tự động cập nhật đồ thị các đường loop_1, loop_2,...)
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

        print(f"[{epoch:3d}] Loss: {train_loss:.4f} | PSNR: {current_psnr:.3f} | SSIM: {val_metrics['val/ssim']:.4f}")

    print("Training completed!")
    return model