import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict
import torch.nn.functional as F
import numpy as np
import wandb

from src.training_utils.checkpoint_saver import CheckpointSaver
from src.training_utils.wandb import WandbLogger
from src.utils.unwarp import unwarp
    
class Trainer:
    """
    Trainer cho document dewarping.
    - Loader được truyền từ ngoài
    - Model là optional (truyền vào nếu đã có, không thì tự build)
    - CheckpointSaver KHÔNG nhận model trong __init__
    - Save/load dùng dict (model_dict) khi gọi hàm
    """

    def __init__(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        device: torch.device | str,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epochs: int = 100,
        lr: float = 2e-4,
        weight_decay: float = 1e-5,
        use_amp: bool = True,
        checkpoint_dir: str = "checkpoints",
        max_checkpoints: int = 5,
        saver: Optional[CheckpointSaver] = None,
        logger: Optional[WandbLogger] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        # Device
        self.device = torch.device(device) if isinstance(device, str) else device

        # Model
        self.model = model
        self.model.to(self.device)

        # Loss
        self.criterion = criterion
        if hasattr(self.criterion, 'to'):
            self.criterion.to(self.device)

        # Optimizer
        self.optimizer = optimizer if optimizer is not None else torch.optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )

        # Mixed Precision
        self.use_amp = use_amp
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # Training params
        self.epochs = epochs

        # DataLoaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Checkpoint Saver
        self.saver = saver if saver is not None else CheckpointSaver(
            save_dir=checkpoint_dir,
            max_ckpt=max_checkpoints,
        )

        # Logger (WandB optional)
        self.logger = logger
        if self.logger is not None:
            self.logger.watch(self.model, log="all", log_freq=200)

    def _run_phase(
        self,
        loader: torch.utils.data.DataLoader,
        is_train: bool = False,
        phase_name: str = "Eval"
    ) -> Dict[str, float]:
        if len(loader) == 0:
            print(f"Warning: {phase_name} loader empty!")
            return {"loss": float('inf')}

        self.model.train() if is_train else self.model.eval()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(loader, desc=phase_name, leave=is_train)

        for batch in pbar:
            img = batch[0].to(self.device)
            tgt = batch[1].to(self.device)

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda', enabled=self.use_amp):
                pred_bm, pred_mask = self.model(img)
                
                pred_unwarped = unwarp(img, pred_bm)
                gt_unwarped = unwarp(img, tgt)
                
                warped_img_np = None
                if self.criterion.lambda_curv > 0:
                    # Lấy ảnh đầu batch (RGB, uint8)
                    sample_img = img[0].cpu().permute(1, 2, 0).clamp(0, 1).numpy()
                    warped_img_np = (sample_img * 255).astype(np.uint8)     
                    
                loss, loss_dict = self.criterion(
                    pred_img=pred_unwarped,
                    gt_img=gt_unwarped,
                    pred_bm=pred_bm,
                    gt_bm=tgt,
                    warped_img_np=warped_img_np
                )
                
            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            if is_train:
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            
            # if self.logger and is_train:
            #     log_dict = {"train/" + k: v for k, v in loss_dict.items()}
            #     self.logger.log(log_dict)
                
        avg_loss = total_loss / num_batches if num_batches > 0 else float('inf')
        return {"loss": avg_loss}

    def train(self, start_epoch: int = 1):
        best_val_loss = float('inf')
        best_epoch = start_epoch - 1

        for epoch in range(start_epoch, self.epochs + 1):
            print(f"\n{'─' * 50}\nEpoch {epoch}/{self.epochs}")

            # Train phase
            train_metrics = self._run_phase(self.train_loader, is_train=True, phase_name="Train")

            # Val phase
            val_metrics = self._run_phase(self.val_loader, is_train=False, phase_name="Val")

            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Val   loss: {val_metrics['loss']:.4f}")

            # Logging
            if self.logger:
                self.logger.log({
                    "train/loss": train_metrics["loss"],
                    "val/loss": val_metrics["loss"],
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

            # Save checkpoint
            save_dict = {
                "model": self.model,
                "optimizer": self.optimizer,
            }

            self.saver.save(
                model_dict=save_dict,
                epoch=epoch,
                loss=val_metrics["loss"]
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_epoch = epoch
                print(f"-> Best val loss updated: {best_val_loss:.4f} (epoch {epoch})")

            if epoch % 1 == 0 or epoch == 1:
                print(f"--> Visualizing results at epoch {epoch}...")
                self._visualize_results(self.val_loader, epoch, phase="Val")
                
        # Test nếu có test_loader
        if self.test_loader is not None:
            test_metrics = self._run_phase(self.test_loader, is_train=False, phase_name="Test")
            print(f"  Test loss: {test_metrics['loss']:.4f}")
            if self.logger:
                self.logger.log({"test/loss": test_metrics["loss"]})

        print(f"\nTraining completed. Best val loss: {best_val_loss:.4f} @ epoch {best_epoch}")

    def resume_from_checkpoint(self, ckpt_path: str) -> int:
        """
        Load checkpoint và trả về epoch tiếp theo để tiếp tục train.
        """
        checkpoint = self.saver.load(
            ckpt_path=ckpt_path,
            model_dict={
                "model": self.model,
                "optimizer": self.optimizer,
            }
        )
        loaded_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from: {ckpt_path} | starting from epoch {loaded_epoch + 1}")
        return loaded_epoch + 1

    def get_model(self) -> torch.nn.Module:
        return self.model

    def get_model_state_dict(self) -> dict:
        return self.model.state_dict()
    
    def _visualize_results(self, loader, epoch, phase="Val"):
        """
        Lấy 1 batch từ loader, unwarp và log kết quả lên WandB.
        """
        self.model.eval()
        with torch.no_grad():
            # Lấy 1 batch đầu tiên
            batch = next(iter(loader))
            img = batch[0][:4].to(self.device)  # Lấy tối đa 4 ảnh để visualize
            tgt = batch[1][:4].to(self.device)

            with autocast(enabled=self.use_amp):
                pred_bm, _ = self.model(img)
                pred_unwarped = unwarp(img, pred_bm)
                gt_unwarped = unwarp(img, tgt)

            # Chuyển đổi tensor sang numpy để hiển thị (B, C, H, W) -> (B, H, W, C)
            def to_img(t):
                t = t.detach().cpu().permute(0, 2, 3, 1).numpy()
                t = np.clip(t, 0, 1)
                return (t * 255).astype(np.uint8)

            imgs_np = to_img(img)
            preds_np = to_img(pred_unwarped)
            gts_np = to_img(gt_unwarped)

            viz_list = []
            for i in range(len(imgs_np)):
                # Ghép 3 ảnh theo chiều ngang: Original | Prediction | GT
                combined = np.concatenate([imgs_np[i], preds_np[i], gts_np[i]], axis=1)
                viz_list.append(combined)

            # Log lên WandB
            if self.logger:
                import wandb
                # Ghép các cặp ảnh theo chiều dọc
                final_viz = np.concatenate(viz_list, axis=0)
                self.logger.log({
                    f"visualize/{phase}": wandb.Image(final_viz, caption=f"Epoch {epoch}: Input | Pred | GT")
                })