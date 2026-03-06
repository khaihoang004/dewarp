# src/trainer.py
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Dict

from ..config import Config
from src.training_utils.checkpoint_saver import CheckpointSaver
from src.training_utils.wandb import WandbLogger


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
        cfg: Config,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        model: nn.Module,
        device,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        saver: Optional[CheckpointSaver] = None,
        logger: Optional[WandbLogger] = None,
        test_loader: Optional[torch.utils.data.DataLoader] = None,
    ):
        self.cfg = cfg
        self.device = cfg.device

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
            lr=cfg.lr,
            weight_decay=getattr(cfg, 'weight_decay', 1e-5),
        )

        # Mixed Precision
        self.scaler = GradScaler(enabled=getattr(cfg, 'amp', True))

        # DataLoaders
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Checkpoint Saver - KHÔNG truyền model vào đây
        self.saver = saver if saver is not None else CheckpointSaver(
            save_dir=str(Path(cfg.checkpoint_dir)),
            max_ckpt=getattr(cfg, 'max_checkpoints', 5),
        )

        # Logger
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
            img = batch["image"].to(self.device)
            tgt = batch["target"].to(self.device)

            if is_train:
                self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.cfg.amp):
                pred = self.model(img)
                loss = self.criterion(pred, tgt)

            if is_train:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

            total_loss += loss.item()
            num_batches += 1

            if is_train:
                pbar.set_postfix(loss=f"{loss.item():.4f}")

        return {"loss": total_loss / num_batches}

    def train(self, start_epoch: int = 1):
        best_val_loss = float('inf')
        best_epoch = start_epoch - 1

        for epoch in range(start_epoch, self.cfg.epochs + 1):
            print(f"\n{'─' * 50}\nEpoch {epoch}/{self.cfg.epochs}")

            # Train phase
            train_metrics = self._run_phase(self.train_loader, is_train=True, phase_name="Train")

            # Val phase
            val_metrics = self._run_phase(self.val_loader, is_train=False, phase_name="Val")

            print(f"  Train loss: {train_metrics['loss']:.4f}")
            print(f"  Val   loss: {val_metrics['loss']:.4f}")

            if self.logger:
                self.logger.log({
                    "train/loss": train_metrics["loss"],
                    "val/loss": val_metrics["loss"],
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]["lr"],
                })

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
                print(f"  → Best val loss updated: {best_val_loss:.4f} (epoch {epoch})")

        if self.test_loader is not None:
            test_metrics = self._run_phase(self.test_loader, is_train=False, phase_name="Test")
            print(f"  Test loss: {test_metrics['loss']:.4f}")
            if self.logger:
                self.logger.log({"test/loss": test_metrics["loss"]})

        print(f"\nTraining completed. Best val loss: {best_val_loss:.4f} @ epoch {best_epoch}")

    def resume_from_checkpoint(self, ckpt_path: str) -> int:
        """
        Load checkpoint và trả về epoch tiếp theo.
        """
        checkpoint = self.saver.load(
            ckpt_path=ckpt_path,
            model_dict={
                "model": self.model,
                "optimizer": self.optimizer,
            }
        )
        loaded_epoch = checkpoint.get("epoch", 0)
        print(f"Resumed from: {ckpt_path} | epoch {loaded_epoch}")
        return loaded_epoch + 1