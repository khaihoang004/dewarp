import os
import glob
import torch

class CheckpointSaver:
    """Saves model checkpoints during training based on specified criteria."""
    def __init__(self, save_dir='checkpoints', max_ckpt=5):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.max_ckpt = max_ckpt

    def save(self, model_dict, epoch, loss):
        """Saves the model checkpoint."""
        ckpt_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}_loss_{loss:.4f}.pt')
        # Create a clean dictionary for saving
        save_dict = {}
        for k, v in model_dict.items():
            if hasattr(v, "state_dict"):
                save_dict[k] = v.state_dict()
            else:
                save_dict[k] = v  # e.g., scalars, metrics, etc.
        save_dict['epoch'] = epoch
        save_dict['loss'] = loss
        torch.save(save_dict, ckpt_path)
        print(f"Checkpoint saved: {ckpt_path}")
        # Optionally limit number of saved checkpoints
        self._cleanup_old_checkpoints()

    def load(self, ckpt_path, model_dict=None):
        """Loads a model checkpoint and optionally restores model/optimizer states."""
        if not os.path.isfile(ckpt_path):
            raise FileNotFoundError(f"No checkpoint found at: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        print(f"Checkpoint loaded: {ckpt_path}")
        if model_dict is not None:
            for k, v in model_dict.items():
                if k in checkpoint and hasattr(v, "load_state_dict"):
                    v.load_state_dict(checkpoint[k])
        return checkpoint

    def _cleanup_old_checkpoints(self):
        """Remove older checkpoints to maintain max_ckpt limit."""
        ckpts = sorted(glob.glob(os.path.join(self.save_dir, 'checkpoint_epoch_*.pt')), key=os.path.getmtime)
        if len(ckpts) > self.max_ckpt:
            for f in ckpts[:-self.max_ckpt]:
                os.remove(f)
                print(f"Old checkpoint removed: {f}")
