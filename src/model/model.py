import torch.nn as nn

class LightUnwarp(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # ... paste toàn bộ model code của bạn ở đây

def get_model(cfg):
    if cfg.model.name == "light_unwarp":
        return LightUnwarp(cfg)
    raise ValueError(...)