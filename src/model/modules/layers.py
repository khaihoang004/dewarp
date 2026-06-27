import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return norm * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

class LayerNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(dim=1, keepdim=True)
        s = (x - u).pow(2).mean(dim=1, keepdim=True)
        x = (x - u) * torch.rsqrt(s + self.eps)
        return x * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1) + self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

class SwiGLU_FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Giảm hệ số ẩn xuống 8/3 (~2.67) để bảo toàn lượng tham số
        hidden_dim = int(dim * 8 / 3)
        
        self.w_v = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        w, v = torch.chunk(self.w_v(x), 2, dim=1)
        # Swish(w) * v
        return self.project_out(F.silu(w) * v)

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1)