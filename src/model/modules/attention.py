import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from src.model.monarch_attn import MonarchAttention


class DocumentStripAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv_w1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.dwconv_w = nn.Conv2d(dim, dim, kernel_size=(1, 9), padding=(0, 4), groups=dim)
        
        self.conv_h1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.dwconv_h = nn.Conv2d(dim, dim, kernel_size=(9, 1), padding=(4, 0), groups=dim)
        
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)

    def forward(self, x):
        x_w = self.dwconv_w(self.conv_w1(x))
        x_h = self.dwconv_h(self.conv_h1(x))
        return self.proj(F.gelu(x_w + x_h))

class RepAttention(nn.Module):
    """ Re-parameterizable Attention Block using MonarchAttention """
    def __init__(self, dim, num_heads=8, block_size=16, num_steps=2, pad_type="pre", impl="torch", deploy=False):
        super().__init__()
        self.num_heads = num_heads
        self.deploy = deploy
        
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        
        self.monarch_attn = MonarchAttention(
            block_size=block_size,
            num_steps=num_steps,
            pad_type=pad_type,
            impl=impl
        )

        self.attn_fn = self.monarch_attn if deploy else self.common_attn

    def common_attn(self, q, k, v):
        return F.scaled_dot_product_attention(q, k, v)

    @torch.no_grad()
    def fuse(self, delete_branches: bool = True):
        if not self.deploy:
            self.attn_fn = self.monarch_attn
            self.deploy = True

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(x)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head (h w) c', head=self.num_heads)
        attn_out = self.attn_fn(q, k, v)
        attn_out = rearrange(attn_out, 'b head (h w) c -> b (head c) h w', head=self.num_heads, h=H, w=W)
        out = self.proj(attn_out)
        return out