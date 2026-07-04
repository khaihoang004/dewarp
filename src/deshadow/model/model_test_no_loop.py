import torch
import torch.nn as nn
import torch.nn.functional as F

from src.deshadow.model.modules.repconv import RepConv3, RepConv7
from src.deshadow.model.modules.attention import DocumentAttn, RestormerAttention, MonarchAttention


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

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1)

class SwiGLU_FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Reduce number of hidden dim to 8/3 of normal FFN
        hidden_dim = int(dim * 8 / 3)
        self.w_v = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        w, v = torch.chunk(self.w_v(x), 2, dim=1)
        return self.project_out(F.silu(w) * v)

class ResidualRepConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups=1, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.conv_1 = RepConv3(in_channels, out_channels, groups, deploy=deploy)
        self.act_1 = nn.GELU()
        self.conv_2 = RepConv3(out_channels, out_channels, groups, deploy=deploy)
        self.act_2 = nn.GELU()

    def fuse(self):
        if not self.deploy:
            if hasattr(self.conv_1, 'fuse'): self.conv_1.fuse()
            if hasattr(self.conv_2, 'fuse'): self.conv_2.fuse()
            if hasattr(self.conv_1, 'deploy'): self.conv_1.deploy = True
            if hasattr(self.conv_2, 'deploy'): self.conv_2.deploy = True
            self.deploy = True

    def forward(self, x):
        out = self.act_1(self.conv_1(x))
        out = self.act_2(self.conv_2(out))
        return out + x

# Adaptive Bottleneck Architecture

class BottleneckBlock(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        assert dim % num_heads == 0, \
            f"dim ({dim}) must be divisible by num_heads ({num_heads})"

        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # -------------------------
        # Global Modeling
        # -------------------------
        self.norm1 = RMSNorm2d(dim)
        
        # Thay thế MonarchAttention bằng RestormerAttention
        self.attn = RestormerAttention(dim=dim, num_heads=num_heads)
        
        self.scale1 = LayerScale(dim, init_value=0.5)

        # -------------------------
        # Channel Mixing
        # -------------------------
        self.norm2 = RMSNorm2d(dim)
        self.ffn = SwiGLU_FFN(dim)
        self.scale2 = LayerScale(dim, init_value=0.5)

    def forward(self, x):
        # Không cần B, C, H, W ở đây nữa vì Restormer tự xử lý shape
        
        # ===== Global Attention =====
        x_norm = self.norm1(x)

        # Truyền trực tiếp x_norm [B, C, H, W] vào RestormerAttention
        # Toàn bộ phần flatten, transpose và reshape cồng kềnh đã được lược bỏ
        attn_out = self.attn(x_norm)

        x = x + self.scale1(attn_out)

        # ===== Channel Mixing =====
        x = x + self.scale2(self.ffn(self.norm2(x)))

        return x

class BottleneckLayer(nn.Module):
    def __init__(self, dim, num_blocks=1, num_heads=4, deploy=False):
        super().__init__()
        self.blocks = nn.ModuleList([BottleneckBlock(dim, num_heads=num_heads) for _ in range(num_blocks)])

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(),
        )

        self.exit_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1),
        )

    def forward(self, x, x_orig, step_embed=None):
        concat = torch.cat([x, x_orig], dim=1)
        fused = self.fusion(concat)
        
        if step_embed is not None:
            fused = fused + step_embed
            
        x = x + fused
        for block in self.blocks:
            x = block(x)

        exit_logit = self.exit_head(x).view(-1)
        return x, exit_logit

class SinglePassBottleneck(nn.Module):
    """
    Clean non-iterative bottleneck:
    - no loop
    - no halting head
    - no step embedding
    - no x/x_orig fusion for recurrent refinement
    """
    def __init__(self, dim, num_blocks=1, num_heads=4):
        super().__init__()
        self.blocks = nn.ModuleList([
            BottleneckBlock(dim, num_heads=num_heads)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
# ==========================================
# 3. Main Network Components
# ==========================================

class ShallowExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.conv = RepConv7(in_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))

class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, deploy=False):
        super().__init__()
        self.conv = RepConv3(in_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()
        
        self.stacked_layers = nn.ModuleList([
            ResidualRepConv(out_channels, out_channels, groups=1, deploy=deploy)
            for _ in range(num_blocks)
        ])
        
        self.norm_feat = RMSNorm2d(out_channels)
        self.down = nn.PixelUnshuffle(downscale_factor=2)
        self.channel_compress = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.act_down = nn.GELU()
        self.norm_down = RMSNorm2d(out_channels)

    def forward(self, x):
        feat = self.act(self.conv(x))
        for layer in self.stacked_layers:
            feat = layer(feat)

        skip_feat = self.norm_feat(feat)
        downsampled = self.down(skip_feat)
        downsampled = self.act_down(self.channel_compress(downsampled))
        return skip_feat, self.norm_down(downsampled)

class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=1, deploy=False):
        super().__init__()
        self.expand_channels = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1, bias=False)
        self.up = nn.PixelShuffle(upscale_factor=2)
        self.act_up = nn.GELU()
        
        self.gate_conv = nn.Conv2d(in_channels, skip_channels, kernel_size=1, bias=True)
        self.conv = RepConv3(in_channels + skip_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()
        
        self.stacked_layers = nn.ModuleList([
            ResidualRepConv(out_channels, out_channels, groups=1, deploy=deploy)
            for _ in range(num_blocks)
        ])
        
        self.norm_out = RMSNorm2d(out_channels)

    def forward(self, x, skip_feat):
        x = self.expand_channels(x)
        x = self.up(x)
        x = self.act_up(x)

        gate_mask = torch.sigmoid(self.gate_conv(x))
        gated_skip = skip_feat * gate_mask

        x = torch.cat([x, gated_skip], dim=1)
        x = self.act(self.conv(x))

        for layer in self.stacked_layers:
            x = layer(x)

        return self.norm_out(x)

# ==========================================
# 4. Final Model Architecture
# ==========================================
class NoLoopCleanRepDocEnhanceNet(nn.Module):
    """
    Baseline 1:
    Remove the looped bottleneck entirely and replace it with
    a single-pass bottleneck.
    """
    def __init__(
        self,
        base_dim=32,
        num_heads=4,
        bottleneck_blocks=1,
        enc_blocks=[1, 1, 2],
        dec_blocks=[1, 1, 2],
        deploy=False,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.deploy = deploy

        enc_blocks = [enc_blocks] * 3 if isinstance(enc_blocks, int) else enc_blocks
        dec_blocks = [dec_blocks] * 3 if isinstance(dec_blocks, int) else dec_blocks

        self.shallow_extractor = ShallowExtractor(3, base_dim, deploy=deploy)

        self.enc1 = EncoderStage(base_dim, base_dim, num_blocks=enc_blocks[0], deploy=deploy)
        self.enc2 = EncoderStage(base_dim, base_dim * 2, num_blocks=enc_blocks[1], deploy=deploy)
        self.enc3 = EncoderStage(base_dim * 2, base_dim * 4, num_blocks=enc_blocks[2], deploy=deploy)

        assert (base_dim * 4) % num_heads == 0, \
            f"base_dim*4 ({base_dim*4}) must be divisible by num_heads ({num_heads})"

        self.bottleneck = SinglePassBottleneck(
            dim=base_dim * 4,
            num_blocks=bottleneck_blocks,
            num_heads=num_heads,
        )

        self.dec3 = DecoderStage(base_dim * 4, base_dim * 4, base_dim * 2, num_blocks=dec_blocks[2], deploy=deploy)
        self.dec2 = DecoderStage(base_dim * 2, base_dim * 2, base_dim, num_blocks=dec_blocks[1], deploy=deploy)
        self.dec1 = DecoderStage(base_dim, base_dim, base_dim, num_blocks=dec_blocks[0], deploy=deploy)

        self.final_head = nn.Sequential(
            RepConv3(base_dim, base_dim, deploy=deploy),
            nn.GELU(),
            nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)
        )

    def _decode(self, bottleneck_feat, skip1, skip2, skip3, x_ori):
        d3 = self.dec3(bottleneck_feat, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        residual = self.final_head(d1)
        return x_ori + residual

    def forward(self, x):
        x_ori = x

        s0 = self.shallow_extractor(x)
        skip1, e1 = self.enc1(s0)
        skip2, e2 = self.enc2(e1)
        skip3, e3 = self.enc3(e2)

        bottleneck_feat = self.bottleneck(e3)
        output = self._decode(bottleneck_feat, skip1, skip2, skip3, x_ori)

        if not self.training:
            output = torch.clamp(output, 0.0, 1.0)

        return output

    @torch.no_grad()
    def fuse_entire_model(self):
        fused = []
        modules = list(self.named_modules())
        for name, module in reversed(modules):
            if (
                hasattr(module, 'fuse')
                and hasattr(module, 'deploy')
                and not getattr(module, 'deploy', True)
            ):
                module.fuse()
                fused.append(name)
                module.deploy = True

        print(">>> Fused NoLoopCleanRepDocEnhanceNet successfully.")