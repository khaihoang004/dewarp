import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.modules.repconv import *
from src.model.modules.attention import RepAttention


class RMSNorm2d(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return norm * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1)

class ResidualRepConv(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.conv_1 = RepDWSeparable3BN(in_channels, out_channels, deploy=deploy)
        self.conv_2 = RepDWSeparable3BN(out_channels, out_channels, deploy=deploy)

    def fuse(self, delete_branches: bool = True):
        if not self.deploy:
            self.conv_1.fuse(delete_branches)
            self.conv_2.fuse(delete_branches)
            self.deploy = True

    def forward(self, x):
        out = self.conv_1(x)
        out = self.conv_2(out)
        return out + x

class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, deploy=False):
        super().__init__()
        self.conv = RepConv3BN(in_channels, out_channels, groups=1, deploy=deploy)
        self.act = nn.GELU()

        self.stacked_layers = nn.ModuleList([
            ResidualRepConv(out_channels, out_channels, deploy=deploy)
            for _ in range(num_blocks)
        ])
        self.norm_feat = RMSNorm2d(out_channels)

        self.down = nn.PixelUnshuffle(downscale_factor=2)
        self.channel_compress = RepPointwiseBN(out_channels * 4, out_channels, deploy=deploy)
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
        self.expand_channels = RepConv3BN(in_channels, in_channels * 4, groups=1, deploy=deploy)
        self.up = nn.PixelShuffle(upscale_factor=2)
        self.act_up = nn.GELU()

        self.gate_conv = RepPointwiseBN(in_channels, skip_channels, deploy=deploy)

        self.conv = RepConv3BN(in_channels + skip_channels, out_channels, groups=1, deploy=deploy)
        self.act = nn.GELU()

        self.stacked_layers = nn.ModuleList([
            ResidualRepConv(out_channels, out_channels, deploy=deploy)
            for _ in range(num_blocks)
        ])

        self.norm_out = RMSNorm2d(out_channels)

    def forward(self, x, skip_feat):
        # expand_channels -> PixelShuffle -> Act
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


# Bottleneck
class SwiGLU_FFN(nn.Module):
    def __init__(self, dim, deploy=False):
        super().__init__()
        self.deploy = deploy
        # Giảm hệ số ẩn xuống 8/3 (~2.67) để bảo toàn lượng tham số của tầng FFN thông thường
        hidden_dim = int(dim * 8 / 3)
        
        # Thay thế Conv2d thuần bằng RepPointwiseBN nâng cao
        self.w_v = RepPointwiseBN(dim, hidden_dim * 2, deploy=deploy)
        self.project_out = RepPointwiseBN(hidden_dim, dim, deploy=deploy)

    def fuse(self, delete_branches: bool = True):
        if self.deploy: return
        self.w_v.fuse(delete_branches)
        self.project_out.fuse(delete_branches)
        self.deploy = True

    def forward(self, x):
        # Chia đôi tensor đầu ra thành 2 nhánh W và V
        w, v = torch.chunk(self.w_v(x), 2, dim=1)
        # Công thức: Swish(w) * v
        return self.project_out(F.silu(w) * v)


class BottleneckBlock(nn.Module):
    def __init__(self, dim, num_heads=4, deploy=False, **attn_kwargs):
        super().__init__()
        self.deploy = deploy
        
        # 1. Global Branch
        self.norm1 = RMSNorm2d(dim)
        self.attn = RepAttention(dim=dim, num_heads=num_heads, deploy=deploy, **attn_kwargs)
        self.scale1 = LayerScale(dim, init_value=0.5)

        # 2. Local Branch
        self.conv = DWRepConv3BN(channels=dim, deploy=deploy)
        self.scale2 = LayerScale(dim, init_value=0.5)

        # 3. Channel Branch
        self.norm3 = RMSNorm2d(dim)
        self.ffn = SwiGLU_FFN(dim, deploy=deploy)
        self.scale3 = LayerScale(dim, init_value=0.5)

    def fuse(self, delete_branches: bool = True):
        if self.deploy: return
        self.attn.fuse(delete_branches)
        self.conv.fuse(delete_branches)
        self.ffn.fuse(delete_branches)
        self.deploy = True

    def forward(self, x):
        # 1. Global Attention
        x = x + self.scale1(self.attn(self.norm1(x)))

        # 2. Local Convolution
        x = x + self.scale2(self.conv(x))

        # 3. Channel FFN
        x = x + self.scale3(self.ffn(self.norm3(x)))
        return x

class BottleneckLayer(nn.Module):
    def __init__(self, dim, num_blocks=1, num_heads=4, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.blocks = nn.ModuleList([
            BottleneckBlock(dim, num_heads=num_heads, deploy=deploy)
            for _ in range(num_blocks)
        ])
        self.fusion_pw = RepPointwiseBN(dim * 2, dim, deploy=deploy)
        self.fusion_dw = DWRepConv3BN(dim, deploy=deploy)
        self.act = nn.GELU()
        self.exit_head_1 = nn.Conv2d(dim, dim // 4, kernel_size=1, bias=True)
        self.exit_head_2 = nn.Conv2d(dim // 4, 1, kernel_size=1, bias=True)
 
    def fuse(self, delete_branches=True):
        if self.deploy:
            return
        self.fusion_pw.fuse(delete_branches)
        self.fusion_dw.fuse(delete_branches)

        for b in self.blocks:
            b.fuse(delete_branches)
        self.deploy = True
 
    def forward(self, x, x_orig, step_embed=None):
        concat = torch.cat([x, x_orig], dim=1)
        fused = self.act(self.fusion_pw(concat))
        fused = self.act(self.fusion_dw(fused))

        if step_embed is not None:
            fused = fused + step_embed

        x = x + fused

        for block in self.blocks:
            x = block(x)

        # Exit head
        exit_feat = F.adaptive_avg_pool2d(x, 1) # (B, C, 1, 1)
        exit_feat = self.act(self.exit_head_1(exit_feat)) # (B, dim//4, 1, 1)
        exit_logit = self.exit_head_2(exit_feat) # (B, 1, 1, 1)
        
        exit_logit = exit_logit.view(exit_logit.size(0)) 

        return x, exit_logit
 
 
class AdaptiveLoopedBottleneck(nn.Module):
    def __init__(self, dim, max_loops=6, num_heads=4, deploy=False):
        super().__init__()
        self.deploy = deploy
        self.max_loops = max_loops
        self.layer = BottleneckLayer(dim, num_heads=num_heads, deploy=deploy)
        self.step_embeddings = nn.Parameter(
            torch.zeros(max_loops, 1, dim, 1, 1)
        )
        nn.init.normal_(self.step_embeddings, std=0.02)
 
    def fuse(self, delete_branches=True):
        if self.deploy:
            return
        self.layer.fuse(delete_branches)
        self.deploy = True

    def _build_halting_distribution(self, exit_logits):
        p = torch.sigmoid(exit_logits)  # (T, B)
        T, B = p.shape
        
        one_minus_p = 1.0 - p
        cumprod = torch.cat([torch.ones(1, B, device=p.device), one_minus_p[:-1]], dim=0)
        S = torch.cumprod(cumprod, dim=0)
        h = p * S
        
        # Absorb leftover probability into (T-1)
        remainder = 1.0 - h.sum(dim=0, keepdim=True)
        h[-1] += remainder.squeeze(0)
        
        return p, S, h

    def forward(self, x, halt_threshold=0.85, return_all=False):

        B = x.shape[0]
        x_orig = x.clone()

        states = []
        exit_logits = []

        state = x

        for t in range(self.max_loops):
            state, logit = self.layer(
                state,
                x_orig,
                self.step_embeddings[t]
            )

            states.append(state)
            exit_logits.append(logit)

        exit_logits = torch.stack(exit_logits, dim=0)  # (T, B)

        if self.training or return_all:
            p, S, h = self._build_halting_distribution(exit_logits)

            return {
                "final_state": state,
                "states": states,
                "exit_logits": exit_logits,   # raw
                "exit_prob": p,               # p_t
                "survival": S,                # S_t
                "halting": h                  # h_t (distribution)
            }
        state = x
        survival = torch.ones(B, device=x.device)

        for t in range(self.max_loops):
            state, logit = self.layer(
                state,
                x_orig,
                self.step_embeddings[t]
            )

            p = torch.sigmoid(logit)  # (B,)

            survival = survival * (1.0 - p)

            if (1.0 - survival).mean() > halt_threshold:
                break

        return state
    
 
class ShallowExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.conv = RepConv7BN(in_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()
 
    def forward(self, x):
        return self.act(self.conv(x))
 
 
class LoopRepDocEnhanceNet(nn.Module):
    def __init__(
        self,
        base_dim=32,
        max_loops=4,
        num_heads=4,
        enc_blocks=[2, 2, 2],
        dec_blocks=[2, 2, 2],
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
 
        self.bottleneck = AdaptiveLoopedBottleneck(
            dim=base_dim * 4,
            max_loops=max_loops,
            num_heads=num_heads,
            deploy=deploy,
        )
 
        self.dec3 = DecoderStage(base_dim * 4, base_dim * 4, base_dim * 2, num_blocks=dec_blocks[2], deploy=deploy)
        self.dec2 = DecoderStage(base_dim * 2, base_dim * 2, base_dim, num_blocks=dec_blocks[1], deploy=deploy)
        self.dec1 = DecoderStage(base_dim, base_dim, base_dim, num_blocks=dec_blocks[0], deploy=deploy)
 
        self.final_head = nn.Sequential(
            RepConv3BN(base_dim, base_dim, groups=1, deploy=deploy),
            nn.GELU(),
            nn.Conv2d(base_dim, 3, kernel_size=3, padding=1),
        )
 
    def _decode(self, bottleneck_feat, skip1, skip2, skip3, x_ori):
        d3 = self.dec3(bottleneck_feat, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        residual = self.final_head(d1)
        output = x_ori + residual
        return torch.clamp(output, 0.0, 1.0)
 
    def forward(self, x, halt_threshold=0.80, return_all=False):
        x_ori = x
 
        s0 = self.shallow_extractor(x)
        skip1, e1 = self.enc1(s0)
        skip2, e2 = self.enc2(e1)
        skip3, e3 = self.enc3(e2)
 
        bottleneck_out = self.bottleneck(
            e3,
            halt_threshold=halt_threshold,
            return_all=True,
        )
 
        if isinstance(bottleneck_out, dict):
            final_feat = bottleneck_out["final_state"]
        else:
            final_feat = bottleneck_out

        if self.training or return_all:
            if isinstance(bottleneck_out, dict):
                states = bottleneck_out["states"]
                exit_logits = bottleneck_out["exit_logits"]
                exit_prob = bottleneck_out["exit_prob"]
                survival = bottleneck_out["survival"]
                halting = bottleneck_out["halting"]

                final_out = self._decode(final_feat, skip1, skip2, skip3, x_ori)
                intermediate_preds = [self._decode(s, skip1, skip2, skip3, x_ori) for s in states]

                return {
                    "final": final_out,
                    "intermediate": intermediate_preds,
                    "exit_logits": exit_logits,
                    "exit_prob": exit_prob,
                    "survival": survival,
                    "halting": halting,
                }
 
        return self._decode(final_feat, skip1, skip2, skip3, x_ori)
 
    @torch.no_grad()
    def fuse_entire_model(self):
        modules = list(self.named_modules())
        for name, module in reversed(modules):
            if (
                hasattr(module, "fuse")
                and hasattr(module, "deploy")
                and not module.deploy
            ):
                module.fuse()
        print(">>> Model fused.")
        self.deploy = True
 