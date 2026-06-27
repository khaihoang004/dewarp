import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.modules.layers import RMSNorm2d, SwiGLU_FFN, LayerScale
from src.model.modules.attention import DocumentStripAttention, RepAttn
from src.model.modules.repconv import RepConv3


class RepDATBlock(nn.Module):
    def __init__(self, dim, num_heads=8, deploy=False):
        super().__init__()
        self.norm1 = RMSNorm2d(dim)
        self.strip_attn = DocumentStripAttention(dim)
        self.rep_attn = RepAttn(dim, num_heads=num_heads, deploy=deploy)
        self.scale1 = LayerScale(dim, init_value=0.5)
        
        self.norm2 = RMSNorm2d(dim)
        self.ffn = SwiGLU_FFN(dim)
        self.scale2 = LayerScale(dim, init_value=0.5)

    def forward(self, x):
        nx = self.norm1(x)
        attn_out = self.strip_attn(nx) + self.rep_attn(nx)
        x = x + self.scale1(attn_out)
        
        x = x + self.scale2(self.ffn(self.norm2(x)))
        return x

class LocalStripBlock(nn.Module):
    def __init__(self, dim, deploy=False):
        super().__init__()
        self.norm1 = RMSNorm2d(dim)
        self.strip_attn = DocumentStripAttention(dim)
        self.scale1 = LayerScale(dim, init_value=0.5)
        
        self.norm2 = RMSNorm2d(dim)
        self.ffn = SwiGLU_FFN(dim)
        self.scale2 = LayerScale(dim, init_value=0.5)

    def forward(self, x):
        nx = self.norm1(x)
        x = x + self.scale1(self.strip_attn(nx))
        x = x + self.scale2(self.ffn(self.norm2(x)))
        return x

class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1, num_heads=8, deploy=False):
        super().__init__()
        self.conv = RepConv3(in_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()
        
        self.blocks = nn.ModuleList([
            LocalStripBlock(out_channels, num_heads=num_heads, deploy=deploy) for _ in range(num_blocks)
        ])
        
        self.norm_feat = RMSNorm2d(out_channels)
        self.down = nn.PixelUnshuffle(downscale_factor=2)
        self.channel_compress = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.norm_down = RMSNorm2d(out_channels)

    def forward(self, x):
        feat = self.act(self.conv(x))
        for blk in self.blocks:
            feat = blk(feat)
            
        skip_feat = self.norm_feat(feat)
        downsampled = self.down(skip_feat)
        downsampled = F.gelu(self.channel_compress(downsampled))
        return skip_feat, self.norm_down(downsampled)

class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, num_blocks=1, num_heads=8, deploy=False):
        super().__init__()
        self.expand = nn.Conv2d(in_channels, in_channels * 4, kernel_size=3, padding=1, bias=False)
        self.up = nn.PixelShuffle(upscale_factor=2)
        
        self.gate_conv = nn.Conv2d(in_channels, skip_channels, kernel_size=1)
        self.conv = RepConv3(in_channels + skip_channels, out_channels, deploy=deploy)
        self.blocks = nn.ModuleList([
            LocalStripBlock(out_channels, num_heads=num_heads, deploy=deploy) for _ in range(num_blocks)
        ])
        self.norm_out = RMSNorm2d(out_channels)

    def forward(self, x, skip_feat):
        x = F.gelu(self.up(self.expand(x)))
        gate_mask = torch.sigmoid(self.gate_conv(x))
        gated_skip = skip_feat * gate_mask
        
        x = torch.cat([x, gated_skip], dim=1)
        x = F.gelu(self.conv(x))
        for blk in self.blocks:
            x = blk(x)
        return self.norm_out(x)


class BottleneckLayer(nn.Module):
    def __init__(self, dim, num_blocks=1, num_heads=8, deploy=False):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            RepDATBlock(dim, num_heads=num_heads, deploy=deploy) for _ in range(num_blocks)
        ])

        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            RepConv3(dim, dim, deploy=deploy), 
            nn.GELU(),
        )

        self.exit_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1),
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
        exit_prob = torch.sigmoid(exit_logit)

        return x, exit_logit, exit_prob

class AdaptiveLoopedBottleneck(nn.Module):
    def __init__(self, dim, max_loops=6, num_heads=8, deploy=False):
        super().__init__()
        self.max_loops = max_loops
        
        self.layer = BottleneckLayer(dim, num_heads=num_heads, deploy=deploy)

        self.step_embeddings = nn.Parameter(torch.zeros(max_loops, 1, dim, 1, 1))
        nn.init.normal_(self.step_embeddings, std=0.02)

    def forward(self, x, halt_threshold=0.85, return_all=False):
        B = x.shape[0]
        device = x.device
        if self.training or return_all:
            x_orig = x.clone()
        else:
            x_orig = x.detach()

        if self.training or return_all:
            states = []
            exit_probs_list = []
            logits_list = []

            state = x
            for t in range(self.max_loops):
                step_embed = self.step_embeddings[t]
                state, g_logit, e_prob = self.layer(state, x_orig, step_embed)

                states.append(state)
                exit_probs_list.append(e_prob)
                logits_list.append(g_logit)

            exit_probs = torch.stack(exit_probs_list, dim=0)   # (T, B)
            gate_logits = torch.stack(logits_list, dim=0)      # (T, B)

            survival = torch.ones(B, device=device)
            halting_weights = torch.zeros_like(exit_probs)     # (T, B)

            for t in range(self.max_loops):
                if t < self.max_loops - 1:
                    p_t = exit_probs[t] * survival
                    halting_weights[t] = p_t
                    survival = survival * (1.0 - exit_probs[t]).clamp(min=1e-6)
                else:
                    halting_weights[t] = survival

            final_state = torch.zeros_like(states[0])
            for t in range(self.max_loops):
                w = halting_weights[t].view(B, 1, 1, 1)
                final_state = final_state + w * states[t]

            if return_all:
                return final_state, states, halting_weights, gate_logits, exit_probs
            return final_state

        else:
            state = x
            cumulative = torch.zeros(B, device=device)

            for t in range(self.max_loops):
                step_embed = self.step_embeddings[t]

                state, _, lambda_t = self.layer(state, x_orig, step_embed)

                survival = (1.0 - cumulative).clamp(min=1e-6)
                cumulative = torch.clamp(cumulative + lambda_t * survival, min=0.0, max=1.0)

                if (cumulative >= halt_threshold).all():
                    break

            return state