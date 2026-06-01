import torch
import torch.nn as nn
import torch.nn.functional as F

from .repconv import RepConv3, RepConv7
from .attention import DocumentAttn, RestormerAttention


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(dim=1, keepdim=True) + self.eps)
        return norm * self.weight.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

# --- 3. Tầng SwiGLU Feed-Forward Network (Tối Ưu Tham Số) ---
class SwiGLU_FFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Giảm hệ số ẩn xuống 8/3 (~2.67) để bảo toàn lượng tham số của tầng FFN thông thường
        hidden_dim = int(dim * 8 / 3)
        
        self.w_v = nn.Conv2d(dim, hidden_dim * 2, kernel_size=1)
        self.project_out = nn.Conv2d(hidden_dim, dim, kernel_size=1)

    def forward(self, x):
        # Chia đôi tensor đầu ra thành 2 nhánh W và V
        w, v = torch.chunk(self.w_v(x), 2, dim=1)
        # Công thức: Swish(w) * v
        return self.project_out(F.silu(w) * v)

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=0.1):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim) * init_value)

    def forward(self, x):
        return x * self.scale.view(1, -1, 1, 1)

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
            self.conv_1.fuse()
            self.conv_2.fuse()
            
            if hasattr(self.conv_1, 'deploy'): self.conv_1.deploy = True
            if hasattr(self.conv_2, 'deploy'): self.conv_2.deploy = True
            
            self.deploy = True

    def forward(self, x):
        out = self.conv_1(x)
        out = self.act_1(out)
        out = self.conv_2(out)
        out = self.act_2(out)
        return out + x

class BottleneckLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = RestormerAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU_FFN(dim)
        
        self.scale1 = LayerScale(dim, init_value=0.1)  # sau Attention
        self.scale2 = LayerScale(dim, init_value=0.1)  # sau FFN

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1)
        )

    def forward(self, x):
        attn_out = self.attn(self.norm1(x))
        x = x + self.scale1(attn_out)
        
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.scale2(ffn_out)
        
        gate_logits = self.gate(x)
        exit_prob = torch.sigmoid(gate_logits).view(-1)

        return x, gate_logits, exit_prob


class AdaptiveLoopedBottleneck(nn.Module):
    def __init__(self, dim, max_loops=6):
        super().__init__()
        self.max_loops = max_loops
        self.layer = BottleneckLayer(dim)

    def forward(self, x, halt_threshold=0.8, return_all=False):
        B = x.shape[0]
        device = x.device

        if self.training or return_all:
            states = []
            exit_probs = []          # list of (B,)
            gate_logits_list = []

            state = x

            for _ in range(self.max_loops):
                state, g_logits, lambda_t = self.layer(state)
                states.append(state)
                exit_probs.append(lambda_t)
                gate_logits_list.append(g_logits)
            
            exit_probs = torch.stack(exit_probs, dim=0)        # (T, B)
            gate_logits = torch.stack(gate_logits_list, dim=0)

            survival = torch.ones(B, device=device)            # S_0 = 1
            halting_weights = torch.zeros_like(exit_probs)

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
                final_state += w * states[t]

            return final_state, states, halting_weights, gate_logits, exit_probs

        else:
            # Inference
            state = x
            cumulative = torch.zeros(B, device=device)

            for t in range(self.max_loops):
                state, _, lambda_t = self.layer(state)
                survival = (1.0 - cumulative).clamp(min=1e-6)
                cumulative = torch.clamp(cumulative + lambda_t * survival, min=0.0, max=1.0)
                
                if (cumulative >= halt_threshold).all():
                    return state

            return state


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

        self.down = nn.PixelUnshuffle(downscale_factor=2)
        self.channel_compress = nn.Conv2d(out_channels * 4, out_channels, kernel_size=1, bias=False)
        self.act_down = nn.GELU()

        self.stacked_layers = nn.ModuleList([
            ResidualRepConv(out_channels, out_channels, groups=1, deploy=deploy)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        feat = self.act(self.conv(x))
        downsampled = self.down(feat)
        downsampled = self.act_down(self.channel_compress(downsampled))

        for layer in self.stacked_layers:
            downsampled = layer(downsampled)
            
        return feat, downsampled


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
            
        return x


class LoopRepDocEnhanceNet(nn.Module):
    def __init__(self, base_dim=32, max_loops=4, enc_blocks=[1, 1, 2], dec_blocks=[1, 1, 2], deploy=False):
        super().__init__()
        self.base_dim = base_dim
        
        if isinstance(enc_blocks, int):
            enc_blocks = [enc_blocks] * 3
        if isinstance(dec_blocks, int):
            dec_blocks = [dec_blocks] * 3
            
        self.shallow_extractor = ShallowExtractor(3, base_dim, deploy=deploy)
        
        self.enc1 = EncoderStage(base_dim, base_dim, num_blocks=enc_blocks[0], deploy=deploy)
        self.enc2 = EncoderStage(base_dim, base_dim * 2, num_blocks=enc_blocks[1], deploy=deploy)
        self.enc3 = EncoderStage(base_dim * 2, base_dim * 4, num_blocks=enc_blocks[2], deploy=deploy)
        
        self.bottleneck = AdaptiveLoopedBottleneck(dim=base_dim * 4, max_loops=max_loops)
        
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
        output = x_ori + residual
        return output

    def forward(self, x, halt_threshold=0.80, return_all=False):
        x_ori = x 
        
        s0 = self.shallow_extractor(x)
        skip1, e1 = self.enc1(s0)
        skip2, e2 = self.enc2(e1)
        skip3, e3 = self.enc3(e2)
        
        if self.training or return_all:
            b_out, layer_outputs, halting_weights, gate_logits, exit_probs = self.bottleneck(e3, halt_threshold=halt_threshold, return_all=return_all)
        else:
            b_out = self.bottleneck(e3, halt_threshold=halt_threshold)

        output = self._decode(b_out, skip1, skip2, skip3, x_ori)
        
        if not self.training:
            output = torch.clamp(output, 0.0, 1.0)

        if return_all:
            intermediate_preds = []
            for state in layer_outputs:
                pred_t = self._decode(state, skip1, skip2, skip3, x_ori)
                if not self.training:
                    pred_t = torch.clamp(pred_t, 0.0, 1.0)
                intermediate_preds.append(pred_t)

            return output, intermediate_preds, halting_weights, gate_logits, exit_probs

        return output

    @torch.no_grad()
    def fuse_entire_model(self):
        fused = []
        for name, module in self.named_modules():
            if (
                hasattr(module, 'deploy')
                and hasattr(module, 'fuse')
                and not module.deploy
                and name != ''
            ):
                module.fuse()
                fused.append(name)
 
        print(f">>> ĐÃ FUSE {len(fused)} MODULE: {fused}")