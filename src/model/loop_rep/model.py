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


class BottleneckLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = RestormerAttention(dim)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU_FFN(dim)
        
        # Gate chạy song song với mỗi layer để tính Exit Probability
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, 1)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        
        # Tính xác suất exit tại bước này: dạng tensor (B, 1, 1, 1) -> view về (B,)
        gate_logits = self.gate(x).view(-1)
        exit_prob = torch.sigmoid(gate_logits)
        return x, gate_logits, exit_prob


# --- KIẾN TRÚC LOOPED BOTTLENECK CHÍNH ---
class AdaptiveLoopedBottleneck(nn.Module):

    def __init__(self, dim, max_loops=6):
        super().__init__()

        self.max_loops = max_loops

        self.layer = BottleneckLayer(dim)

    def forward(self, x, halt_threshold=0.8):

        B = x.shape[0]

        # =================================================
        # TRAINING
        # =================================================

        if self.training:

            state = x

            layer_outputs = []

            halt_logits = []

            halt_probs = []

            for _ in range(self.max_loops):

                state, logits_t, p_t = self.layer(state)

                layer_outputs.append(state)

                halt_logits.append(logits_t)

                halt_probs.append(p_t)
            
            halt_logits = torch.stack(halt_logits, dim=0)

            # ---------------------------------------------
            # ACT weights
            # ---------------------------------------------

            halting_weights = []

            survival_prob = torch.ones(
                B,
                device=x.device
            )

            for t in range(self.max_loops):

                p_t = halt_probs[t]

                if t < self.max_loops - 1:

                    w_t = survival_prob * p_t

                    survival_prob = survival_prob * (
                                    1.0 - p_t
                                ).clamp(min=1e-6)

                else:
                    w_t = survival_prob

                halting_weights.append(w_t)

            halting_weights = torch.stack(
                halting_weights,
                dim=0
            )

            halting_weights = halting_weights / (
                halting_weights.sum(
                    dim=0,
                    keepdim=True
                ) + 1e-8
            )

            # ---------------------------------------------
            # weighted state
            # ---------------------------------------------

            final_state = torch.zeros_like(layer_outputs[0])

            for t in range(self.max_loops):

                w_t = halting_weights[t].view(
                    B,
                    1,
                    1,
                    1
                )

                final_state += (
                    w_t * layer_outputs[t]
                )

            return (
                final_state,
                layer_outputs,
                halting_weights,
                halt_logits
            )

        # =================================================
        # INFERENCE
        # =================================================

        else:

            state = x

            cumulative_halt = torch.zeros(
                B,
                device=x.device
            )

            for t in range(self.max_loops):

                state, logits_t, p_t = self.layer(state)

                cumulative_halt = torch.clamp(
                    cumulative_halt + p_t,
                    max=1.0
                )

                if (cumulative_halt >= halt_threshold).all():

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
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.conv = RepConv3(in_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        feat = self.act(self.conv(x))
        downsampled = self.down(feat)
        return feat, downsampled


class DecoderStage(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels, deploy=False):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv = RepConv3(in_channels + skip_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()

    def forward(self, x, skip_feat):
        x = self.up(x)
        x = torch.cat([x, skip_feat], dim=1)
        return self.act(self.conv(x))


class LoopRepDocEnhanceNet(nn.Module):
    def __init__(self, base_dim=32, max_loops=4, deploy=False):
        super().__init__()
        self.base_dim = base_dim
        
        # Shallow Extraction (H, W) -> base_dim (Kênh 32)
        self.shallow_extractor = ShallowExtractor(3, base_dim, deploy=deploy)
        
        # Encoder Blocks (Hạ không gian từ 1/1 xuống 1/8)
        self.enc1 = EncoderStage(base_dim, base_dim, deploy=deploy)         # 1/1 -> 1/2
        self.enc2 = EncoderStage(base_dim, base_dim * 2, deploy=deploy)     # 1/2 -> 1/4
        self.enc3 = EncoderStage(base_dim * 2, base_dim * 4, deploy=deploy) # 1/4 -> 1/8
        
        # Bottleneck thích ứng lõi động (Chạy ở scale 1/8)
        self.bottleneck = AdaptiveLoopedBottleneck(dim=base_dim * 4, max_loops=max_loops)
        
        # Decoder Blocks (Khôi phục không gian từ 1/8 lên lại 1/1)
        self.dec3 = DecoderStage(base_dim * 4, base_dim * 4, base_dim * 2, deploy=deploy) # 1/8 -> 1/4
        self.dec2 = DecoderStage(base_dim * 2, base_dim * 2, base_dim, deploy=deploy)     # 1/4 -> 1/2
        self.dec1 = DecoderStage(base_dim, base_dim, base_dim, deploy=deploy)             # 1/2 -> 1/1
        
        # Output Head: Dự đoán phần dư toàn cục (Global Residual Mapping)
        self.final_head = nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)


    def _decode(
        self,
        bottleneck_feat,
        skip1,
        skip2,
        skip3,
        x_ori
    ):

        d3 = self.dec3(bottleneck_feat, skip3)

        d2 = self.dec2(d3, skip2)

        d1 = self.dec1(d2, skip1)

        residual = self.final_head(d1)

        output = x_ori + residual

        return output

    def forward(self, x, return_all=False):
        x_ori = x # Giữ lại ảnh gốc đầu vào
        
        # --- ENCODER PATH ---
        s0 = self.shallow_extractor(x)
        skip1, e1 = self.enc1(s0)
        skip2, e2 = self.enc2(e1)
        skip3, e3 = self.enc3(e2)
        
        # --- ADAPTIVE BOTTLENECK ---
        if self.training:
            (
                b_out,
                layer_outputs,
                halting_weights,
                halt_logits
            ) = self.bottleneck(e3)

        else:
            b_out = self.bottleneck(e3)
            
        # --- DECODER PATH ---
        output = self._decode(
            b_out,
            skip1,
            skip2,
            skip3,
            x_ori
        )
        if not self.training:
            output = torch.clamp(output, 0.0, 1.0)
            return output

        intermediate_preds = None

        if return_all:
            intermediate_preds = []

            for state in layer_outputs:
                pred_t = self._decode(
                    state,
                    skip1,
                    skip2,
                    skip3,
                    x_ori
                )

                if not self.training:
                    pred_t = torch.clamp(pred_t, 0.0, 1.0)

                intermediate_preds.append(pred_t)

        return (
            output,
            intermediate_preds,
            halting_weights,
            halt_logits
        )

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
