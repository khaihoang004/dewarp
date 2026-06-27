import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.modules.layers import RMSNorm2d, SwiGLU_FFN, LayerScale
from src.model.modules.attention import DocumentStripAttention, RepAttn
from src.model.modules.repconv import RepConv3


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
            LocalStripBlock(out_channels, deploy=deploy) for _ in range(num_blocks)
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
            LocalStripBlock(out_channels, deploy=deploy) for _ in range(num_blocks)
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


class RepAttnBottleneckLayer(nn.Module):
    """Thay thế LocalStripBlock bằng RepAttn + FFN"""
    def __init__(self, dim, num_heads=8, deploy=False):
        super().__init__()
        self.norm1 = RMSNorm2d(dim)
        self.attn = RepAttn(dim, num_heads=num_heads, deploy=deploy)
        self.norm2 = RMSNorm2d(dim)
        self.ffn = SwiGLU_FFN(dim)

        # Skip-connection xuyên thời gian với x_orig
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, kernel_size=1),
            nn.GELU(),
            RepConv3(dim, dim, deploy=deploy),
            nn.GELU(),
        )

    def forward(self, x, x_orig, step_embed=None):
        # 1. Dung hợp với input gốc của bottleneck
        concat = torch.cat([x, x_orig], dim=1)
        fused = self.fusion(concat)
        
        if step_embed is not None:
            fused = fused + step_embed
        x = x + fused

        # 2. Xử lý qua RepAttn và FFN (Pre-Norm style)
        nx = self.norm1(x)
        x = x + self.attn(nx)
        
        nx2 = self.norm2(x)
        x = x + self.ffn(nx2)
        
        return x


class EntropyLoopedBottleneck(nn.Module):
    """Cơ chế Vòng lặp dừng động dựa trên Shannon Entropy của LoopViT"""
    def __init__(self, dim, num_classes, max_loops=6, num_heads=8, deploy=False):
        super().__init__()
        self.max_loops = max_loops
        self.layer = RepAttnBottleneckLayer(dim, num_heads=num_heads, deploy=deploy)

        # Step embeddings để phân biệt các bước (tùy chọn nhưng khuyến nghị)
        self.step_embeddings = nn.Parameter(torch.zeros(max_loops, 1, dim, 1, 1))
        nn.init.normal_(self.step_embeddings, std=0.02)

        # Khối Auxiliary Head để ánh xạ feature map về phân bố xác suất (Predictive Space)
        # Bắt buộc phải có để đo Entropy chuẩn theo LoopViT
        self.pred_head = nn.Conv2d(dim, num_classes, kernel_size=1)

    def compute_entropy(self, state):
        """
        Tính average pixel-wise Shannon entropy.
        Eq (11) LoopViT: H_t = - 1/N * sum( P * log(P) )
        """
        logits = self.pred_head(state) # (B, num_classes, H, W)
        probs = F.softmax(logits, dim=1)
        log_probs = F.log_softmax(logits, dim=1)
        
        # (B, H, W) -> Trung bình qua các điểm ảnh H, W để ra (B,)
        entropy = -(probs * log_probs).sum(dim=1).mean(dim=[1, 2])
        return entropy

    def forward(self, x, tau=0.05, return_all=False):
        B = x.shape[0]
        device = x.device
        
        # Khi training, luôn giữ computational graph của x_orig
        x_orig = x.clone() if (self.training or return_all) else x.detach()

        state = x
        states = []
        entropies_list = []

        # Các biến quản lý Early Exit cho Inference
        finished_mask = torch.zeros(B, dtype=torch.bool, device=device)
        cached_final = torch.zeros_like(x)
        exit_steps = torch.full((B,), self.max_loops, dtype=torch.long, device=device)

        for t in range(self.max_loops):
            step_embed = self.step_embeddings[t]

            # 1. Cập nhật state qua RepAttn
            state = self.layer(state, x_orig, step_embed)
            
            # 2. Tính Entropy để đo độ "kết tinh" (Crystallization)
            entropy = self.compute_entropy(state)

            if self.training or return_all:
                states.append(state)
                entropies_list.append(entropy)

            # 3. Dynamic Hard-Exit (Chỉ kích hoạt khi Inference)
            if not self.training:
                exit_now = (entropy < tau) & (~finished_mask)

                if exit_now.any():
                    # Lưu lại state cho các sample đã thỏa mãn điều kiện dừng
                    cached_final = torch.where(
                        exit_now.view(B, 1, 1, 1), 
                        state, 
                        cached_final
                    )
                    exit_steps[exit_now] = t + 1
                    finished_mask = finished_mask | exit_now

                state = torch.where(
                    finished_mask.view(B, 1, 1, 1),
                    cached_final,
                    state
                )

                if finished_mask.all():
                    break

        if self.training:
            final_state = state
            
            if return_all:
                return final_state, states, entropies_list
            return final_state

        else:
            final_state = torch.where(
                finished_mask.view(B, 1, 1, 1), 
                cached_final, 
                state
            )
            return final_state, exit_steps