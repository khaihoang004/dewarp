import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.modules.repconv import RepConv3, RepConv7
from src.model.deshadow.blocks import EncoderStage, DecoderStage, EntropyLoopedBottleneck


class ShallowExtractor(nn.Module):
    def __init__(self, in_channels, out_channels, deploy=False):
        super().__init__()
        self.conv = RepConv7(in_channels, out_channels, deploy=deploy)
        self.act = nn.GELU()

    def forward(self, x):
        return self.act(self.conv(x))


class TextureRecoveryModule(nn.Module):
    def __init__(self, dim, deploy=False):
        super().__init__()
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // 4, dim, kernel_size=1),
            nn.Sigmoid()
        )
        
        self.mask_conv = nn.Sequential(
            RepConv3(dim, dim, deploy=deploy),
            nn.Sigmoid()
        )
        
        # 3. Cascaded Dilated Convolutions
        self.d1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, dilation=1)
        self.d2 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3)
        self.d3 = nn.Conv2d(dim, dim, kernel_size=3, padding=5, dilation=5)
        
        # 4. Projection
        self.proj = RepConv3(dim, dim, deploy=deploy)

    def forward(self, x):
        ca_out = x * self.ca(x)
        mask = self.mask_conv(ca_out)
        
        x_masked = x * mask
        
        d1 = self.d1(x_masked)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        
        out = x + d1 + d2 + d3
        
        return self.proj(out)

class DocDeshadowNet(nn.Module):
    def __init__(self, base_dim=16, max_loops=4, num_heads=8, enc_blocks=[1, 1, 2], dec_blocks=[1, 1, 2], deploy=False):
        super().__init__()
        self.base_dim = base_dim
        
        # Shallow extractors (Multi-scale input)
        self.shallow_L1 = ShallowExtractor(3, base_dim, deploy=deploy)
        self.shallow_L2 = ShallowExtractor(3, base_dim, deploy=deploy)
        self.shallow_L3 = ShallowExtractor(3, base_dim * 2, deploy=deploy)

        self.enc1 = EncoderStage(base_dim, base_dim, enc_blocks[0], num_heads, deploy)
        self.enc2 = EncoderStage(base_dim * 2, base_dim * 2, enc_blocks[1], num_heads, deploy)
        self.enc3 = EncoderStage(base_dim * 4, base_dim * 4, enc_blocks[2], num_heads, deploy)

        self.bottleneck = EntropyLoopedBottleneck(
            dim=base_dim * 4, 
            num_classes=2, 
            max_loops=max_loops, 
            num_heads=num_heads, 
            deploy=deploy
        )

        self.dec3 = DecoderStage(base_dim * 4, base_dim * 4, base_dim * 2, dec_blocks[2], num_heads, deploy)
        self.dec2 = DecoderStage(base_dim * 2, base_dim * 2, base_dim, dec_blocks[1], num_heads, deploy)
        self.dec1 = DecoderStage(base_dim, base_dim, base_dim, dec_blocks[0], num_heads, deploy)

        self.final_head = nn.Sequential(
            TextureRecoveryModule(base_dim, deploy=deploy),
            RepConv3(base_dim, base_dim, deploy=deploy),
            nn.GELU(),
            nn.Conv2d(base_dim, 3, kernel_size=3, padding=1)
        )

    def forward(self, x, tau=0.05, return_all=False):
        x_ori = x.clone()
        
        # 1. Multi-scale Inputs
        x_L2 = F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)
        x_L3 = F.interpolate(x, scale_factor=0.25, mode='bilinear', align_corners=False)

        # 2. Extract & Fuse
        feat_L1 = self.shallow_L1(x)
        skip1, down1 = self.enc1(feat_L1)
        
        feat_L2 = self.shallow_L2(x_L2)
        skip2, down2 = self.enc2(torch.cat([down1, feat_L2], dim=1))
        
        feat_L3 = self.shallow_L3(x_L3)
        skip3, down3 = self.enc3(torch.cat([down2, feat_L3], dim=1))

        # 3. Looped Bottleneck (LoopViT Strategy)
        if self.training or return_all:
            b_out, layer_outputs, entropies = self.bottleneck(down3, tau=tau, return_all=True)
        else:
            b_out, exit_steps = self.bottleneck(down3, tau=tau)

        # 4. Decode
        d3 = self.dec3(b_out, skip3)
        d2 = self.dec2(d3, skip2)
        d1 = self.dec1(d2, skip1)
        
        # 5. Final Output (Texture Recovery & Residual)
        residual = self.final_head(d1)
        output = x_ori + residual

        if not self.training:
            output = torch.clamp(output, 0.0, 1.0)

        # 6. Trả về thông tin phụ (dùng cho huấn luyện, debug)
        if return_all:
            intermediate_preds = []
            bottleneck_logits_list = []
            
            for state in layer_outputs:
                logits = self.bottleneck.pred_head(state)
                bottleneck_logits_list.append(logits)
                
                d3_t = self.dec3(state, skip3)
                d2_t = self.dec2(d3_t, skip2)
                d1_t = self.dec1(d2_t, skip1)
                res_t = self.final_head(d1_t)
                
                pred_t = x_ori + res_t
                intermediate_preds.append(pred_t)
                
            return output, intermediate_preds, bottleneck_logits_list

        return output, exit_steps

    @torch.no_grad()
    def fuse_entire_model(self):
        fused = []
        for name, module in self.named_modules():
            if hasattr(module, 'deploy') and hasattr(module, 'fuse') and not module.deploy and name != '':
                module.fuse()
                fused.append(name)
        print(f">>> ĐÃ FUSE {len(fused)} MODULE")