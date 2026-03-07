import torch
from torch import nn

class MBRConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, deploy=False):
        super(MBRConv, self).__init__()
        self.deploy = deploy
        if deploy:
            self.reparam_conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        else:
            self.branch_3x3 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
            self.branch_1x1 = nn.Conv2d(in_channels, out_channels, 1, stride, 0)
            self.identity = nn.Identity() if in_channels == out_channels and stride == 1 else None

    def forward(self, x):
        if self.deploy:
            return self.reparam_conv(x)
        
        out = self.branch_3x3(x) + self.branch_1x1(x)
        if self.identity is not None:
            out += self.identity(x)
        return out

# 2. Feature Self-Transform (FST) Module
# Tăng tính phi tuyến bằng cách sử dụng tương tác bậc hai (quadratic interaction)
class FST(nn.Module):
    def __init__(self, channels):
        super(FST, self).__init__()
        self.scale = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Công thức: x * (scale * x + bias)
        return x * (self.scale * x + self.bias)

# 3. Hierarchical Dual-Path Attention (HDPA)
class HDPA(nn.Module):
    def __init__(self, channels):
        super(HDPA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Nhánh global (Channel attention đơn giản)
        self.global_conv = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 2, channels, 1),
            nn.Sigmoid()
        )
        # Nhánh local (Spatial attention đơn giản)
        self.local_conv = nn.Sequential(
            nn.Conv2d(channels, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        g_attn = self.global_conv(self.avg_pool(x) + self.max_pool(x))
        l_attn = self.local_conv(x)
        return x * g_attn * l_attn

class MobileIE(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, mid_channels=16, num_blocks=2, deploy=False):
        super(MobileIE, self).__init__()
        
        # Shallow Feature Extraction
        self.shallow_extract = MBRConv(in_channels, mid_channels, deploy=deploy)
        self.prelu = nn.PReLU(mid_channels)
        
        # Deep Feature Blocks
        self.deep_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            block = nn.Sequential(
                MBRConv(mid_channels, mid_channels, deploy=deploy),
                FST(mid_channels),
                MBRConv(mid_channels, mid_channels, deploy=deploy),
                nn.PReLU(mid_channels)
            )
            self.deep_blocks.append(block)
        
        # Attention
        self.attention = HDPA(mid_channels)
        
        # Reconstruction
        self.final_conv = MBRConv(mid_channels, out_channels, deploy=deploy)

    def forward(self, x):
        identity = x
        
        x = self.prelu(self.shallow_extract(x))
        
        for block in self.deep_blocks:
            x = block(x)
            
        x = self.attention(x)
        x = self.final_conv(x)
        
        return x + identity

model = MobileIE(mid_channels=16)
print(f"Số lượng tham số: {sum(p.numel() for p in model.parameters())}")