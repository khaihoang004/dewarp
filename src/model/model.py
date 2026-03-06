# src/model/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleDewarpNet(nn.Module):
    """
    Model encoder-decoder đơn giản, không skip connection.
    Dùng để test pipeline nhanh, không quan tâm size input có chia hết hay không.
    Input: (B, 3, H, W) → Output: (B, 2, H, W) backward map
    """
    def __init__(self, in_channels=3, out_channels=2, base_filters=32):
        super(SimpleDewarpNet, self).__init__()
        
        # Encoder (downsample)
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 4, base_filters * 8)
        
        # Decoder (upsample + conv)
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 4),
            nn.ReLU(inplace=True),
        )
        self.dec3 = self._conv_block(base_filters * 4, base_filters * 4)
        
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters * 2),
            nn.ReLU(inplace=True),
        )
        self.dec2 = self._conv_block(base_filters * 2, base_filters * 2)
        
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(base_filters),
            nn.ReLU(inplace=True),
        )
        self.dec1 = self._conv_block(base_filters, base_filters)
        
        # Final
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()  # nếu backward map range [-1, 1]

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e3, 2))
        
        # Decoder - upsample + conv (không concat)
        d3 = self.up3(b)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        out = self.tanh(out)  # hoặc sigmoid nếu range [0,1]
        
        # Đảm bảo output size bằng input size (nếu lệch do pooling)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        
        return out


def build_model():
    # Không cần cfg nữa, hardcode đơn giản cho test
    return SimpleDewarpNet(
        in_channels=3,
        out_channels=2,
        base_filters=32   # nhỏ để train nhanh
    )


# Test nhanh
if __name__ == "__main__":
    model = build_model()
    dummy = torch.randn(2, 3, 356, 244)
    out = model(dummy)
    print(out.shape)  # nên là [2, 2, 356, 244]