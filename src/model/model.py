import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=2, base_filters=32):
        super(SimpleUNet, self).__init__()
        
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_filters)
        self.enc2 = self._conv_block(base_filters, base_filters * 2)
        self.enc3 = self._conv_block(base_filters * 2, base_filters * 4)
        self.enc4 = self._conv_block(base_filters * 4, base_filters * 8)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base_filters * 8, base_filters * 16)
        
        # Decoder
        self.up4 = nn.ConvTranspose2d(base_filters * 16, base_filters * 8, kernel_size=2, stride=2)
        self.dec4 = self._conv_block(base_filters * 16, base_filters * 8)  # skip + up
        
        self.up3 = nn.ConvTranspose2d(base_filters * 8, base_filters * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base_filters * 8, base_filters * 4)
        
        self.up2 = nn.ConvTranspose2d(base_filters * 4, base_filters * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base_filters * 4, base_filters * 2)
        
        self.up1 = nn.ConvTranspose2d(base_filters * 2, base_filters, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_filters * 2, base_filters)
        
        # Final conv
        self.final = nn.Conv2d(base_filters, out_channels, kernel_size=1)
        
        self.tanh = nn.Tanh()

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
        e4 = self.enc4(F.max_pool2d(e3, 2))
        
        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        
        # Decoder + skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)
        
        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        
        out = self.final(d1)
        out = self.tanh(out)
        
        return out


def build_model(cfg):

    return SimpleUNet(
        in_channels=cfg.in_channels if hasattr(cfg, 'in_channels') else 3,
        out_channels=cfg.out_channels if hasattr(cfg, 'out_channels') else 2,
        base_filters=32
    )


if __name__ == "__main__":
    model = build_model(type('Config', (), {'in_channels': 3, 'out_channels': 2})())
    dummy_input = torch.randn(2, 3, 384, 384)
    output = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")