import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class ForCenUNet(nn.Module):
    def __init__(self, in_channels=3, base_ch=64):
        super().__init__()

        # =========================
        # Encoder
        # =========================
        self.enc1 = ConvBlock(in_channels, base_ch)
        self.enc2 = ConvBlock(base_ch, base_ch * 2)
        self.enc3 = ConvBlock(base_ch * 2, base_ch * 4)
        self.enc4 = ConvBlock(base_ch * 4, base_ch * 8)

        self.bottleneck = ConvBlock(base_ch * 8, base_ch * 16)

        # =========================
        # Decoder upsample convs
        # =========================
        self.up4 = nn.Conv2d(base_ch * 16, base_ch * 8, 1)
        self.up3 = nn.Conv2d(base_ch * 8, base_ch * 4, 1)
        self.up2 = nn.Conv2d(base_ch * 4, base_ch * 2, 1)
        self.up1 = nn.Conv2d(base_ch * 2, base_ch, 1)

        # =========================
        # Decoder conv blocks
        # =========================
        self.dec4 = ConvBlock(base_ch * 16, base_ch * 8)
        self.dec3 = ConvBlock(base_ch * 8, base_ch * 4)
        self.dec2 = ConvBlock(base_ch * 4, base_ch * 2)
        self.dec1 = ConvBlock(base_ch * 2, base_ch)

        # =========================
        # Backward map head
        # =========================
        self.bm_head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 2, 1),
            nn.Tanh()
        )

        # =========================
        # Foreground mask head
        # =========================
        self.mask_head = nn.Conv2d(base_ch, 2, 1)

    def forward(self, x):

        # =========================
        # Encoder
        # =========================
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        b = self.bottleneck(F.max_pool2d(e4, 2))

        # =========================
        # Decoder
        # =========================

        d4 = self.up4(b)
        d4 = F.interpolate(d4, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = torch.cat([d4, e4], dim=1)
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # =========================
        # Heads
        # =========================
        bm = self.bm_head(d1)
        mask_logits = self.mask_head(d1)

        return bm, mask_logits

if __name__ == "__main__":

    model = ForCenUNet()

    dummy = torch.randn(2, 3, 356, 244)

    bm, mask = model(dummy)

    print("Backward map shape :", bm.shape)
    print("Mask logits shape  :", mask.shape)