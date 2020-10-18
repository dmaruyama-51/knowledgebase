import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv  = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1: 下から（Decoder） h, w = 28px
        # x2: 横から（Encoder） h, w = 56px

        x1 = self.up(x1) # x1はチャネルは in_channels // 2 になり、 h, w が 2倍に。

        diffY = torch.tensor([x2.size()[2] - x1.size()[2]]) #(batch_size, c, h, w) なので [2] は height
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]]) # [3] は width

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1) # (batch_size, c, h, w) なので channel 方向に結合という意味

        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inconv = DoubleConv(n_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)

        self.up1 = Up(1024, 256)
        self.up2 = Up(512, 128)
        self.up3 = Up(256, 64)
        self.up4 = Up(128, 64)

        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inconv(x) # (3, 224, 224) -> (64, 224, 224)

        x2 = self.down1(x1) # (128, 112, 112)
        x3 = self.down2(x2) # (256, 56, 56)
        x4 = self.down3(x3) # (512, 28, 28)
        x5 = self.down4(x4) # (512, 14, 14)

        x = self.up1(x5, x4) # (256, 28, 28)
        x = self.up2(x, x3) # (128, 56, 56)
        x = self.up3(x, x2) # (64, 112, 112)
        x = self.up4(x, x1) # (32, 224, 224)

        logits = self.outconv(x)

        return logits

