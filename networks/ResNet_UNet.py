import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50


class DoubleConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class UpConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpConv3D, self).__init__()

        self.upconv = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.upconv(x1)

        # match tensor shape of x2
        diffZ = x2.size()[2] - x1.size()[2]
        diffY = x2.size()[3] - x1.size()[3]
        diffX = x2.size()[4] - x1.size()[4]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2, diffZ // 2, diffZ - diffZ // 2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class ResNet50UNet3D(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super(ResNet50UNet3D, self).__init__()

        self.encoder = resnet50(pretrained=False)

        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu1 = nn.ReLU(inplace=True)

        self.layer1 = nn.Sequential(self.encoder.conv1, self.encoder.bn1, self.encoder.relu, self.encoder.maxpool,
                                    self.encoder.layer1)
        self.layer2 = nn.Sequential(self.encoder.layer2)
        self.layer3 = nn.Sequential(self.encoder.layer3)
        self.layer4 = nn.Sequential(self.encoder.layer4)

        self.up1 = UpConv3D(2048, 1024)
        self.up2 = UpConv3D(1024, 512)
        self.up3 = UpConv3D(512, 256)
        self.up4 = UpConv3D(256, 128)

        self.conv_out = nn.Conv3d(128, num_classes, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x5 = self.up1(x4, x3)
        x6 = self.up2(x5, x2)
        x7 = self.up3(x6, x1)
        x8 = self.up4(x7, x)
        return x8
