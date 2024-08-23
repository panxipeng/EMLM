import copy

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepLabV3Plus3D(nn.Module):
    def __init__(self, num_classes, in_channels=1, output_stride=16):
        super(DeepLabV3Plus3D, self).__init__()
        channels = [256, 512, 1024, 2048]
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(7, 7, 7), padding=(3, 3, 3))
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d((2, 2, 2))

        self.layer1 = ResBlock3(64, channels[0], 0, head=True)
        self.layer2 = ResBlock4(channels[0], channels[1], 0)
        self.layer3 = ResBlock6(channels[1], channels[2], 0)
        self.layer4 = ResBlock3(channels[2], channels[3], 0)

        self.conv2 = nn.Conv3d(channels[3], channels[3], kernel_size=(1, 1, 1), stride=(1, 1, 1))

        self.aspp = ASPP3D(channels[-1], output_stride)

        self.decoder = Decoder3D(channels, num_classes)

    def forward(self, x):
        x_size = x.size()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x = self.aspp(x4)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.decoder([x1, x2, x3, x])

        return x


class ASPP3D(nn.Module):
    def __init__(self, in_channels, out_channels=256, output_stride=16, dilation_rates=[1, 6, 12, 18]):
        super(ASPP3D, self).__init__()

        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        self.relu2 = nn.ModuleList()

        for dilation_rate in dilation_rates:
            self.conv2.append(nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=dilation_rate,
                                        dilation=dilation_rate, bias=False))
            self.bn2.append(nn.BatchNorm3d(out_channels))
            self.relu2.append(nn.ReLU(inplace=True))

        self.conv3 = nn.Conv3d(out_channels * (len(dilation_rates) + 1), out_channels, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)

        x2 = []
        for conv, bn, relu in zip(self.conv2, self.bn2, self.relu2):
            x2.append(relu(bn(conv(x))))

        x2.append(x1)

        x3 = torch.cat(x2, dim=1)
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)

        return x3


class Decoder3D(nn.Module):
    def __init__(self, channels, n_class):
        super(Decoder3D, self).__init__()
        self.channels = channels
        self.n_class = n_class
        self.up1 = UpBlock(
            self.channels[-1], self.channels[-2], self.channels[-2], dropout_p=0.0)
        self.up2 = UpBlock(
            self.channels[-2], self.channels[-3], self.channels[-3], dropout_p=0.0)
        self.up3 = UpBlock(
            self.channels[-4], self.channels[-4], self.channels[-4], dropout_p=0.0)
        self.out_conv = nn.Conv3d(self.channels[0], self.n_class,
                                  kernel_size=(3, 3, 3), padding=1)

    def forward(self, feature):
        x0 = feature[0]
        x1 = feature[1]
        x2 = feature[2]
        x3 = feature[3]

        x = self.up1(x3, x2)
        x = self.up2(x2, x1)
        x = self.up3(x1, x0)
        output = self.out_conv(x)
        return output


class UpBlock(nn.Module):
    """Upssampling followed by ConvBlock"""

    def __init__(self, in_channels1, in_channels2, out_channels, dropout_p,
                 bilinear=True):
        super(UpBlock, self).__init__()
        self.bilinear = bilinear
        if bilinear:
            self.conv1x1 = nn.Conv3d(in_channels1, in_channels2, kernel_size=(1, 1, 1))
            self.up = nn.Upsample(
                scale_factor=2, mode='trilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(
                in_channels1, in_channels2, kernel_size=(2, 2, 2), stride=(2, 2, 2))
        self.conv = ConvBlock3D(in_channels2 * 2, out_channels, dropout_p)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = self.conv1x1(x1)
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p=0):
        super(ConvBlock3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1))
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return x


#
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p, id_conv=False, head=False):
        super(ResBlock, self).__init__()
        self.id_conv = id_conv
        if head:
            self.conv_conv = nn.Sequential(
                ConvBlock3D(in_channels, in_channels, 0),
                ConvBlock3D(in_channels, in_channels, 0),
                nn.Sequential(
                    nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
                    nn.BatchNorm3d(out_channels)
                )
            )
        else:
            self.conv_conv = nn.Sequential(
                ConvBlock3D(in_channels, int(out_channels/4), 0),
                ConvBlock3D(int(out_channels/4), int(out_channels/4), 0),
                nn.Sequential(
                    nn.Conv3d(int(out_channels/4), out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
                    nn.BatchNorm3d(out_channels)
                ),
            )
        self.relu = nn.ReLU(inplace=True)
        if id_conv:
            self.resconnect = ID_Block(in_channels, out_channels)

    def forward(self, x):
        x1 = self.conv_conv(x)
        if self.id_conv:
            return self.relu(x1 + self.resconnect(x))
        else:
            return self.relu(x1 + x)


class ID_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ID_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        return self.conv(x)


class ResBlock3(nn.Module):
    def __init__(self, in_channels1, out_channels, dropout_p, head=False):
        super(ResBlock3, self).__init__()
        self.conv_conv = nn.Sequential(
            ResBlock(in_channels1, out_channels, dropout_p, id_conv=True, head=head),
            ResBlock(out_channels, out_channels, dropout_p),
            ResBlock(out_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        x2 = self.conv_conv(x)
        return x2


class ResBlock4(nn.Module):
    def __init__(self, in_channels1, out_channels, dropout_p):
        super(ResBlock4, self).__init__()
        self.conv_conv = nn.Sequential(
            ResBlock(in_channels1, out_channels, dropout_p, id_conv=True),
            ResBlock(out_channels, out_channels, dropout_p),
            ResBlock(out_channels, out_channels, dropout_p),
            ResBlock(out_channels, out_channels, dropout_p),
        )

    def forward(self, x):
        x2 = self.conv_conv(x)
        return x2


class ResBlock6(nn.Module):
    def __init__(self, in_channels1, out_channels, dropout_p):
        super(ResBlock6, self).__init__()
        self.conv_conv = nn.Sequential(
            ResBlock(in_channels1, out_channels, dropout_p, id_conv=True),
            ResBlock(out_channels, out_channels, dropout_p),
            ResBlock(out_channels, out_channels, dropout_p),
            ResBlock(out_channels, out_channels, dropout_p),
            ResBlock(out_channels, out_channels, dropout_p),
            ResBlock(out_channels, out_channels, dropout_p),
        )
        self.id_block = nn.Sequential(
            nn.Conv3d(in_channels1, out_channels, kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.BatchNorm3d(out_channels)
        )

    def forward(self, x):
        x2 = self.conv_conv(x)
        return x2
