import sys
import torch
import torch.nn.functional as F
from torch import nn

sys.path.append(r"../utils")
from utils import initialize_weights

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv3d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class CARAFE(nn.Module):
    def __init__(self, in_channels, mid_channels=64, scale=2, k_up=5, k_enc=3):
        super(CARAFE, self).__init__()
        self.scale = scale
        self.comp = ConvBNReLU(in_channels, mid_channels, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(mid_channels, (scale * k_up) ** 3, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w, d = X.size()
        h_, w_ , d_ = h * self.scale, w * self.scale, d * self.scale

        W = self.comp(X)  # b * m * h * w * d
        W = self.enc(W)  # b * 100 * h * w * d
        W = self.pix_shf(W)  # b * 25 * h_ * w_ * d_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_ * d_

        X = self.upsmp(X)  # b * c * h_ * w_ * d_
        X = self.unfold(X)  # b * 125c * h_ * w_ * d_
        X = X.view(b, c, -1, h_, w_, d_)  # b * 25 * c * h_ * w_ * d_

        X = torch.einsum('bkhwd,bckhwd->bchwd', W, X)  # b * c * h_ * w_ * d_
        return X

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=(1, 1, 1), downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.BN1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.BN2 = nn.BatchNorm3d(out_channels)
        self.stride = stride
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.BN1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.BN2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class _EncoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, pool_kernal=2, dropout=False):
        super(_EncoderBlock3D, self).__init__()
        if out_channels == 32:
            layers = [
                BasicBlock(in_channels, out_channels)
            ]
        else:
            downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=pool_kernal),
                nn.BatchNorm3d(out_channels * BasicBlock.expansion)
            )
            layers = [
                BasicBlock(in_channels, out_channels, stride=pool_kernal, downsample=downsample),
            ]

        if dropout:
            layers.append(nn.Dropout(p=0.5))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock3D, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv3d(in_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(middle_channels, middle_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(middle_channels, out_channels, kernel_size=2, stride=2, bias=False),
        )

    def forward(self, x):
        return self.decode(x)


class U_net3D_Resnet(nn.Module):
    def __init__(self, num_classes, dropout=True, initial=True):
        super(U_net3D_Resnet, self).__init__()
        self.conv3D = nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False)
        self.enc1 = _EncoderBlock3D(32, 32)
        self.enc2 = _EncoderBlock3D(32, 64)
        self.enc3 = _EncoderBlock3D(64, 128, dropout=dropout)
        self.center = _DecoderBlock3D(128, 256, 128)
        self.dec3 = _DecoderBlock3D(256, 128, 64)
        self.dec2 = _DecoderBlock3D(128, 64, 32)
        self.dec1 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.final = nn.Conv3d(32, num_classes, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid
        self.softmax = nn.Softmax(dim=1)
        if initial:
            initialize_weights(self)

    def forward(self, x):
        con1 = self.conv3D(x)
        enc1 = self.enc1(con1)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)

        center = self.center(enc3)
        dec3 = self.dec3(torch.cat([F.interpolate(center, enc3.size()[2:], mode='trilinear', align_corners=False), enc3], 1))
        dec2 = self.dec2(torch.cat([F.interpolate(dec3, enc2.size()[2:], mode='trilinear', align_corners=False), enc2], 1))
        dec1 = self.dec1(torch.cat([F.interpolate(dec2, enc1.size()[2:], mode='trilinear', align_corners=False), enc1], 1))
        final = self.final(dec1)
        return F.interpolate(final, x.size()[2:], mode='trilinear', align_corners=False)
