import sys
import torch
import torch.nn.functional as F
from torch import nn

sys.path.append(r"../utils")
from utils import initialize_weights
class ExNet(nn.Module):
    def __init__(self, num_classes=1):
        super(ExNet, self).__init__()
        self.CBR1 = nn.Sequential(
            nn.Conv3d(2, num_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        self.CBR2 = nn.Sequential(
            nn.Conv3d(16, num_classes, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(1),
            nn.ReLU(inplace=True)
        )
        self.sigmoid = nn.Sigmoid()
        initialize_weights(self)

    def forward(self, image, label256):
        x = self.CBR1(torch.cat([image, label256], 1))
        # x = self.CBR2(x)
        x = self.sigmoid(x)
        return x

