from turtle import forward
from torch import nn
import torch.nn.functional as F


class SrNet(nn.Module):
    def __init__(self):
        super(SrNet, self).__init__()

        self.betas_0 = []
        self.betas_1 = []
        self.betas_2 = []
        self.betas_3 = []

        self.coding_0 = []
        self.coding_1 = []
        self.coding_2 = []
        self.coding_3 = []

        self.clipModule = nn.Sequential(
            nn.Conv2d(4, 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(3),
            nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=0),
        )

    def forward(self, x):
        x = self.clipModule(x)
        print(x)
        return x