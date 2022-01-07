import torch
from torch import nn
import numpy
import torch.nn.functional as F

class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out):
        super(ReLUConvBN, self).__init__()
        kernel_size = 3
        padding = 1
        self.scale = 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )

        self.scale = C_in/C_out
        self._initialize_weights()



    def forward(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = F.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class ReLUConv5BN(nn.Module):

    def __init__(self, C_in, C_out):
        super(ReLUConv5BN, self).__init__()
        kernel_size = 5
        padding = 2
        self.scale = 1
        self.op = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(C_out)
        )

        self.scale = C_in/C_out
        self._initialize_weights()



    def forward(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = F.interpolate(x, [feature_size_h, feature_size_w], mode='bilinear', align_corners=True)
        return self.op(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                if m.weight is not None:
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))