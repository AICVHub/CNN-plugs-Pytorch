"""
from https://github.com/ruinmessi/RFBNet/blob/master/models/RFB_Net_mobile.py.
"""

import torch
import torch.nn as nn


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True,
                 bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // 8
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv((inter_planes // 2) * 3, (inter_planes // 2) * 3, kernel_size=(3, 1), stride=stride,
                      padding=(1, 0)),
            BasicSepConv((inter_planes // 2) * 3, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1),
            BasicConv((inter_planes // 2) * 3, (inter_planes // 2) * 3, kernel_size=3, stride=stride, padding=1),
            BasicSepConv((inter_planes // 2) * 3, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
        )

        self.ConvLinear = BasicConv(3 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        if in_planes == out_planes:
            self.identity = True
        else:
            self.identity = False
            self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x1, x2), 1)
        out = self.ConvLinear(out)
        if self.identity:
            out = out * self.scale + x
        else:
            short = self.shortcut(x)
            out = out * self.scale + short
        out = self.relu(out)

        return out


class RFBblock(nn.Module):
    def __init__(self, in_ch, residual=False):
        super(RFBblock, self).__init__()
        inter_c = in_ch // 4
        self.branch_0 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
        )
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1)
        )
        self.branch_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=2, padding=2)
        )
        self.branch_3 = nn.Sequential(
            nn.Conv2d(in_channels=in_ch, out_channels=inter_c, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=5, stride=1, padding=2),
            nn.Conv2d(in_channels=inter_c, out_channels=inter_c, kernel_size=3, stride=1, dilation=3, padding=3)
        )
        self.residual = residual

    def forward(self, x):
        x_0 = self.branch_0(x)
        x_1 = self.branch_1(x)
        x_2 = self.branch_2(x)
        x_3 = self.branch_3(x)
        out = torch.cat((x_0, x_1, x_2, x_3), 1)
        if self.residual:
            out += x
        return out


if __name__ == '__main__':
    in_planes, out_planes = 16, 16
    # rfb = BasicRFB(in_planes=in_planes, out_planes=out_planes)
    rfb = RFBblock(in_ch=in_planes, residual=True)
    x = torch.rand(1, in_planes, 8, 8)
    y = rfb(x)
    print(y.shape)
    print(x[0, 0, :, :])
    print(y[0, 0, :, :])
