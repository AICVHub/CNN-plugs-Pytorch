"""
The define of Dilated Encoder from YOLOF: https://github.com/megvii-model/YOLOF:
No detectron2, only Pytorch.

"""

import torch
import torch.nn as nn


class DilatedEncoder(nn.Module):
    """
    Dilated Encoder for YOLOF.

    This module contains two types of components:
        - the original FPN lateral convolution layer and fpn convolution layer,
          which are 1x1 conv + 3x3 conv
        - the dilated residual block
    """

    def __init__(self,
                 in_channels=2048,
                 encoder_channels=512,
                 block_mid_channels=128,
                 num_residual_blocks=4,
                 block_dilations=[2, 4, 6, 8]
                 ):
        super(DilatedEncoder, self).__init__()
        # fmt: off
        self.in_channels = in_channels
        self.encoder_channels = encoder_channels
        self.block_mid_channels = block_mid_channels
        self.num_residual_blocks = num_residual_blocks
        self.block_dilations = block_dilations

        assert len(self.block_dilations) == self.num_residual_blocks

        # init
        self._init_layers()
        self._init_weight()

    def _init_layers(self):
        self.lateral_conv = nn.Conv2d(self.in_channels,
                                      self.encoder_channels,
                                      kernel_size=1)
        self.lateral_norm = nn.BatchNorm2d(self.encoder_channels)
        self.fpn_conv = nn.Conv2d(self.encoder_channels,
                                  self.encoder_channels,
                                  kernel_size=3,
                                  padding=1)
        self.fpn_norm = nn.BatchNorm2d(self.encoder_channels)
        encoder_blocks = []
        for i in range(self.num_residual_blocks):
            dilation = self.block_dilations[i]
            encoder_blocks.append(
                Bottleneck(
                    self.encoder_channels,
                    self.block_mid_channels,
                    dilation=dilation
                )
            )
        self.dilated_encoder_blocks = nn.Sequential(*encoder_blocks)

    def xavier_init(self, layer):
        if isinstance(layer, nn.Conv2d):
            # print(layer.weight.data.type())
            # m.weight.data.fill_(1.0)
            nn.init.xavier_uniform_(layer.weight, gain=1)

    def _init_weight(self):
        self.xavier_init(self.lateral_conv)
        self.xavier_init(self.fpn_conv)
        for m in [self.lateral_norm, self.fpn_norm]:
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        for m in self.dilated_encoder_blocks.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0, std=0.01)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            if isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.SyncBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, feature: torch.Tensor) -> torch.Tensor:
        out = self.lateral_norm(self.lateral_conv(feature))
        out = self.fpn_norm(self.fpn_conv(out))
        return self.dilated_encoder_blocks(out)


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels: int = 512,
                 mid_channels: int = 128,
                 dilation: int = 1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels,
                      kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out + identity
        return out


if __name__ == '__main__':
    encoder = DilatedEncoder()
    print(encoder)

    x = torch.rand(1, 2048, 32, 32)
    y = encoder(x)
    print(y.shape)
