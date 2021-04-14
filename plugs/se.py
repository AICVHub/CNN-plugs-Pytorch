import torch
from torch import nn


class SE_Block(nn.Module):
    def __init__(self, ch_in, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 全局自适应池化
        self.fc = nn.Sequential(
            nn.Linear(ch_in, ch_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(ch_in // reduction, ch_in, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # squeeze操作
        y = self.fc(y).view(b, c, 1, 1) # FC获取通道注意力权重，是具有全局信息的
        return x * y.expand_as(x) # 注意力作用每一个通道上


if __name__ == '__main__':
    import time
    ch_in = 1024
    m = SE_Block(ch_in=ch_in).cuda()
    print(m)

    x = torch.rand(1, ch_in, 64, 64).cuda()
    for i in range(10):
        t0 = time.time()
        y = m(x)
        print("time:{}, shape:{}".format(time.time()-t0, y.shape))