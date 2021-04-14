import torch
from plugs import DilatedEncoder


if __name__ == '__main__':
    encoder = DilatedEncoder()
    print(encoder)

    x = torch.rand(1, 2048, 32, 32)
    y = encoder(x)
    print(y.shape)