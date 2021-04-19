# CNN-plugs-Pytorch
> Plug and play! Some  CNN modules Implemented by Pytorch.
>



### List of available plugs:

- [x] DilatedEncoder
- [x] NonLocal
- [x] SE_Block
- [x] CABM
- [x] BlurPool
- [x] ASSP
- [x] RFB
- [x] ASFF



### Usage：

```Python
from plugs import*
# then, use the plugs that you need.
```

#### 1. DilatedEncoder

```python
encoder = DilatedEncoder()
print(encoder)

x = torch.rand(1, 2048, 32, 32)
y = encoder(x)
print(y.shape)
```

#### 2. NonLocal

```Python
in_channels = 1024
m = NonLocalBlockND(in_channels=in_channels, dimension=2).cuda()
print(m)

x = torch.rand(1, in_channels, 64, 64).cuda()
y = m(x)
print(y.shape)
```

#### 3. SE

```Python
ch_in = 1024
m = SE_Block(ch_in=ch_in).cuda()
print(m)

x = torch.rand(1, ch_in, 64, 64).cuda()
y = m(x)
print(y.shape)
```

#### 4. CBAM

```Python
import time
in_planes = 1024
ca = ChannelAttention(in_planes=in_planes).cuda()
sa = SpatialAttention(kernel_size=3).cuda()
print(ca)
print(sa)

x = torch.rand(1, in_planes, 64, 64).cuda()
for i in range(10):
    t0 = time.time()
    y = ca(x) * x
    print("time:{}, shape:{}".format(time.time()-t0, y.shape))
    t1 = time.time()
    y = sa(y) * y
    print("time:{}, shape:{}".format(time.time() - t0, y.shape))
```

#### 5. BlurPool

```Python
C = 1024
pool = BlurPool(channels=C)
x = torch.rand(1, 1024, 32, 32)
print(pool(x).shape)
```

#### 6. ASPP

```python
in_channels = 1024
aspp = ASSP(in_channels=in_channels, output_stride=8)
x = torch.rand(16, in_channels, 32, 32)
print(aspp(x).shape)
```

#### 7.RBF

```Python
in_planes, out_planes = 16, 16
# rfb = BasicRFB(in_planes=in_planes, out_planes=out_planes)
rfb = RFBblock(in_ch=in_planes, residual=True)
x = torch.rand(1, in_planes, 8, 8)
y = rfb(x)
print(y.shape)
print(x[0, 0, :, :])
print(y[0, 0, :, :])
```

#### 8.ASFF

```python
asff = ASFF(level=0)
level0 = torch.rand(1, 512, 16, 16)
level1 = torch.rand(1, 256, 32, 32)
level2 = torch.rand(1, 256, 64, 64)
y = asff(level0, level1, level2)
print(y.shape)
```

