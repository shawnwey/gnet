import torch
import torch.nn as nn
from architectures.gmodel import Interpolate

x = torch.randn(1,1,56,56)
# l = Interpolate()
# l = nn.ConvTranspose2d(1,1,3, stride=2, padding=1, output_padding=1)#Conv2d(1, 1, kernel_size=3,stride=1,padding=0)
l = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)
y = l(x) # y.shape:[1,1,4,4]
print(y.shape)