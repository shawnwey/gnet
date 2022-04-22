import torch
from efficientnet_pytorch import EfficientNet
from torch import nn as nn
from torch.nn import functional as F
from torchvision import transforms
from functools import partial


class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class SAM(nn.Module):
    def __init__(self, feats_shape: list, out_channels):
        super(SAM, self).__init__()
        # branches
        self.branches = self._make_branch(feats_shape)
        self.NUM_BRANCH = len(self.branches)
        # --------------- att ------------------
        # branch att
        # cat + channel att
        self.eca = ECA(1536)
        self.out_transition = nn.Sequential(*self._make_transition(1536, out_channels))

    def forward(self, x_dict):
        x_list = list(x_dict.values())
        for i in range(self.NUM_BRANCH):
            x_list[i] = self.branches[i](x_list[i])
        # --------------- channel att ----------------
        x = torch.cat(x_list, dim=1)
        x = self.eca(x)
        return self.out_transition(x)

    def _make_branch(self, feats_shape) -> list:  # deconv
        self.b1_dd = nn.Sequential(*self._make_downSample(24, 128),
                                   *self._make_downSample(128, 256),
                                   *self._make_transition(256, 256)
                                   )
        self.b2_d = nn.Sequential(*self._make_downSample(32, 256),
                                  *self._make_transition(256, 256)
                                  )
        self.b3 = nn.Sequential(*self._make_transition(56, 256))

        self.b4_u = nn.Sequential(*self._make_upSample(160, 256),
                                  *self._make_transition(160, 256)
                                  )
        self.b5_uu = nn.Sequential(*self._make_upSample(448, 256),
                                   *self._make_upSample(256, 256),
                                   *self._make_transition(448, 256))
        self.b6_uu = nn.Sequential(*self._make_upSample(1792, 512),
                                   *self._make_upSample(512, 256),
                                   *self._make_transition(1792, 256))
        return [self.b1_dd, self.b2_d, self.b3, self.b4_u, self.b5_uu, self.b6_uu]

    def _make_upSample(self, inChannel, outChannel, method='interpolate'):  # deconv
        layers = []
        if method == 'interpolate':
            layers.append(Interpolate())
        else:
            return [
                nn.ConvTranspose2d(inChannel, outChannel, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(outChannel),
                nn.ReLU(inplace=True)
            ]
        return layers

    def _make_downSample(self, inChannel, outChannel) -> list:  # deconv
        layers = [
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True)
        ]
        return layers

    def _make_transition(self, inChannel, outChannel) -> list:
        layers = [
            nn.Conv2d(inChannel, outChannel, 1, 1, bias=False),
            nn.BatchNorm2d(outChannel),
            nn.ReLU(inplace=True)
        ]
        return layers


class ModulizedFunction(nn.Module):
    """Convert a function to an nn.Module."""

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self.fn = partial(fn, *args, **kwargs)

    def forward(self, x):
        return self.fn(x)


class Interpolate(ModulizedFunction):
    def __init__(self, scale_factor=2, mode='bilinear', align_corners=False, **kwargs):
        super().__init__(
            F.interpolate, scale_factor=2, mode='bilinear', align_corners=False, **kwargs)
