from typing import Tuple, Sequence, Optional, Iterable

import torch
from torch import nn as nn
from torch.nn import init
from torch.nn import functional as F
from torchvision import transforms
from functools import partial


from isplutils.utils import _get_shape
from isplutils.containers import (Parallel, SequentialMultiInputMultiOutput)
from isplutils.layers import (Interpolate, Reverse, Sum)


class MSPHead(nn.Module):
    def __init__(self, model):
        super(MSPHead, self).__init__()
        self.efficientNet = model
        # fpn
        feats_shapes = _get_shape(model)
        out_channels = 128
        self.fpn = FPN(feats_shapes, hidden_channels=256, out_channels=out_channels)
        self.predictors = nn.ModuleList([Predictor(out_channels, shape) for shape in feats_shapes])
        # specific classifier
        self.sp_ap = nn.AdaptiveAvgPool2d(1)
        self.sp_classifier = nn.Linear(self.efficientNet._conv_head.out_channels, 1)
        # final classifier
        self.final_classifier = nn.Linear(len(feats_shapes) + 1, 1)

    def forward(self, x_dict):
        feats = tuple(x_dict.values())
        # specific classification
        x = feats[-1]
        x = self.sp_ap(x).flatten(start_dim=1)  # [B, 1792, 1, 1]
        x = self.efficientNet._dropout(x)
        x = self.sp_classifier(x)

        feats = self.fpn(feats)
        outs = [x]
        for i in range(len(self.predictors)):
            outs.append(self.predictors[i](feats[i]))

        x = torch.cat(outs, dim=1)
        x = self.final_classifier(x)
        return x


class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()

    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output


class AttentionConv(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super(AttentionConv, self).__init__()
        # self.sa = SpatialAttention()
        self.eca = ECA(hidden_channels)
        self.conv = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        # att = self.sa(x)
        att = self.eca(x)
        x = x * att
        x = self.conv(x)
        return x


class FPN(nn.Sequential):
    """
    Implementation of the architecture described in the paper
    "Feature Pyramid Networks for Object Detection" by Lin et al.,
    https://arxiv.com/abs/1612.03144.

    Takes in an n-tuple of feature maps in reverse order
    (1st feature map, 2nd feature map, ..., nth feature map), where
    the 1st feature map is the one produced by the earliest layer in the
    backbone network.

    The feature maps are passed through the architecture shown below, producing
    n outputs, such that the height and width of the ith output is equal to
    that of the corresponding input feature map and the number of channels
    is equal to out_channels.

    Returns all outputs as a tuple like so: (1st out, 2nd out, ..., nth out)

    Architecture diagram:

    nth feat. map ────────[nth in_conv]──────────┐────────[nth out_conv]────> nth out
                                                 │
                                             [upsample]
                                                 │
                                                 V
    (n-1)th feat. map ────[(n-1)th in_conv]────>(+)────[(n-1)th out_conv]────> (n-1)th out
                                                 │
                                             [upsample]
                                                 │
                                                 V
            .                     .                           .                    .
            .                     .                           .                    .
            .                     .                           .                    .
                                                 │
                                             [upsample]
                                                 │
                                                 V
    1st feat. map ────────[1st in_conv]────────>(+)────────[1st out_conv]────> 1st out

    """

    def __init__(self,
                 in_feats_shapes: Sequence[Tuple[int, ...]],
                 hidden_channels: int = 256,
                 out_channels: int = 2):
        """Constructor.

        Args:
            in_feats_shapes (Sequence[Tuple[int, ...]]): Shapes of the feature
                maps that will be fed into the network. These are expected to
                be tuples of the form (., C, H, ...).
            hidden_channels (int, optional): The number of channels to which
                all feature maps are convereted before being added together.
                Defaults to 256.
            out_channels (int, optional): Number of output channels. This will
                normally be the number of classes. Defaults to 2.
        """
        # reverse so that the deepest (i.e. produced by the deepest layer in
        # the backbone network) feature map is first.
        in_feats_shapes = in_feats_shapes[::-1]
        in_feats_channels = [s[1] for s in in_feats_shapes]

        # 1x1 conv to make the channels of all feature maps the same
        in_convs = Parallel([
            nn.Conv2d(in_channels, hidden_channels, kernel_size=1)
            for in_channels in in_feats_channels
        ])

        # yapf: disable
        def resize_and_add(to_size):
            return nn.Sequential(
                Parallel([nn.Identity(), Interpolate(size=to_size)]),
                Sum()
            )

        top_down_layer = SequentialMultiInputMultiOutput(
            nn.Identity(),
            *[resize_and_add(to_size=shape[-2:]) for shape in in_feats_shapes[1:]]
        )

        out_convs = Parallel([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
            for _ in in_feats_shapes
        ])
        layers = [
            Reverse(),
            in_convs,
            top_down_layer,
            out_convs,
            Reverse()
        ]
        # yapf: enable
        super().__init__(*layers)


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


class Predictor(nn.Module):
    def __init__(self, in_channel, input_shape):
        super(Predictor, self).__init__()
        self.dropout_rate = 0.5
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(in_channel * input_shape[2] * input_shape[3], 1)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


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
            layers.append(Interpolate(scale_factor=2))
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


class SpatialGroupEnhance(nn.Module):

    def __init__(self, groups=8):
        super().__init__()
        self.groups=groups
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.weight=nn.Parameter(torch.zeros(1,groups,1,1))
        self.bias=nn.Parameter(torch.zeros(1,groups,1,1))
        self.sig=nn.Sigmoid()
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h,w=x.shape
        x=x.view(b*self.groups,-1,h,w) #bs*g,dim//g,h,w
        xn=x*self.avg_pool(x) #bs*g,dim//g,h,w
        xn=xn.sum(dim=1,keepdim=True) #bs*g,1,h,w
        t=xn.view(b*self.groups,-1) #bs*g,h*w

        t=t-t.mean(dim=1,keepdim=True) #bs*g,h*w
        std=t.std(dim=1,keepdim=True)+1e-5
        t=t/std #bs*g,h*w
        t=t.view(b,self.groups,h,w) #bs,g,h*w

        t=t*self.weight+self.bias #bs,g,h*w
        t=t.view(b*self.groups,1,h,w) #bs*g,1,h*w
        x=x*self.sig(t)
        x=x.view(b,c,h,w)

        return x