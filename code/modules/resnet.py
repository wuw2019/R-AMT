from collections import OrderedDict
from typing import Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo

import modules.masklayers as nl

__all__ = ['MaskModifiedResNet', 'mask_resnet50']

class MaskBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, mask_init, mask_scale, threshold_fn, threshold, stride=1):
        super().__init__()
        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nl.ElementWiseConv2d(inplanes, planes, kernel_size=1, bias=False,mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nl.ElementWiseConv2d(planes, planes, kernel_size=3, padding=1, bias=False,mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)
        self.bn2 = nn.BatchNorm2d(planes)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nl.ElementWiseConv2d(planes, planes * self.expansion, kernel_size=1, bias=False, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * self.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nl.ElementWiseConv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn, threshold=threshold)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class MaskAttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None, mask_init: str = 'lr', mask_scale: float = 1e-2, threshold_fn: str = 'binarizer', threshold: float = 5e-3):
        super().__init__()
        # self.positional_embedding = Variable(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5, requires_grad=False).cuda()
        # positional_embedding_mask_real = self.positional_embedding.data.new(self.positional_embedding.size())
        # self.threshold = threshold
        # if mask_init == '1s':
        #     positional_embedding_mask_real.fill_(mask_scale)
        # elif mask_init == 'uniform':
        #     positional_embedding_mask_real.uniform_(0, mask_scale)
        # self.positional_embedding_mask_real = nn.Parameter(positional_embedding_mask_real)
        # self.threshold_fn = nl.Binarizer().apply
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)

        self.k_proj = nl.ElementWiseLinear(embed_dim, embed_dim, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)
        self.q_proj = nl.ElementWiseLinear(embed_dim, embed_dim, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)
        self.v_proj = nl.ElementWiseLinear(embed_dim, embed_dim, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)
        self.c_proj = nl.ElementWiseLinear(embed_dim, output_dim or embed_dim, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        #positional_embedding = self.positional_embedding*self.threshold_fn(self.positional_embedding_mask_real, self.threshold)
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.get_weight(),
            k_proj_weight=self.k_proj.get_weight(),
            v_proj_weight=self.v_proj.get_weight(),
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.get_bias(), self.k_proj.get_bias(), self.v_proj.get_bias()]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.get_weight(),
            out_proj_bias=self.c_proj.get_bias(),
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class MaskModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, mask_init, mask_scale, threshold_fn, threshold: float = 5e-3, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        
        # the 3-layer stem
        self.conv1 = nl.ElementWiseConv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn,threshold=threshold)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.conv2 = nl.ElementWiseConv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn,threshold=threshold)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.conv3 = nl.ElementWiseConv2d(width // 2, width, kernel_size=3, padding=1, bias=False, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn,threshold=threshold)
        self.bn3 = nn.BatchNorm2d(width)
        self.avgpool = nn.AvgPool2d(2)
        self.relu = nn.ReLU(inplace=True)

        # self.conv1 = nl.ElementWiseConv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False, apply_mask=False)
        # self.bn1 = nn.BatchNorm2d(width // 2)
        # self.conv2 = nl.ElementWiseConv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False, apply_mask=False)
        # self.bn2 = nn.BatchNorm2d(width // 2)
        # self.conv3 = nl.ElementWiseConv2d(width // 2, width, kernel_size=3, padding=1, bias=False, apply_mask=False)
        # self.bn3 = nn.BatchNorm2d(width)
        # self.avgpool = nn.AvgPool2d(2)
        # self.relu = nn.ReLU(inplace=True)
        
        
        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0], mask_init, mask_scale, threshold_fn, threshold)
        self.layer2 = self._make_layer(width * 2, layers[1], mask_init, mask_scale, threshold_fn, threshold, stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], mask_init, mask_scale, threshold_fn, threshold, stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], mask_init, mask_scale, threshold_fn, threshold, stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = MaskAttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim,mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn, threshold=threshold)

    def _make_layer(self, planes, blocks, mask_init, mask_scale, threshold_fn, threshold, stride=1):
        layers = [MaskBottleneck(self._inplanes, planes, mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold, stride=stride)]

        self._inplanes = planes * MaskBottleneck.expansion
        for _ in range(1, blocks):
            layers.append(MaskBottleneck(self._inplanes, planes,mask_init=mask_init, mask_scale=mask_scale,
                 threshold_fn=threshold_fn, threshold=threshold))

        return nn.Sequential(*layers)

    def forward(self, x, return_map=False):
        def stem(x):
            for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2), (self.conv3, self.bn3)]:
                x = self.relu(bn(conv(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        f = self.attnpool(x)
        if return_map:
            return f, x
        return f

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(MaskModifiedResNet, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()


def mask_resnet50(mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', **kwargs):
    """Constructs a ResNet-50 model."""
    model = MaskModifiedResNet(MaskBottleneck, [3, 4, 6, 3], mask_init,
                   mask_scale, threshold_fn, **kwargs)
    return model
