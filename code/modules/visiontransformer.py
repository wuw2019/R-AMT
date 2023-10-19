from collections import OrderedDict
from typing import Tuple, Union
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo

import modules.masklayers as nl
from clip.model import LayerNorm, QuickGELU

__all__ = ['MaskTransformer', 'MaskVisionTransformer']

class MaskResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, mask_init: str = 'lr', mask_scale: float = 1e-2, threshold_fn: str = 'binarizer', threshold: float = 5e-3, mask_mlp: bool = True):
        super().__init__()

        self.attn = nl.ElementWiseMultiheadAttention(d_model, n_head, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)
        # self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)

        if mask_mlp:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nl.ElementWiseLinear(d_model, d_model * 4, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)),
                ("gelu", QuickGELU()),
                ("c_proj", nl.ElementWiseLinear(d_model * 4, d_model, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold))
            ]))
        else:
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, d_model * 4)),
                ("gelu", QuickGELU()),
                ("c_proj", nn.Linear(d_model * 4, d_model))
            ]))

        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]


    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class MaskTransformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, mask_init: str = 'lr', mask_scale: float = 1e-2, threshold_fn: str = 'binarizer', threshold: float = 5e-3, mask_mlp: bool = True):
        super().__init__()
        self.width = width
        self.heads = heads
        self.layers = layers
        self.resblocks = nn.Sequential(*[MaskResidualAttentionBlock(width, heads, attn_mask, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold, mask_mlp=mask_mlp) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class MaskVisionTransformer(nn.Module):
    def __init__(self, 
        input_resolution: int, 
        patch_size: int, 
        width: int, 
        layers: int, 
        heads: int, 
        output_dim: int, 
        mask_init: str = 'lr', 
        mask_scale: float = 1e-2, 
        threshold_fn: str = 'binarizer', 
        threshold: float = 5e-3,
        mask_mlp: bool = True):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nl.ElementWiseConv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False, \
        mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold)
        #nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = MaskTransformer(
            width, layers, heads, mask_init=mask_init, mask_scale=mask_scale, threshold_fn=threshold_fn,threshold=threshold, mask_mlp=mask_mlp
            )

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x