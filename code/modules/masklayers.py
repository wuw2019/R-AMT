from typing import Optional, List, Tuple, Union
"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

DEFAULT_THRESHOLD = None

class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self):
        super(Binarizer, self).__init__()

    @staticmethod
    def forward(self, inputs, threshold):
        outputs = inputs.clone()
        outputs[inputs.le(threshold)] = 0
        outputs[inputs.gt(threshold)] = 1
        return outputs

    @staticmethod
    def backward(self, gradOutput):
        return (gradOutput, None)
        


class ElementWiseConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None, apply_mask=True):
        super(ElementWiseConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.apply_mask = apply_mask

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(
                out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)


        if apply_mask:
            # Initialize real-valued mask weights.
            self.mask_real = self.weight.data.new(self.weight.size())
            self.threshold = threshold
            if mask_init == '1s':
                self.mask_real.fill_(mask_scale)
            elif mask_init == 'uniform':
                self.mask_real.uniform_(0, mask_scale)
                # self.mask_real.uniform_(-1 * mask_scale, mask_scale)
            # mask_real is now a trainable parameter.
            self.mask_real = Parameter(self.mask_real)

            self.soft=False
            # Initialize the thresholder.
            if threshold_fn == 'binarizer':
                self.threshold_fn = Binarizer().apply
            elif threshold_fn == 'ternarizer':
                self.threshold_fn = Ternarizer().apply

    def forward(self, input):
        if self.apply_mask:
            # Get binarized/ternarized mask from real-valued mask.
            if self.soft:
                mask_thresholded = self.threshold_fn(self.mask_real, "simple")
            elif self.training:
                mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
            else:
                mask_thresholded = (self.mask_real>self.threshold) #.float()
            # Mask weights with above mask.
            weight_thresholded = mask_thresholded * self.weight
            # Perform conv using modified weight.
            return F.conv2d(input, weight_thresholded, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
        else:
            return F.conv2d(input, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
    
    def get_weight(self):
        if self.training:
            mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
        else:
            mask_thresholded = (self.mask_real>self.threshold) #.float()
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        return weight_thresholded

    # def get_bias(self):
    #     return self.bias



    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if self.bias is not None and self.bias.data is not None:
            self.bias.data = fn(self.bias.data)


class ElementWiseLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super(ElementWiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(
            out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(
                out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # Initialize real-valued mask weights.
        self.mask_real = self.weight.data.new(self.weight.size())
        if mask_init == '1s':
            self.mask_real.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.mask_real.uniform_(0, mask_scale)
            # self.mask_real.uniform_(-1 * mask_scale, mask_scale)

        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)
        self.threshold = threshold
        self.soft=False
        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer().apply
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer().apply

    def forward(self, input):
        # Get binarized/ternarized mask from real-valued mask.
        if self.soft:
            mask_thresholded = self.threshold_fn(self.mask_real, "simple")
        elif self.training:
            mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
        else:
            mask_thresholded = (self.mask_real>self.threshold) #.float()
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        # Get output using modified weight.
        return F.linear(input, weight_thresholded, self.bias)

    def get_weight(self):
        if self.soft:
            mask_thresholded = self.threshold_fn(self.mask_real, "simple")
        elif self.training:
            mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
        else:
            mask_thresholded = (self.mask_real>self.threshold) #.float()
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        return weight_thresholded

    def get_bias(self):
        return self.bias

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)

class ElementWiseNonDynamicallyQuantizableLinear(ElementWiseLinear):
    def __init__(self, in_features, out_features, bias = True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super().__init__(in_features, out_features, bias=bias,
                         mask_init=mask_init, mask_scale=mask_scale,
                        threshold_fn=threshold_fn, threshold=threshold)

class NonDynamicallyQuantizableLinear(nn.Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)


class ElementWiseMultiheadAttention(nn.Module):

    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False,
                 kdim=None, vdim=None, batch_first=False, device=None, dtype=None,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None, mask_attn=True) -> None:

        super(ElementWiseMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        assert self._qkv_same_embed_dim

        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init
        self.mask_attn = mask_attn

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        self.in_proj_weight = Variable(torch.Tensor(3 * embed_dim, embed_dim), requires_grad=False)
        self.register_parameter('q_proj_weight', None)
        self.register_parameter('k_proj_weight', None)
        self.register_parameter('v_proj_weight', None)

        if bias:
            self.in_proj_bias = Variable(torch.Tensor(
                3 * embed_dim), requires_grad=False)
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = ElementWiseNonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, mask_init=mask_init, mask_scale=mask_scale,
                        threshold_fn=threshold_fn, threshold=threshold)

        assert not add_bias_kv
        self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        if self.mask_attn:
            # Initialize real-valued mask weights.
            self.mask_real = self.in_proj_weight.data.new(self.in_proj_weight.size())
            if mask_init == '1s':
                self.mask_real.fill_(mask_scale)
            elif mask_init == 'uniform':
                self.mask_real.uniform_(0, mask_scale)
                # self.mask_real.uniform_(-1 * mask_scale, mask_scale)

            # mask_real is now a trainable parameter.
            self.mask_real = Parameter(self.mask_real)
            self.threshold = threshold
            self.soft=False
            # Initialize the thresholder.
            if threshold_fn == 'binarizer':
                self.threshold_fn = Binarizer().apply
            elif threshold_fn == 'ternarizer':
                self.threshold_fn = Ternarizer().apply


    def forward(self, query: Tensor, key: Tensor, value: Tensor, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:

        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        if self.mask_attn:
            if self.soft:
                mask_thresholded = self.threshold_fn(self.mask_real, "simple")
            elif self.training:
                mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
            else:
                mask_thresholded = (self.mask_real>self.threshold) #.float()
            # Mask weights with above mask.
            weight_thresholded = mask_thresholded * self.in_proj_weight
        else: weight_thresholded = self.in_proj_weight
        
        attn_output, attn_output_weights = F.multi_head_attention_forward(
            query, key, value, self.embed_dim, self.num_heads,
            weight_thresholded, self.in_proj_bias,
            self.bias_k, self.bias_v, self.add_zero_attn,
            self.dropout, self.out_proj.get_weight(), self.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask, need_weights=need_weights,
            attn_mask=attn_mask, average_attn_weights=average_attn_weights)

        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.in_proj_weight.data = fn(self.in_proj_weight.data)
        self.in_proj_bias.data = fn(self.in_proj_bias.data)

    def get_weight(self):
        if self.mask_attn:
            if self.soft:
                mask_thresholded = self.threshold_fn(self.mask_real, "simple")
            elif self.training:
                mask_thresholded = self.threshold_fn(self.mask_real, self.threshold)
            else:
                mask_thresholded = (self.mask_real>self.threshold) #.float()
            # Mask weights with above mask.
            weight_thresholded = mask_thresholded * self.in_proj_weight
        else: weight_thresholded = self.in_proj_weight

        return weight_thresholded, self.out_proj.get_weight()