import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _reverse_repeat_tuple, _pair
from torch.nn import functional as F

import math
from typing import Optional, List, Tuple, Union

__all__ = (
    "Conv_all_lora_m",
    "C2f_all_lora_m",
    "SPPF_all_lora_m",
)

class _ConvNd_lora(nn.Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:  # type: ignore[empty-body]
        ...

    in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    # transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                #  transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if groups <= 0:
            raise ValueError('groups must be a positive integer')
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    f"Invalid padding string {padding!r}, should be one of {valid_padding_strings}")
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError(f"padding_mode must be one of {valid_padding_modes}, but got padding_mode='{padding_mode}'")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        # self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        # if transposed:
        #     self.weight = nn.Parameter(torch.empty(
        #         (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        # else:
        #     self.weight = nn.Parameter(torch.empty(
        #         (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        # self.weight = nn.Parameter(torch.empty(
        #         (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        # nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            if fan_in != 0:
                bound = 1 / math.sqrt(fan_in)
                nn.init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
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
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'

class Conv2d_all_lora_m(_ConvNd_lora):
    def __init__(self, c1, c2, k, r, lora_alpha, lora_dropout, \
                 stride, padding, groups, dilation, bias,\
                padding_mode: str = 'zeros', device=None, dtype=None):  # merge_weights, 
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(k)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super().__init__(
            c1, c2, kernel_size_, stride_, padding_, dilation_,
            _pair(0), groups, bias, padding_mode, **factory_kwargs)
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # self.merge_weights = merge_weights

        # self.inter_r = math.floor(min(k * k, c2 * c1) * r) if math.floor(min(k * k, c2 * c1) * r) >= 1 else 1
        # self.inter_r_aux = math.ceil(min(k * k, c2 * c1) * (1 - r)) if math.floor(min(k * k, c2 * c1) * (1 - r)) >= 1 else 1
        self.inter_r = math.floor(((c1*c2*k*k) / (c1*k + c2*k)) * r) if math.floor(((c1*c2*k*k) / (c1*k + c2*k)) * r) >= 1 else 1
        self.inter_r_aux = math.floor(((c1*c2*k*k) / (c1*k + c2*k)) * (1 - r)) if math.floor(((c1*c2*k*k) / (c1*k + c2*k)) * (1 - r)) >= 1 else 1
        
        self.lora_A = nn.Parameter(torch.empty((self.inter_r, c2 * k), **factory_kwargs))    # min(k * k, c2 * c1)
        self.lora_B = nn.Parameter(torch.empty((k * c1, self.inter_r), **factory_kwargs))
        self.lora_A_rgb = nn.Parameter(torch.empty((self.inter_r_aux, c2 * k), **factory_kwargs))    # min(k * k, c2 * c1)
        self.lora_B_rgb = nn.Parameter(torch.empty((k * c1, self.inter_r_aux), **factory_kwargs))
        self.lora_A_ir = nn.Parameter(torch.empty((self.inter_r_aux, c2 * k), **factory_kwargs))    # min(k * k, c2 * c1)
        self.lora_B_ir = nn.Parameter(torch.empty((k * c1, self.inter_r_aux), **factory_kwargs))
        self.scaling = self.lora_alpha / self.r
        self.reset_parameters()

    def reset_parameters(self):
        # self.conv.reset_parameters()
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_rgb, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B_rgb, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_ir, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B_ir, a=math.sqrt(5))
            # nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if self.padding_mode != 'zeros':
            x_rgb = F.conv2d(F.pad(x[0], self._reversed_padding_repeated_twice, mode=self.padding_mode),
                        (torch.cat([self.lora_B, self.lora_B_rgb], 1) @ torch.cat([self.lora_A, self.lora_A_rgb], 0))\
                            .view(self.out_channels // self.groups, self.in_channels, self.kernel_size[0], self.kernel_size[1]) \
                            * self.scaling, 
                        self.bias, 
                        self.stride,
                        _pair(0), self.dilation, self.groups)
            x_ir = F.conv2d(F.pad(x[1], self._reversed_padding_repeated_twice, mode=self.padding_mode),
                        (torch.cat([self.lora_B, self.lora_B_ir], 1) @ torch.cat([self.lora_A, self.lora_A_ir], 0))\
                            .view(self.out_channels // self.groups, self.in_channels, self.kernel_size[0], self.kernel_size[1]) \
                            * self.scaling, 
                        self.bias, 
                        self.stride,
                        _pair(0), self.dilation, self.groups)
            return (x_rgb, x_ir)
        
        x_rgb = F.conv2d(x[0], 
                        (torch.cat([self.lora_B, self.lora_B_rgb], 1) @ torch.cat([self.lora_A, self.lora_A_rgb], 0))\
                            .view(self.out_channels // self.groups, self.in_channels, self.kernel_size[0], self.kernel_size[1]) \
                            * self.scaling, 
                        self.bias, 
                        self.stride,
                        self.padding, self.dilation, self.groups)
        x_ir = F.conv2d(x[1], 
                        (torch.cat([self.lora_B, self.lora_B_ir], 1) @ torch.cat([self.lora_A, self.lora_A_ir], 0))\
                            .view(self.out_channels // self.groups, self.in_channels, self.kernel_size[0], self.kernel_size[1]) \
                            * self.scaling, 
                        self.bias, 
                        self.stride,
                        self.padding, self.dilation, self.groups)
        return (x_rgb, x_ir)
   

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv_all_lora_m(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, r=0, lora_alpha=1, lora_dropout=0., p=None, g=1, d=1, act=True):
        # merge_weights=True, 
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = Conv2d_all_lora_m(c1, c2, k, r, lora_alpha, lora_dropout, \
                           stride=s, padding=autopad(k, p, d), groups=g, dilation=d, bias=False)
        # merge_weights, 
        self.bn_rgb = nn.BatchNorm2d(c2)
        self.bn_ir = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        x = self.conv(x)
        x_rgb = self.act(self.bn_rgb(x[0]))
        x_ir = self.act(self.bn_ir(x[1]))
        return (x_rgb, x_ir)

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        x = self.conv(x)
        x_rgb = self.act(x[0])
        x_ir = self.act(x[1])
        return (x_rgb, x_ir)


class C2f_all_lora_m(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, r=0, lora_alpha=1, lora_dropout=0., g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv_all_lora_m(c1, 2 * self.c, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_all_lora_m((2 + n) * self.c, c2, 1, 1, r, lora_alpha, lora_dropout)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_all_lora_m(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, \
                                          r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)\
                                              for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        x = self.cv1(x)
        y_rgb = list(x[0].chunk(2, 1))
        y_ir = list(x[1].chunk(2, 1))
        # y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y1 = m((y_rgb[-1], y_ir[-1]))
            y_rgb.append(y1[0])
            y_ir.append(y1[1])
        y_rgb = torch.cat(y_rgb, 1)
        y_ir = torch.cat(y_ir, 1)
        return self.cv2((y_rgb, y_ir))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        x = self.cv1(x)
        y_rgb = list(x[0].split((self.c, self.c), 1))
        y_ir = list(x[1].split((self.c, self.c), 1))
        # y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m)
        for m in self.m:
            y1 = m((y_rgb[-1], y_ir[-1]))
            y_rgb.append(y1[0])
            y_ir.append(y1[1])
        y_rgb = torch.cat(y_rgb, 1)
        y_ir = torch.cat(y_ir, 1)
        return self.cv2((y_rgb, y_ir))

class Bottleneck_all_lora_m(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, r=0, lora_alpha=1, lora_dropout=0.):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_all_lora_m(c1, c_, k[0], 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_all_lora_m(c_, c2, k[1], 1, r, lora_alpha, lora_dropout, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        if self.add:
            x_1 = self.cv2(self.cv1(x))
            return (x[0] + x_1[0], x[1] + x_1[1])
        else:
            return self.cv2(self.cv1(x))
        # return (x[0] + x_1[0], x[1] + x_1[1]) if self.add \
        #     else self.cv2(self.cv1(x))


class SPPF_all_lora_m(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, r=0, lora_alpha=1, lora_dropout=0.):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv_all_lora_m(c1, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_all_lora_m(c_ * 4, c2, 1, 1, r, lora_alpha, lora_dropout)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        x = self.cv1(x)
        y1_rgb = self.m(x[0])
        y1_ir = self.m(x[1])
        y2_rgb = self.m(y1_rgb)
        y2_ir = self.m(y1_ir)
        return self.cv2((torch.cat((x[0], y1_rgb, y2_rgb, self.m(y2_rgb)), 1), \
                         torch.cat((x[1], y1_ir, y2_ir, self.m(y2_ir)), 1)))