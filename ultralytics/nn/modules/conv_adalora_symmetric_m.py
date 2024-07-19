import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.modules.utils import _reverse_repeat_tuple, _pair
from torch.nn import functional as F

import math
from typing import Optional, List, Tuple, Union

__all__ = (
    "Conv_adalora_sym_m",
    "C2f_adalora_sym_m",
    "SPPF_adalora_sym_m",
)

import csv
import numpy as np

# pearson = False
# if pearson:
#     f_cnt = open('count.csv', 'w', encoding='utf-8')
#     fcnt = csv.writer(f_cnt)
#     fcnt.writerow(['0'])
#     f_cnt.close()

# def calculate_pearson(x0, x1):
#     x0_ = x0 - np.mean(x0)
#     x1_ = x1 - np.mean(x1)
#     p = np.dot(x0_, x1_) / (np.linalg.norm(x0_) * np.linalg.norm(x1_))
#     return p

class Conv2d_adalora_sym_m(nn.Module):
    def __init__(self, c1, c2, k, r, lora_alpha, lora_dropout, \
                 stride, padding, groups, dilation, bias):  # merge_weights, 
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        # self.merge_weights = merge_weights

        self.conv = nn.Conv2d(c1, c2, k, stride, padding, dilation, groups, bias)
        if r > 0:
            self.lora_A_rgb = nn.Parameter(self.conv.weight.new_zeros((r, c1 * k)))
            self.lora_A_ir = nn.Parameter(self.conv.weight.new_zeros((r, c1 * k)))
            self.lora_E = nn.Parameter(self.conv.weight.new_zeros((r, 1)))
            self.lora_B_rgb = nn.Parameter(self.conv.weight.new_zeros((c2//self.conv.groups*k, r)))
            self.lora_B_ir = nn.Parameter(self.conv.weight.new_zeros((c2//self.conv.groups*k, r)))
            self.ranknum = nn.Parameter(self.conv.weight.new_zeros(1), requires_grad=False)
            self.ranknum.data.fill_(float(self.r))
            self.scaling = self.lora_alpha / self.r
            self.ranknum.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        self.conv.reset_parameters()
        if hasattr(self, 'lora_A_rgb'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A_rgb, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_A_ir, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B_rgb, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.lora_B_ir, a=math.sqrt(5))
            nn.init.zeros_(self.lora_E)

    def forward(self, x):
        x_rgb = self.conv._conv_forward(
                x[0],
                self.conv.weight + (self.lora_B_rgb @ (self.lora_A_rgb * self.lora_E)).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        x_ir = self.conv._conv_forward(
                x[1],
                self.conv.weight + (self.lora_B_ir @ (self.lora_A_ir * self.lora_E)).view(self.conv.weight.shape) * self.scaling,
                self.conv.bias
            )
        
        # x_rgb1 = self.conv._conv_forward(
        #         x[0],
        #         self.conv.weight,
        #         self.conv.bias
        #     )
        # x_rgb2 = self.conv._conv_forward(
        #         x[0],
        #         (self.lora_B_rgb @ (self.lora_A_rgb * self.lora_E)).view(self.conv.weight.shape) * self.scaling,
        #         self.conv.bias
        #     )
        # x_ir1 = self.conv._conv_forward(
        #         x[1],
        #         self.conv.weight,
        #         self.conv.bias
        #     )
        # x_ir2 = self.conv._conv_forward(
        #         x[1],
        #         (self.lora_B_ir @ (self.lora_A_ir * self.lora_E)).view(self.conv.weight.shape) * self.scaling,
        #         self.conv.bias
        #     )
        
        # if pearson and (x_rgb1.size()[0] != 1):
        #     f_cnt = open('count.csv', 'r', encoding='utf-8')
        #     fcnt = csv.reader(f_cnt)
        #     save_count = int(next(fcnt)[0])
        #     f_cnt.close()

        #     if save_count % 5 == 0:
        #         prefix = 'adalora_r9-6_sym_every'
        #         f_inter = open(f'{prefix}_inter.csv', 'a', encoding='utf-8')
        #         f_rgb_ir = open(f'{prefix}_rgb_ir.csv', 'a', encoding='utf-8')
        #         f_rgb_intra = open(f'{prefix}_rgb_intra.csv', 'a', encoding='utf-8')
        #         f_ir_intra = open(f'{prefix}_ir_intra.csv', 'a', encoding='utf-8')
        #         finter = csv.writer(f_inter)
        #         frgb_ir = csv.writer(f_rgb_ir)
        #         frgb_intra = csv.writer(f_rgb_intra)
        #         fir_intra = csv.writer(f_ir_intra)
                
                
        #         for i in range(x_rgb1.size()[0]):
        #             xrgb1 = x_rgb1[i].view(-1).cpu().numpy()
        #             xrgb2 = x_rgb2[i].view(-1).cpu().numpy()
        #             xir1 = x_ir1[i].view(-1).cpu().numpy()
        #             xir2 = x_ir2[i].view(-1).cpu().numpy()
        #             p_inter = calculate_pearson(xrgb1, xir1)
        #             p_rgb_ir = calculate_pearson(xrgb2, xir2)
        #             p_rgb_intra = calculate_pearson(xrgb1, xrgb2)
        #             p_ir_intra = calculate_pearson(xir1, xir2)
        #             finter.writerow([str(p_inter)])
        #             frgb_ir.writerow([str(p_rgb_ir)])
        #             frgb_intra.writerow([str(p_rgb_intra)])
        #             fir_intra.writerow([str(p_ir_intra)])
        #         f_inter.close()
        #         f_rgb_ir.close()
        #         f_rgb_intra.close()
        #         f_ir_intra.close()
        
        #     save_count += 1
        #     f_cnt = open('count.csv', 'w', encoding='utf-8')
        #     fcnt = csv.writer(f_cnt)
        #     fcnt.writerow([str(save_count)])
        #     f_cnt.close()
        # return (x_rgb1+x_rgb2, x_ir1+x_ir2)

        return (x_rgb, x_ir)


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv_adalora_sym_m(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, r=0, lora_alpha=1, lora_dropout=0., p=None, g=1, d=1, act=True):
        # merge_weights=True, 
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = Conv2d_adalora_sym_m(c1, c2, k, r, lora_alpha, lora_dropout, \
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


class C2f_adalora_sym_m(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, r=0, lora_alpha=1, lora_dropout=0., g=1, e=0.5):
        """Initialize CSP bottleneck layer with two convolutions with arguments ch_in, ch_out, number, shortcut, groups,
        expansion.
        """
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv_adalora_sym_m(c1, 2 * self.c, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m((2 + n) * self.c, c2, 1, 1, r, lora_alpha, lora_dropout)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck_adalora_sym_m(self.c, self.c, shortcut, g, k=(3, 3), e=1.0, \
                                          r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)\
                                              for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        # y = list(self.cv1(x).chunk(2, 1))
        # y.extend(m(y[-1]) for m in self.m)
        # return self.cv2(torch.cat(y, 1))

        x = self.cv1(x)
        y_rgb = list(x[0].chunk(2, 1))
        y_ir = list(x[1].chunk(2, 1))
        for m in self.m:
            y1 = m((y_rgb[-1], y_ir[-1]))
            y_rgb.append(y1[0])
            y_ir.append(y1[1])
        y_rgb = torch.cat(y_rgb, 1)
        y_ir = torch.cat(y_ir, 1)
        return self.cv2((y_rgb, y_ir))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        # y = list(self.cv1(x).split((self.c, self.c), 1))
        # y.extend(m(y[-1]) for m in self.m)
        # return self.cv2(torch.cat(y, 1))

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

class Bottleneck_adalora_sym_m(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5, r=0, lora_alpha=1, lora_dropout=0.):
        """Initializes a bottleneck module with given input/output channels, shortcut option, group, kernels, and
        expansion.
        """
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv_adalora_sym_m(c1, c_, k[0], 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m(c_, c2, k[1], 1, r, lora_alpha, lora_dropout, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """'forward()' applies the YOLO FPN to input data."""
        # return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
        if self.add:
            x_1 = self.cv2(self.cv1(x))
            return (x[0] + x_1[0], x[1] + x_1[1])
        else:
            return self.cv2(self.cv1(x))


class SPPF_adalora_sym_m(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5, r=0, lora_alpha=1, lora_dropout=0.):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv_adalora_sym_m(c1, c_, 1, 1, r, lora_alpha, lora_dropout)
        self.cv2 = Conv_adalora_sym_m(c_ * 4, c2, 1, 1, r, lora_alpha, lora_dropout)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        # x = self.cv1(x)
        # y1 = self.m(x)
        # y2 = self.m(y1)
        # return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))

        x = self.cv1(x)
        y1_rgb = self.m(x[0])
        y1_ir = self.m(x[1])
        y2_rgb = self.m(y1_rgb)
        y2_ir = self.m(y1_ir)
        return self.cv2((torch.cat((x[0], y1_rgb, y2_rgb, self.m(y2_rgb)), 1), \
                         torch.cat((x[1], y1_ir, y2_ir, self.m(y2_ir)), 1)))