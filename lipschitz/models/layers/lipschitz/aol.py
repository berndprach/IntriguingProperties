"""
Almost Orthogonal Lipschitz (AOL) layer.
Proposed in https://arxiv.org/abs/2208.03160
Code adapted from
"1-Lipschitz Layers Compared: Memory, Speed, and Certifiable Robustness", 2023.
"""

import torch
from torch import Tensor, nn
from torch.nn.functional import conv2d, linear

from ..basic.simple_conv import SimpleConv2d
from ..basic.simple_linear import SimpleLinear
from lipschitz.models.layers.train_val_cache_decorator import train_val_cached


def rescale_kernel(weight: Tensor) -> Tensor:
    """ Expected weight shape: out_channels x in_channels x ks1 x ks_2 """
    _, _, k1, k2 = weight.shape
    weight_tp = weight.transpose(0, 1)
    v = torch.nn.functional.conv2d(
        weight_tp, weight_tp, padding=(k1 - 1, k2 - 1))
    v_scaled = v.abs().sum(dim=(1, 2, 3), keepdim=True).transpose(0, 1)
    return weight / (v_scaled + 1e-6).sqrt()


def rescale_matrix(weight: Tensor) -> Tensor:  # shape: out x in
    ls_bounds_squared = linear_bounds_squared(weight)
    return weight / (ls_bounds_squared + 1e-6).sqrt()  # shape: out x in


def linear_bounds_squared(weight: Tensor) -> Tensor:  # shape: out x in
    wwt = torch.matmul(weight.transpose(0, 1), weight)  # shape: in x in
    ls_bounds_squared = wwt.abs().sum(dim=0, keepdim=True)  # shape: 1 x in
    return ls_bounds_squared  # shape: out x in


class AOLConv2d(SimpleConv2d):
    def __init__(self, *args, initializer=nn.init.dirac_, **kwargs):
        super().__init__(*args, initializer=initializer, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        kernel = self.get_kernel()
        res = conv2d(x, kernel, bias=self.bias, padding=self.hp.padding)
        return res

    @train_val_cached
    def get_kernel(self):
        return rescale_kernel(self.weight)


class AOLLinear(SimpleLinear):
    def __init__(self, *args, initializer=nn.init.eye_, **kwargs):
        super().__init__(*args, initializer=initializer, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        weight = self.get_weight()
        return linear(x, weight, self.bias)

    @train_val_cached
    def get_weight(self):
        return rescale_matrix(self.weight)


class TransposeAOLLinear(AOLLinear):
    """ Uses the fact that W and W^T have the same operator norm. """
    @train_val_cached
    def get_weight(self):
        weight_tp = self.weight.transpose(0, 1)
        rescaled_weight_tp = rescale_matrix(weight_tp)
        rescaled_weight = rescaled_weight_tp.transpose(0, 1)
        return rescaled_weight
