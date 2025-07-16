from typing import Union, Optional, Callable, NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

Padding = Union[tuple[int, int], str]


class SimpleConvHyperparameters(NamedTuple):
    in_channels: int
    out_channels: int

    kernel_size: tuple[int, int] = (3, 3)
    padding: Padding = "same"
    initializer: Optional[Callable] = nn.init.xavier_uniform_
    bias: bool = False


Hp = SimpleConvHyperparameters


class SimpleConv2d(nn.Module):
    """
    Much simpler convolution, without stride or circular padding.
    Also, some defaults are different to the standard convolution.
    This convolution should be useful e.g. to use with Power Iteration,
    since there are less edge cases to consider.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.hp = SimpleConvHyperparameters(*args, **kwargs)
        self.weight = get_kernel_parameter(self.hp)
        self.bias = get_bias_parameter(self.hp)

    def __repr__(self):
        return f"{self.__class__.__name__} with {self.hp}."

    def forward(self, x):
        return F.conv2d(x, self.weight, self.bias, padding=self.hp.padding)


def get_kernel_shape(hp: SimpleConvHyperparameters):
    return hp.out_channels, hp.in_channels, *hp.kernel_size


def get_kernel_parameter(hp: SimpleConvHyperparameters):
    p = nn.Parameter(torch.empty(*get_kernel_shape(hp)))
    hp.initializer(p)
    return p


def get_bias_parameter(hp: SimpleConvHyperparameters):
    return nn.Parameter(torch.zeros(hp.out_channels), requires_grad=hp.bias)


def transpose_conv2d(x, weight, bias=None, padding="same"):
    """ Convolution with Jacobian equal to Jac(Conv(weight).transpose()"""
    ks = weight.shape[-2:]
    if padding == "same":
        padding = tuple((k - 1) // 2 for k in ks)
    return F.conv_transpose2d(x, weight, bias, padding=padding)
