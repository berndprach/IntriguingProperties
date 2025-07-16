"""
Based on https://johanwind.github.io/2022/12/28/cifar_94.html.
"""

from functools import partial

from torch import nn

Conv = partial(nn.Conv2d, kernel_size=(3, 3), padding=(1, 1), bias=False)
Linear = partial(nn.Linear, bias=False)


def get_conv_net(w=64, c=10) -> nn.Sequential:
    return nn.Sequential(
        bn_convolution(3, w),

        bn_convolution(w, w),
        bn_convolution_with_pooling(w, 2*w),

        bn_convolution(2*w, 2*w),
        bn_convolution_with_pooling(2*w, 4*w),

        bn_convolution(4*w, 4*w),
        bn_convolution_with_pooling(4*w, 8*w),

        bn_convolution(8*w, 8*w),

        nn.MaxPool2d(4),
        nn.Flatten(),
        Linear(8*w, c)
    )


def bn_convolution(c_in: int, c_out: int):
    return nn.Sequential(
        Conv(c_in, c_out),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )


def bn_convolution_with_pooling(c_in: int, c_out: int):
    return nn.Sequential(
        Conv(c_in, c_out),
        nn.MaxPool2d(2),
        nn.BatchNorm2d(c_out),
        nn.ReLU(),
    )

