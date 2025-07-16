"""
Power Iteration Convolution Layer
"""
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.functional import conv2d, linear

from lipschitz.models.layers.train_val_cache_decorator import train_val_cached
from .power_iteration import (
    ConvolutionalPowerIteration, LinearPowerIteration, get_eigenvalue,
    PowerIteration
)
from ..basic import simple_conv
from ..basic.simple_conv import SimpleConv2d
from ..basic.simple_linear import SimpleLinear

PICConv2dHp = partial(simple_conv.SimpleConvHyperparameters,
                      initializer=nn.init.dirac_)


class PIPConv2d(SimpleConv2d):  # Power Iteration Powered Convolution Layer
    def __init__(self,
                 *args,
                 initializer=nn.init.dirac_,
                 val_iterations=10_000,
                 **kwargs):
        super().__init__(*args, initializer=initializer, **kwargs)
        self.val_iterations = val_iterations

        self.training_pi = self.get_power_iteration()

    def forward(self, x: Tensor) -> Tensor:
        if self.training_pi.u is None:
            self.initialize_u(x)

        x = conv2d(x, self.weight, bias=self.bias, padding=self.hp.padding)
        return x / (self.get_lipschitz_constant() + 1e-12)

    def initialize_u(self, x):
        u_size = (1, self.weight.shape[0], x.shape[-2], x.shape[-1])
        self.training_pi.u = torch.randn(u_size, device=x.device)

    @train_val_cached
    def get_lipschitz_constant(self):
        return get_eigenvalue(self)

    def get_power_iteration(self):
        return ConvolutionalPowerIteration(self.weight, self.hp.padding)


class PIPLinear(SimpleLinear):
    def __init__(self,
                 *args,
                 initializer=nn.init.eye_,
                 val_iterations=10_000,
                 **kwargs):
        super().__init__(*args, initializer=initializer, **kwargs)
        self.val_iterations = val_iterations
        self.training_pi = self.get_power_iteration()

    def forward(self, x: Tensor) -> Tensor:
        if self.training_pi.u is None:
            self.training_pi.u = torch.randn((1, x.shape[-1]), device=x.device)

        x = linear(x, self.weight, bias=self.bias)
        return x / (self.get_lipschitz_constant() + 1e-12)

    @train_val_cached
    def get_lipschitz_constant(self):
        return get_eigenvalue(self)

    def get_power_iteration(self):
        return LinearPowerIteration(self.weight)
