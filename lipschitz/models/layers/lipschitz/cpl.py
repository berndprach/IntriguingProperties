"""
Adapted from
https://github.com/berndprach/1LipschitzLayersCompared/blob/main/models/layers/lipschitz/cpl.py
and
https://github.com/MILES-PSL/Convex-Potential-Layer/blob/main/layers.py

Changes compared to the original implementation:
 - Added caching
 - Initializing u randomly for validation, in order to get guarantees.
 - Not changing the training_u during validation forward passes.
"""
import math
from functools import partial

import torch
from torch import Tensor, nn
from torch.nn.functional import conv2d, linear

from lipschitz.models.layers.train_val_cache_decorator import train_val_cached
from .power_iteration import ConvolutionalPowerIteration, get_eigenvalue
from .power_iteration import LinearPowerIteration
from ..basic.simple_conv import transpose_conv2d, SimpleConv2d
from ..basic.simple_linear import SimpleLinear


class CPLConv2d(SimpleConv2d):
    def __init__(self,
                 *args,
                 initializer=partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
                 val_iterations=1000,
                 bias=True,
                 **kwargs):
        super().__init__(*args, initializer=initializer, bias=bias, **kwargs)
        self.val_iterations = val_iterations

        self.training_pi = self.get_power_iteration()

    def forward(self, x: Tensor) -> Tensor:
        if self.training_pi.u is None:
            u_size = (1, self.weight.shape[0], x.shape[-2], x.shape[-1])
            self.training_pi.initialize(u_size, device=x.device)

        res = conv2d(x, self.weight, bias=self.bias, padding=self.hp.padding)
        residual = torch.relu(res)
        residual = transpose_conv2d(
            residual, self.weight, bias=self.bias, padding=self.hp.padding
        )

        rescaling = self.get_rescaling()
        return x - rescaling * residual

    @train_val_cached
    def get_rescaling(self):
        ev = get_eigenvalue(self)
        return cpl_rescaling(ev)
        # if self.training:
        #     with torch.no_grad():
        #         self.training_pi.step()
        #     return cpl_rescaling(self.training_pi.get_eigenvalue())
        # else:
        #     pi = ConvolutionalPI(self.weight, self.hp.padding)
        #     pi.u = torch.randn_like(self.training_pi.u)
        #     with torch.no_grad():
        #         pi.n_steps(n=self.val_iterations)
        #     return cpl_rescaling(pi.get_eigenvalue())

    def get_power_iteration(self):
        return ConvolutionalPowerIteration(self.weight, self.hp.padding)


def cpl_rescaling(eigenvalue):
    return 2. / (eigenvalue**2 + 1e-12)


class CPLLinear(SimpleLinear):
    def __init__(self,
                 *args,
                 initializer=partial(nn.init.kaiming_uniform_, a=math.sqrt(5)),
                 bias=True,
                 val_iterations=1000,
                 **kwargs):
        super().__init__(*args, initializer=initializer, bias=bias, **kwargs)
        self.val_iterations = val_iterations

        self.training_pi = LinearPowerIteration(self.weight)

    def forward(self, x: Tensor) -> Tensor:
        if self.training_pi.u is None:
            self.training_pi.u = torch.randn((1, x.shape[-1]), device=x.device)

        residual = linear(x, self.weight, bias=self.bias)
        residual = torch.relu(residual)
        residual = linear(residual, self.weight.t(), bias=self.bias)

        rescaling = self.get_rescaling()
        return x - rescaling * residual

    @train_val_cached
    def get_rescaling(self):
        ev = get_eigenvalue(self)
        return cpl_rescaling(ev)

    def get_power_iteration(self):
        return LinearPowerIteration(self.weight)



