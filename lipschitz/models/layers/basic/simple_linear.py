import math
from functools import partial
from typing import Optional, Callable, NamedTuple

import torch
from torch import nn

torch_default_initializer = partial(nn.init.kaiming_uniform_, a=math.sqrt(5))


class SimpleLinearHyperparameters(NamedTuple):
    in_features: int
    out_features: int
    initializer: Optional[Callable] = torch_default_initializer
    bias: bool = False


Hp = SimpleLinearHyperparameters


class SimpleLinear(nn.Module):
    """ Some defaults changed, initializer argument added. """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.hp = SimpleLinearHyperparameters(*args, **kwargs)
        self.weight = get_weight_parameter(self.hp)
        self.bias = get_bias_parameter(self.hp)

    def __repr__(self):
        return f"{self.__class__.__name__} with {self.hp}."

    def forward(self, x):
        return nn.functional.linear(x, self.weight, self.bias)


def get_weight_shape(hp: SimpleLinearHyperparameters):
    return hp.out_features, hp.in_features


def get_weight_parameter(hp: SimpleLinearHyperparameters):
    weight = nn.Parameter(torch.empty(get_weight_shape(hp)))
    hp.initializer(weight)
    return weight


def get_bias_parameter(hp: SimpleLinearHyperparameters):
    initial_bias = torch.zeros(hp.out_features)
    return nn.Parameter(initial_bias, requires_grad=hp.bias)
