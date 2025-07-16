from typing import Protocol

import torch
from torch.nn.functional import conv2d, linear

from ..basic.simple_conv import transpose_conv2d


class PowerIteration:
    def __init__(self, forward_op, transpose_op):
        self.forward_op = forward_op
        self.transpose_op = transpose_op

        self.u = None
        self.v = None  # For eigenvalue computation

    def __call__(self):
        return self.step()

    def initialize(self, u_size, device):
        # Alternatively, set e.g. pi.u = torch.randn(1, 16, 32, 32) directly.
        self.u = torch.randn(u_size, device=device)
        self.v = None

    def step(self):
        self.v = normalize(self.forward_op(self.u))
        self.u = normalize(self.transpose_op(self.v))

    def n_steps(self, n=1):
        for _ in range(n):
            self.step()

    def get_eigenvalue(self):
        return (self.v * self.forward_op(self.u)).sum()


class ConvolutionalPowerIteration(PowerIteration):
    def __init__(self, kernel, padding):
        super().__init__(
            lambda x: conv2d(x, kernel, padding=padding),
            lambda x: transpose_conv2d(x, kernel, padding=padding),
        )


class LinearPowerIteration(PowerIteration):
    def __init__(self, weight):
        super().__init__(
            lambda x: linear(x, weight),
            lambda x: linear(x, weight.t()),
        )


def normalize(x, epsilon=1e-12):
    # return x / (torch.norm(x, p=2) + epsilon)
    return x / (torch.linalg.vector_norm(x, ord=2) + epsilon)


class PILayer(Protocol):
    training_pi: PowerIteration
    training: bool
    val_iterations: int

    def get_power_iteration(self) -> PowerIteration:
        ...


def get_eigenvalue(layer: PILayer) -> float:
    if layer.training:
        with torch.no_grad():
            layer.training_pi.step()
        return layer.training_pi.get_eigenvalue()
    else:
        pi = layer.get_power_iteration()
        pi.u = torch.randn_like(layer.training_pi.u)
        with torch.no_grad():
            pi.n_steps(n=layer.val_iterations)
        return pi.get_eigenvalue()
