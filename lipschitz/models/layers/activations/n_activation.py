
from typing import Tuple, Protocol

import torch
import torch.nn as nn


def n_activation(x: torch.Tensor, theta: torch.Tensor):
    # x.shape e.g. [bs, c, h, w] or [bs, c].
    # theta.shape [c, 2].

    theta_sorted, _ = torch.sort(theta, dim=1)
    for _ in range(len(x.shape) - 2):
        theta_sorted = theta_sorted[..., None]

    line1 = x - 2 * theta_sorted[:, 0]
    line2 = -x
    line3 = x - 2 * theta_sorted[:, 1]

    piece1 = line1
    piece2 = torch.where(
        torch.less(x, theta_sorted[:, 0]),
        piece1,
        line2,
    )
    piece3 = torch.where(
        torch.less(x, theta_sorted[:, 1]),
        piece2,
        line3,
    )

    result = piece3
    return result


class Initializer(Protocol):
    def __call__(self,
                 shape: Tuple[int, ...],
                 device: torch.device = None,
                 ) -> torch.Tensor:
        ...


def zero_initializer(shape, device) -> torch.Tensor:
    return torch.zeros(shape, device=device)


class NActivation(nn.Module):
    """
    N activation function with learnable parameters.
    !!! Requires a model forward pass before passing model.parameters()
    to optimizer. !!!
    """
    def __init__(self,
                 # in_channels: int,
                 initializer: Initializer = zero_initializer,
                 trainable: bool = True,
                 lr_factor: float = 1.,  # Changes grad/theta ratio.
                 ):
        super().__init__()

        self.initializer = initializer
        self.trainable = trainable
        self.rescaling = lr_factor ** 0.5
        self.theta = None

    def forward(self, x: torch.Tensor):
        # x.shape e.g. [bs, c, h, w] or [bs, c].
        if self.theta is None:
            self.initialize(x.shape[1], x.device)

        theta = self.theta * self.rescaling
        assert x.shape[1] == theta.shape[0], \
            f"Dimension mismatch! {x.shape=} != {theta.shape=}"
        return n_activation(x, theta)

    def initialize(self, channels: int, device: torch.device):
        theta = self.initializer(shape=(channels, 2), device=device)
        rescaled_theta = theta / self.rescaling
        self.theta = nn.Parameter(rescaled_theta, requires_grad=self.trainable)


class ThetaInitializer:
    def __init__(self, values: Tuple):
        self.values = values

    def __call__(self, shape, device):
        assert shape[1] == 2, "The second dimension must be 2."

        theta_values = torch.ones(shape, device=device)
        theta_values = theta_values.reshape(-1, len(self.values))
        theta_values = theta_values * torch.tensor(self.values, device=device)
        theta_values = theta_values.reshape(shape)
        return theta_values
