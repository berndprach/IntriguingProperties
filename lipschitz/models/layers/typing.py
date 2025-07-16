from typing import Callable, Protocol

from torch import nn

LinearFactory = Callable[[int, int], nn.Module]
PoolingFactory = Callable[[int], nn.Module]
NormFactory = Callable[[int], nn.Module]
ChannelledFactory = Callable[[int], nn.Module]
ActivationFactory = Callable[[], nn.Module]

KS = tuple[int, int]


class ConvFactory(Protocol):
    def __call__(self,
                 c_in: int,
                 c_out: int,
                 kernel_size: KS = (3, 3),
                 ) -> nn.Module:
        ...
