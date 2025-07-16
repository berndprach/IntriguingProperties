from typing import Callable

import torch
from torch import nn
from torch.nn.init import eye_ as eye
from torch.nn.init import orthogonal_ as orthogonal

Initializer = Callable[[torch.nn.Parameter], None]


def orthogonal_center(tensor: torch.nn.Parameter) -> None:
    """ K[:, :, 1, 1] = orthogonal, other entries are zero. """
    ks1, ks2 = tensor.shape[2:]
    tensor.data.fill_(0)
    nn.init.orthogonal_(tensor[:, :, ks1 // 2, ks2 // 2])
    return None


INITIALIZER: dict[str, Initializer] = {
    "eye": eye,
    "orthogonal": orthogonal,
    "orthogonal_center": orthogonal_center,
}


# def get(name: str, **kwargs) -> Callable[[], None]:
#     fn = globals()[name]
#     return partial(fn, **kwargs)
