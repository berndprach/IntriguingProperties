from typing import Protocol

from torch.optim import SGD as _SGD

from lipschitz.io_functions.key_not_found_error import KeyNotFoundError


class Optimizer(Protocol):
    lr: float  # Used by schedulers.

    def __init__(self, parameters, **kwargs):
        ...

    def step(self):
        ...

    def zero_grad(self):
        ...


class SGD(_SGD):
    def __init__(self, *args, lr=0.1, **kwargs):
        super().__init__(*args, lr=lr, **kwargs)
        self.lr = lr


OPTIMIZERS = {
    "SGD": SGD,
}


def get(parameters, name: str = None, **kwargs) -> Optimizer:
    try:
        optimizer_cls = OPTIMIZERS[name]
    except KeyError:
        raise KeyNotFoundError(name, OPTIMIZERS, "Optimizer")
    return optimizer_cls(parameters, **kwargs)
