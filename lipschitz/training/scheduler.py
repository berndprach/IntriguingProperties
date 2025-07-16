from typing import Protocol

from torch.optim.lr_scheduler import OneCycleLR as _OneCycleLR

from .optimizer import Optimizer
from ..io_functions.key_not_found_error import KeyNotFoundError


class Scheduler(Protocol):
    def __init__(self, optimizer: Optimizer, total_steps: int):
        ...

    def step(self) -> None:
        ...


class OneCycleLR(_OneCycleLR):
    def __init__(self, optimizer: Optimizer, total_steps: int):
        super().__init__(optimizer, optimizer.lr, total_steps=total_steps)

    def step(self, epoch=None) -> None:
        super().step()


SCHEDULERS = {
    "OneCycleLR": OneCycleLR,
}


def get(optimizer: Optimizer, total_steps, name, **kwargs) -> Scheduler:
    try:
        scheduler_cls = SCHEDULERS[name]
    except KeyError:
        raise KeyNotFoundError(name, SCHEDULERS, "Scheduler")
    return scheduler_cls(optimizer, total_steps, **kwargs)
