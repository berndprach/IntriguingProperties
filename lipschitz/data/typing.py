# from collections.abc import Sequence
from typing import TypeVar, Generic, Protocol

import torch
import torchvision

T = TypeVar('T')


class DataLoader(Protocol):
    def __iter__(self) -> iter:
        ...

    def __len__(self) -> int:
        ...


class TrainEval(Generic[T]):
    def __init__(self, train: T, evaluate: T):
        self.train: T = train
        self.eval: T = evaluate


class Sequence(Generic[T]):
    def __getitem__(self, item) -> T:
        ...

    def __len__(self) -> int:
        ...


XY = tuple[torch.Tensor, torch.Tensor]
DataPartition = Sequence[XY]
TorchDataset = torchvision.datasets.VisionDataset

