from collections.abc import Sequence
from functools import partial
from typing import Protocol

import torch
from torch.utils.data import DataLoader as TorchDataLoader

from .typing import TrainEval, TorchDataset, DataLoader, DataPartition
from .typing import TrainEval as TrEv

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Collection(Protocol):
    def __iter__(self) -> iter:
        ...

    def __len__(self) -> int:
        ...


class DataLoader:
    def __init__(self, dl: Collection):
        self.dl: Collection = dl

    @classmethod
    def from_dataset(cls, d: DataPartition, batch_size=256, shuffle=True):
        tdl = TorchDataLoader(d, batch_size=batch_size, shuffle=shuffle)
        return cls(tdl)

    def __iter__(self) -> iter:
        return iter(self.dl)

    def __len__(self) -> int:
        return len(self.dl)

    def map_xy(self, xy_map):
        self.dl = _MappedDataLoader(self.dl, xy_map)

    def map_x(self, x_map):
        def map_function(batch):
            x_batch, y_batch = batch
            return x_map(x_batch), y_batch

        self.map_xy(map_function)

    def to_device(self, device=DEVICE):
        def map_function(batch):
            x_batch, y_batch = batch
            return x_batch.to(device), y_batch.to(device)

        self.map_xy(map_function)

    def repeat(self, epochs: int):
        self.dl = _RepeatDataLoader(self.dl, epochs)

    def cycle(self, batches: int):
        self.dl = _CycleDataLoader(self.dl, batches)


class TrainEvalDataLoader:
    # def __init__(self, d: TrEv[TorchDataset], batch_size=256, shuffle=True):
    def __init__(self, d: TrEv[DataPartition], batch_size=256, shuffle=True):
        get_dl = DataLoader.from_dataset
        self.train = get_dl(d.train, batch_size=batch_size, shuffle=shuffle)
        self.eval = get_dl(d.eval, batch_size=batch_size, shuffle=False)

    def map_xy(self, xy_map):
        self.train.map_xy(xy_map)
        self.eval.map_xy(xy_map)

    def map_x(self, x_map):
        self.train.map_x(x_map)
        self.eval.map_x(x_map)

    def to_device(self, device=DEVICE):
        self.train.to_device(device)
        self.eval.to_device(device)


class _MappedDataLoader:
    def __init__(self, dl: Collection, map_fn):
        self.dl = dl
        self.map_fn = map_fn

    def __iter__(self) -> iter:
        for batch in self.dl:
            yield self.map_fn(batch)

    def __len__(self) -> int:
        return len(self.dl)


class _RepeatDataLoader:
    def __init__(self, dl: Collection, epochs: int = None):
        self.dl = dl
        self.epochs = epochs

    def __iter__(self) -> iter:
        self.epochs_seen = 0
        for _ in range(self.epochs):
            for batch in self.dl:
                yield batch

    def __len__(self) -> int:
        return self.epochs * len(self.dl)


class _CycleDataLoader:
    def __init__(self, dl: Collection, batches: int):
        self.dl = dl
        self.batches = batches
        self.batches_seen = 0

    def __iter__(self) -> iter:
        yield from _cycle(self.dl, amount_of_yields=self.batches)

    def __len__(self) -> int:
        return self.batches


def _infinite_cycle(iterable):
    """ Similar to "itertools.cycle", but uses less memory. """
    while True:
        for x in iterable:
            yield x


def _cycle(iterable, amount_of_yields: int):
    amount_yielded = 0
    for element in _infinite_cycle(iterable):
        if amount_yielded >= amount_of_yields:
            return
        yield element
        amount_yielded += 1
