
"""
Avoiding Circular Imports:
 - first import typing module
 - import types as >> from lipschitz.data.typing import TrainEval
 - This avoids importing this file ("data.__init__.py") multiple times.
"""
from typing import Sequence

import torch

from . import typing, preprocessors
from . import datasets
from . import data_loader
from .data_loader import TrainEvalDataLoader

from .typing import TrainEval, TorchDataset, XY, DataPartition
from .datasets.dataset import Dataset


def get_dataset(name: str, **kwargs) -> Dataset:
    return datasets.get(name, **kwargs)


def get_data(name: str,
             use_test_data=False,
             training_size=None,
             evaluation_size=None,
             **kwargs,
             ) -> TrainEval[DataPartition]:
    ds = get_dataset(name, **kwargs)
    return ds.get_data(use_test_data, training_size, evaluation_size)


def get_data_loader(name, batch_size=256, **kwargs) -> TrainEvalDataLoader:
    data = get_data(name, **kwargs)
    return TrainEvalDataLoader(data, batch_size=batch_size)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_loader_on_device(name, device=DEVICE, **kwargs) -> TrainEvalDataLoader:
    dl = get_data_loader(name, **kwargs)
    dl.to_device(device)
    return dl


def get_loader(loader_kwargs, preprocessing_kwargs, device=DEVICE):
    dl = get_data_loader(**loader_kwargs)
    print(f"Loaded {len(dl.train)}/{len(dl.eval)} train/eval batches.")
    dl.to_device(device)
    dl.map_x(preprocessors.get(**preprocessing_kwargs))
    return dl


def get_batch(name, **kwargs):
    dl = get_data_loader(name, **kwargs)
    batch, labels = next(iter(dl.eval))
    return batch, labels
