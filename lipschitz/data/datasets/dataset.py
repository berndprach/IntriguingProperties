from typing import Optional, Iterable

import numpy as np

from lipschitz.data import typing
from lipschitz.data.typing import TrainEval, DataPartition


class Dataset:
    training_size = None
    class_names = None
    channel_means = None
    root = "data"

    def __init__(self, **kwargs):
        pass

    def download(self):
        raise NotImplementedError

    def get_data(self,
                 use_test_data=False,
                 training_size=None,
                 evaluation_size=None,
                 ) -> TrainEval[DataPartition]:
        """ Overwrite this method or .get_full_data(). """
        full_data = self.get_full_data(use_test_data)
        return subsets(full_data, training_size, evaluation_size)

    def get_full_data(self, use_test_data=False) -> TrainEval[DataPartition]:
        raise NotImplementedError


def head_subset(d: DataPartition, size: Optional[int] = None) -> DataPartition:
    if size is None:
        return d
    size = min(size, len(d))
    return Subset(d, list(range(size)))


def subsets(d: TrainEval[DataPartition],
            training_size=None,
            evaluation_size=None,
            ) -> TrainEval[DataPartition]:
    train_data = head_subset(d.train, training_size)
    eval_data = head_subset(d.eval, evaluation_size)
    return TrainEval(train_data, eval_data)


def split_deterministically(d: DataPartition,
                            split_sizes: Iterable[int],
                            shuffle=True
                            ) -> list[DataPartition]:
    assert sum(split_sizes) <= len(d), "Split sizes exceed dataset size."
    indices = [i for i in range(len(d))]

    np.random.seed(1111)
    if shuffle:
        np.random.shuffle(indices)

    data_partitions: list[DataPartition] = []
    start = 0
    for size in split_sizes:
        subset_indices = indices[start:start+size]
        s = Subset(d, subset_indices)
        data_partitions.append(s)
        start += size

    return data_partitions


class Subset(DataPartition):
    def __init__(self, dataset: DataPartition, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index: int) -> typing.XY:
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.indices)


