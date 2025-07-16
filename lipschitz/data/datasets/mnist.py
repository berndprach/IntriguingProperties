from functools import partial

import torchvision
from torchvision.transforms import transforms as tfs

from lipschitz.data.typing import TrainEval, TorchDataset
from .dataset import Dataset, split_deterministically

CHANNEL_MEANS = (0.131,)
CLASS_NAMES = [str(i) for i in range(10)]


class MNIST(Dataset):
    training_size = 60_000
    class_names = CLASS_NAMES
    channel_means = CHANNEL_MEANS

    def __init__(self, split_sizes=(55_000, 5_000)):
        super().__init__()
        self.split_sizes = split_sizes

    def get_full_data(self, use_test_data=False) -> TrainEval[TorchDataset]:
        if use_test_data:
            td = get_data(train=True)
            ed = get_data(train=False)
        else:
            all_data = get_data(train=True)
            td, ed = split_deterministically(all_data, self.split_sizes)

        return TrainEval(td, ed)


def get_data(train=True, transform=tfs.ToTensor()) -> TorchDataset:
    get_mnist = partial(
        torchvision.datasets.MNIST,
        root="data",
        train=train,
        transform=transform,
    )
    try:
        return get_mnist(download=False)
    except RuntimeError:
        print("Downloading MNIST dataset to data/.")
        return get_mnist(download=True)
