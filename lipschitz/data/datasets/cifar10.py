from functools import partial
from typing import Optional

import torchvision
from torchvision.transforms import transforms as tfs
from torchvision.transforms.functional import pil_to_tensor

from lipschitz.data.typing import TrainEval, TorchDataset, DataPartition
from .dataset import Dataset, split_deterministically

CHANNEL_MEANS = (0.491, 0.482, 0.447)
CENTER = tfs.Normalize(CHANNEL_MEANS, (1., 1., 1.))
CLASS_NAMES = ["plane", "car", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


TRANSFORMS = {
    "None": None,
    "Center": tfs.Compose([tfs.ToTensor(), CENTER]),
    "ToTensor": tfs.ToTensor(),  # PIL | nd.array -> float [.0, 1.] tensor
    "tensor": pil_to_tensor,
}


class CIFAR10(Dataset):
    training_size = 50_000
    class_names = CLASS_NAMES
    channel_means = CHANNEL_MEANS

    def __init__(self, split_sizes=(45_000, 5_000), transform="ToTensor"):
        super().__init__()
        self.split_sizes = split_sizes
        self.transform = TRANSFORMS[transform]

    def get_full_data(self, use_test_data=False) -> TrainEval[TorchDataset]:
        if use_test_data:
            td = get_data(train=True, transform=self.transform)
            ed = get_data(train=False, transform=self.transform)
        else:
            all_data = get_data(train=True, transform=self.transform)
            td, ed = split_deterministically(all_data, self.split_sizes)

        return TrainEval(td, ed)


def get_data(train, transform: Optional) -> DataPartition:
    get_cifar = partial(
        torchvision.datasets.CIFAR10,
        root="data",
        train=train,
        transform=transform,
    )
    try:
        return get_cifar(download=False)
    except RuntimeError:
        print("Downloading CIFAR-10 dataset to data/.")
        return get_cifar(download=True)
