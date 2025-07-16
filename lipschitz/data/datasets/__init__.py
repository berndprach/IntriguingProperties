
from .cifar10 import CIFAR10
from .dataset import Dataset
from .mnist import MNIST
from .edm_cifar10 import EDMCIFAR10, IntegerEDMCIFAR10
from lipschitz.io_functions.key_not_found_error import KeyNotFoundError

DATASETS = {
    "MNIST": MNIST,
    "CIFAR10": CIFAR10,
    "EDMCIFAR10": EDMCIFAR10,
    "IntegerEDMCIFAR10": IntegerEDMCIFAR10,
}
NAMES = sorted(DATASETS.keys())


def get(name, **kwargs) -> Dataset:
    try:
        ds_cls = DATASETS[name]
    except KeyError:
        raise KeyNotFoundError(name, NAMES, "Dataset")
    return ds_cls(**kwargs)


def channel_means(dataset_name: str):
    d = get(dataset_name)
    try:
        return d.channel_means
    except AttributeError as e:
        print(f"Dataset {dataset_name} does not have channel means defined.")
        raise e



