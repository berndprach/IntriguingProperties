from typing import Callable

import torch
from torchvision.transforms import transforms as tfs

from lipschitz.data.preprocessors import convert
from lipschitz.io_functions.key_not_found_error import KeyNotFoundError

AugmentationFunction = Callable[[torch.Tensor], torch.Tensor]


def concatenate(*args):
    return tfs.Compose(args)


def augmentation94percent(h=32, w=32, crop_size=4):
    crop = tfs.RandomCrop((h, w), padding=crop_size, padding_mode="reflect")
    flip = tfs.RandomHorizontalFlip()
    erase = tfs.RandomErasing(p=1., scale=(1 / 16, 1 / 16), ratio=(1., 1.))
    return tfs.Compose([crop, flip, erase])


def random_crop(h=32, w=32, padding=4, padding_mode="reflect"):
    return tfs.RandomCrop((h, w), padding=padding, padding_mode=padding_mode)


def mnist_crop(h=28, w=28, padding=2, padding_mode="constant"):
    return tfs.RandomCrop((h, w), padding=padding, padding_mode=padding_mode)


class GaussianNoise:
    def __init__(self, sd):
        self.sd = sd

    def __call__(self, tensor):
        return tensor + torch.randn_like(tensor) * self.sd


def gaussian_94percent(sd=1/8, **kwargs):
    return concatenate(
        augmentation94percent(**kwargs),
        GaussianNoise(sd=sd),
    )


class Convert:
    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return convert(tensor, **self.kwargs)


AUGMENTATION_FUNCTIONS: dict[str, type[AugmentationFunction]] = {
    "None": lambda: tfs.Compose([]),
    "94percent": augmentation94percent,
    "flip": tfs.RandomHorizontalFlip,
    "94percent_gaussian": gaussian_94percent,
    "crop": random_crop,
    "mnist_crop": mnist_crop,
    "con94percent": lambda: tfs.Compose([Convert(), augmentation94percent()]),
}

NAMES = list(AUGMENTATION_FUNCTIONS.keys())
AUGMENTATION = {name: func() for name, func in AUGMENTATION_FUNCTIONS.items()}


def get(name, **kwargs) -> tfs.Compose:
    try:
        aug_cls = AUGMENTATION_FUNCTIONS[name]
    except KeyError:
        raise KeyNotFoundError(name, NAMES, "Augmentation function")
    return aug_cls(**kwargs)
