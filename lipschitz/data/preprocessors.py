from functools import partial
from typing import Iterable, Callable

import torch

from lipschitz.io_functions.key_not_found_error import KeyNotFoundError

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PreprocessingFunction = Callable[[torch.Tensor], torch.Tensor]


def get(name: str = None, **kwargs) -> PreprocessingFunction:
    """
    Preprocessing function factory.
    Preprocessing is usually applied to a batch of images,
    and to both training and evaluation data.
    """
    if name is None:
        return lambda x: x

    try:
        preprocessor_class = PREPROCESSORS[name]
    except KeyError:
        raise KeyNotFoundError(name, NAMES, "Preprocessor")
    return partial(preprocessor_class, **kwargs)


def center_image(batch, channel_means: Iterable[int]) -> torch.Tensor:
    means_tensor = torch.tensor(channel_means, device=batch.device)
    return batch - means_tensor[None, :, None, None]


def add_gaussian_noise(batch: torch.Tensor, sd: float) -> torch.Tensor:
    noise = torch.randn_like(batch) * sd
    return batch + noise


def convert(batch, dtype: str = "float32", divisor: int = 255):
    dtype = getattr(torch, dtype)
    return batch.to(dtype) / divisor


def convert_and_center(batch: torch.Tensor,
                       channel_means: Iterable[int],
                       dtype: str = "float32",
                       divisor: int = 255) -> torch.Tensor:
    batch = convert(batch, dtype, divisor)
    return center_image(batch, channel_means)


PREPROCESSORS: dict[str, PreprocessingFunction] = {
    "center": center_image,
    "add_gaussian_noise": add_gaussian_noise,
    "convert": convert,
    "convert_and_center": convert_and_center,
}
NAMES = sorted(PREPROCESSORS.keys())


class Concatenate:
    def __init__(self, *preprocessors: PreprocessingFunction):
        self.preprocessors = preprocessors

    def __call__(self, batch: torch.Tensor) -> torch.Tensor:
        for preprocessor in self.preprocessors:
            batch = preprocessor(batch)
        return batch


def get_multiple(all_kwargs: list[dict]) -> PreprocessingFunction:
    return Concatenate(*(get(**kwargs) for kwargs in all_kwargs))
