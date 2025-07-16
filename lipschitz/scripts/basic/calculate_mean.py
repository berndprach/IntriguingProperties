import torch

from lipschitz import data
from lipschitz.io_functions.parser import dictionary_str

CIFAR10_kwargs = {"name": "CIFAR-10"}


def calculate_channel_means(dataset_kwargs: str = str(CIFAR10_kwargs)):
    data_kwargs = dictionary_str(dataset_kwargs)
    print(f"Loading dataset with kwargs: {data_kwargs}")

    dl = data.get_data_loader(**data_kwargs)
    print(f"Loaded {len(dl.train)}/{len(dl.eval)} train/eval batches.")
    dl.to_device()

    tensor_sum = None
    number_of_images = 0
    for x_batch, _ in dl.train:
        if tensor_sum is None:
            tensor_sum = torch.zeros_like(x_batch[0], dtype=torch.float32)
        else:
            tensor_sum += x_batch.sum(dim=0)
            number_of_images += x_batch.shape[0]  # To account for final batch.

    tensor_mean = tensor_sum / number_of_images
    channel_means = tensor_mean.mean(dim=(1, 2))
    print(f"Channel means: {channel_means}")
