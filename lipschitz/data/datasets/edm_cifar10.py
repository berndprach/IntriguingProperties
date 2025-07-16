import os
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from torch.utils.data import TensorDataset

from lipschitz.data.typing import TrainEval, TorchDataset
from lipschitz.io_functions.formatting.memory import format_memory
from . import cifar10
from .dataset import Dataset

GITHUB = "https://github.com/wzekai99/DM-Improves-AT"
MIL = 1_000_000
BASE_URL = "https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main"
EDM = {
    "1M": {"filename": "1m.npz", "size": 1 * MIL},
    "10M": {"filename": "10m.npz", "size": 10 * MIL},
    "20M": {"parts": 2, "filename": "20m.npz", "size": 20 * MIL},
    "50M": {"parts": 4, "filename": "50m.npz", "size": 50 * MIL},
}

DATA_FOLDER = Path("data", "edm_cifar10")
os.makedirs(DATA_FOLDER, exist_ok=True)


class EDMCIFAR10(Dataset):
    channel_means = cifar10.CHANNEL_MEANS
    class_names = cifar10.CLASS_NAMES

    def __init__(self, version="1M"):
        super().__init__()
        self.version = version
        self.cifar10 = cifar10.CIFAR10()
        self.training_size = EDM[self.version]["size"]

    def get_data(self,
                 use_test_data=False,
                 training_size=None,
                 evaluation_size=None,
                 ) -> TrainEval[TorchDataset]:
        ed = self.cifar10.get_data(use_test_data, 1, evaluation_size).eval
        if training_size is None:
            training_size = self.training_size
        td = load_emd_ds(self.version, training_size, None)
        return TrainEval[TorchDataset](td, ed)

    def download(self):
        data = EDM[self.version]
        if "parts" in data:
            download_multiple(data["parts"], data["filename"])
        else:
            download_data(data["filename"])


def download_data(filename: str):
    # Use command line to download the data:
    # >> wget https://huggingface.co/datasets/P2333/DM-Improves-AT/resolve/main/cifar10/1m.npz
    local_path = DATA_FOLDER / filename
    if not os.path.exists(local_path):
        url = f"{BASE_URL}/cifar10/{filename}"
        print(f"Downloading from {url} to {local_path}...")
        urlretrieve(url, local_path)
        print("Finished.")
    else:
        print(f"File {local_path} already exists, skipping download.")


def download_multiple(parts: int, filename: str):
    for i in range(1, parts + 1):
        part_name = filename.replace(".npz", f"_part{i}.npz")
        download_data(part_name)

    print("Merging parts...")
    merge_parts(filename, parts)


def merge_parts(filename: str, number_of_parts: int):
    data = []
    for i in range(1, number_of_parts + 1):
        # part_name = filename.replace("part4", f"part{i}")
        part_name = filename.replace(".npz", f"_part{i}.npz")
        data.append(np.load(DATA_FOLDER / part_name))

    images = np.concatenate([d["image"] for d in data])
    labels = np.concatenate([d["label"] for d in data])

    np.savez(DATA_FOLDER / filename, image=images, label=labels)
    print("Finished merging.")


def to_float32(x):
    return x.astype(np.float32) / 255.0


def load_edm_integer(version, size=1_000, transform=None) -> TensorDataset:
    return load_emd_ds(
        version, size, transform,
        np_process=lambda x: x,
        dtype=torch.uint8,
    )


def load_emd_ds(version,
                size=1_000,
                transform=None,
                np_process=to_float32,
                dtype=torch.float32,
                ) -> TensorDataset:
    print(f"Loading EDM data, {version}...")
    train_images, train_labels = load_emd_data(size, version)

    train_images = train_images.transpose(0, 3, 1, 2)
    train_images = np_process(train_images)
    print(f"Numpy size:", format_memory(train_images.nbytes))

    train_x_tensor = torch.tensor(train_images, dtype=dtype)
    train_y_tensor = torch.tensor(train_labels)

    if transform is not None:
        train_x_tensor = transform(train_x_tensor)

    torch_size = train_x_tensor.element_size() * train_x_tensor.numel()
    print(f"Torch size:", format_memory(torch_size))

    return TensorDataset(train_x_tensor, train_y_tensor)


def raise_file_not_found_error(path, version):
    text = f"\n\nData file {path} does not exist. Please download it first. "

    if version.get("parts", 1) == 1:
        filenames = [version["filename"]]
    else:
        filenames = [
            version["filename"].replace(".npz", f"_part{i}.npz")
            for i in range(1, version["parts"] + 1)
        ]
    urls = [f"{BASE_URL}/cifar10/{fn}" for fn in filenames]
    text += "\nYou can download it using:\n"
    text += "\n".join(f">> wget {url}" for url in urls)
    text += f"\nAlternatively, you can download it from {GITHUB}.\n\n"
    print(text)
    raise FileNotFoundError(text)


def load_emd_data(size, version):
    path = DATA_FOLDER / EDM[version]["filename"]
    if not path.exists():
        raise_file_not_found_error(path, EDM[version])

    data = np.load(DATA_FOLDER / EDM[version]["filename"])
    images = data["image"]
    labels = data["label"]
    assert images.shape == (EDM[version]["size"], 32, 32, 3)
    assert labels.shape == (EDM[version]["size"],)
    rng = np.random.default_rng(1111)
    indices = rng.choice(EDM[version]["size"], size, replace=False)
    train_images = images[indices]
    train_labels = labels[indices]
    return train_images, train_labels


class IntegerEDMCIFAR10(Dataset):
    class_names = cifar10.CLASS_NAMES

    def __init__(self, version="1M"):
        super().__init__()
        self.version = version
        self.cifar10 = cifar10.CIFAR10(transform="tensor")
        self.training_size = EDM[self.version]["size"]

    def get_data(self,
                 use_test_data=False,
                 training_size=None,
                 evaluation_size=None,
                 ) -> TrainEval[TorchDataset]:
        ed = self.cifar10.get_data(use_test_data, 1, evaluation_size).eval
        if training_size is None:
            training_size = self.training_size
        td = load_edm_integer(self.version, training_size)
        return TrainEval[TorchDataset](td, ed)

    def download(self):
        download_data(EDM[self.version]["filename"])


