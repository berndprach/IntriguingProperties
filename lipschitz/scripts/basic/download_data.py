from lipschitz import data
from lipschitz.io_functions.parser import dictionary_str


CIFAR10_kwargs = {"name": "CIFAR-10"}


def download_data(dataset_kwargs: str = "{'name': 'CIFAR-10'}"):
    data_kwargs = dictionary_str(dataset_kwargs)
    print(f"Loading dataset with kwargs: {data_kwargs}")

    ds = data.get_dataset(**data_kwargs)
    ds.download()
