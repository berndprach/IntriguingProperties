import torch

from lipschitz import models
from lipschitz.io_functions.parser import dictionary_str

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_parameters(model_kwarg_str: str = "{'name': 'SimpleConvNet'}"):
    mkw = dictionary_str(model_kwarg_str)
    print(f"Model: {mkw}")

    model = models.load_on_device(**mkw)
    model(torch.zeros(1, 3, 32, 32).to(DEVICE))

    print(f"Trainable parameters: {_count_model_parameters(model):_}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):_}")
