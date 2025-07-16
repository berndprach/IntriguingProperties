from lipschitz import models
from lipschitz.io_functions.parser import dictionary_str


def _count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model_kwarg_str: str = "{'name': 'SimpleConvNet'}"):
    mkw = dictionary_str(model_kwarg_str)
    print(f"Model: {mkw}")

    model = models.load_on_device(**mkw)

    print(f"Trainable parameters: {_count_model_parameters(model):_}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):_}")

    print(f"\nModel architecture:\n{model}")

