import torch

from lipschitz.io_functions.key_not_found_error import KeyNotFoundError
from . import lipschitz_convolutional_network as lcn
from . import simple_conv_net
from .lipschitz_mlp import get_lipschitz_mlp
from .lipschitz_mlp import get_lipschitz_mlp

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ARCHITECTURES = {
    "SimpleConvNet": simple_conv_net.get_conv_net,
    "LipschitzMLP": get_lipschitz_mlp,
    "AOL-MLP": get_lipschitz_mlp,
    "LCN": lcn.get_lcn,
    "AOL-LCN": lcn.get_aol_lcn,
    "CPL-LCN": lcn.get_cpl_lcn,
}
NAMES = list(ARCHITECTURES.keys())


def load(name, **kwargs):
    try:
        return ARCHITECTURES[name](**kwargs)
    except KeyError:
        raise KeyNotFoundError(name, NAMES, "Model architecture")


def load_on_device(name, device=DEVICE, **kwargs):
    model = load(name, **kwargs)
    model.to(device)
    return model
