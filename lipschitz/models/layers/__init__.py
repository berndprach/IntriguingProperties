
from torch.nn import ReLU, Identity

from .activations.max_min import MaxMin
from .activations.n_activation import NActivation

from .basic.first_channels import FirstChannels
from .basic.simple_conv import SimpleConv2d
from .basic.simple_linear import SimpleLinear
from .basic.zero_channel_concatenation import ZeroChannelConcatenation

from .lipschitz.aol import AOLConv2d, AOLLinear
from .lipschitz.cpl import CPLConv2d, CPLLinear


def load(*args, name: str = None, **kwargs):
    """
    Usage e.g.: load("AOLLinear", 128, 64, bias=True)
    Using a keyword argument for "name" eases the use of "partial".
    """
    try:
        return globals()[name](*args, **kwargs)
    except KeyError:
        raise ValueError(f"Layer \"{name}\" not defined.")
