from functools import partial
from typing import Any

from torch import nn

from lipschitz.io_functions.configuration import Configuration
from lipschitz.models import layers
from lipschitz.models.layers import typing as t
from lipschitz.models.layers.initializer import INITIALIZER

Linear = layers.AOLLinear
Activation = layers.MaxMin


DEFAULTS_YAML = f"""
width: {32 * 32 * 3}
depth: 8
number_of_outputs: 10
linear:
    name: AOLLinear
head: null
activation:
    name: MaxMin
width_reduction:
    name: FirstChannels
"""

DEFAULTS = Configuration.from_yaml(DEFAULTS_YAML)


def get_lipschitz_mlp(**updates: dict[str, Any]) -> nn.Sequential:
    c = DEFAULTS.copy().update(*updates.items())
    c = c.copy()  # to avoid modifying the keyword arguments ("updates").

    if "initializer" in c["linear"]:
        c["linear"]["initializer"] = INITIALIZER[c["linear"]["initializer"]]
    if c["head"] is not None and "initializer" in c["head"]:
        c["head"]["initializer"] = INITIALIZER[c["head"]["initializer"]]

    get_linear = partial(layers.load, **c["linear"])
    if c["head"] is None:
        get_head = get_linear
    else:
        get_head = partial(layers.load, **c["head"])

    get_activation = partial(layers.load, **c["activation"])
    get_width_reduction = partial(layers.load, **c["width_reduction"])

    return construct_general_mlp(
        width=c["width"],
        depth=c["depth"],
        number_of_outputs=c["number_of_outputs"],
        linear=get_linear,
        activation=get_activation,
        head=get_head,
        width_reduction=get_width_reduction,
    )


def construct_general_mlp(width: int,
                          depth: int,
                          number_of_outputs: int,
                          linear: t.LinearFactory,
                          activation: t.ActivationFactory,
                          head: t.LinearFactory,
                          width_reduction: t.ChannelledFactory,
                          ) -> nn.Sequential:
    model = nn.Sequential()
    append = model.append
    append(nn.Flatten())
    append(layers.ZeroChannelConcatenation(width))

    for i in range(depth-1):
        append(linear(width, width))
        append(activation())

    append(head(width, width))
    append(width_reduction(number_of_outputs))
    return model
