from functools import partial

from torch import nn

from lipschitz.io_functions.configuration import Configuration
from lipschitz.models import layers
from lipschitz.models.layers import typing as t

DEFAULTS_YAML = f"""
base_width: 64
number_of_classes: 10
kernel_size: [3, 3]
number_of_blocks: 5
block_depth: 3
number_of_linear_layers: 1
conv:
    name: AOLConv2d
activation:
    name: MaxMin
linear:
    name: AOLLinear
"""

DEFAULTS = Configuration.from_yaml(DEFAULTS_YAML)


def get_lcn(**updates: dict[str, any]) -> nn.Sequential:
    c = DEFAULTS.copy().update(*updates.items())

    c["conv"] = partial(layers.load, **c["conv"])
    c["linear"] = partial(layers.load, **c["linear"])
    c["activation"] = partial(layers.load, **c["activation"])

    return construct_lcn(**c.as_dict)


def construct_lcn(conv: t.ConvFactory,
                  activation: t.ActivationFactory,
                  linear: t.LinearFactory,
                  **hp,
                  ) -> nn.Sequential:

    def convolution(c_in: int, c_out: int, ks=hp["kernel_size"]):
        return nn.Sequential(
            conv(c_in, c_out, kernel_size=ks),
            activation(),
        )

    def down_sampling(c_in: int, c_out: int):
        return nn.Sequential(
            layers.FirstChannels(c_out // 4),
            nn.PixelUnshuffle(2),
        )

    backbone = nn.Sequential()
    c = hp["base_width"]
    for i in range(hp["number_of_blocks"]):
        for _ in range(hp["block_depth"]):
            # Note that ks=(3, 3) also in the last block,
            # so circular padding is illegal!
            backbone.append(convolution(c, c))
        backbone.append(down_sampling(c, 2 * c))
        c *= 2

    s = 32 // 2 ** hp["number_of_blocks"]

    return nn.Sequential(
        layers.ZeroChannelConcatenation(hp["base_width"]),
        convolution(hp["base_width"], hp["base_width"], ks=(1, 1)),

        backbone,

        nn.MaxPool2d(s),
        nn.Flatten(),

        *(linear(c, c) for _ in range(hp["number_of_linear_layers"])),

        layers.FirstChannels(hp["number_of_classes"]),
    )


aol_kwargs = {"conv": {"name": "AOLConv2d"}, "linear": {"name": "AOLLinear"}}
get_aol_lcn = partial(get_lcn, **aol_kwargs)

cpl_kwargs = {"conv": {"name": "CPLConv2d"}, "linear": {"name": "CPLLinear"}}
get_cpl_lcn = partial(get_lcn, **cpl_kwargs)
