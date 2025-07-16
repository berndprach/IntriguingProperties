from math import sqrt


from .metric import Metric
from .basic import Accuracy, Margin
from .cra import CRA
from .offset_xent import OffsetCrossEntropy
from lipschitz.io_functions.key_not_found_error import KeyNotFoundError


METRICS = {
    "Accuracy": Accuracy,
    "Margin": Margin,
    "CRA": CRA,
    "OffsetCrossEntropy": OffsetCrossEntropy,
    "OX": OffsetCrossEntropy,
}


def get(name=None, **kwargs):
    try:
        metric_class = METRICS[name]
    except KeyError:
        raise KeyNotFoundError(name, METRICS, "Metric")

    return metric_class(**kwargs)


LIPSCHITZ_METRICS = [
    Accuracy(),
    Margin(),
    CRA(36 / 255),
    CRA(72 / 255),
    CRA(108 / 255),
    CRA(1.0),
    OffsetCrossEntropy(sqrt(2) * 36 / 255, temperature=1 / 4),
]


LIST_OF_METRICS = {
    "Lipschitz": LIPSCHITZ_METRICS,
    "Accuracy": [Accuracy()],
}
