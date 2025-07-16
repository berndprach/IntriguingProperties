from math import sqrt

import torch
from torch.nn.functional import one_hot

from .metric import Metric


STANDARD_CRA_RADII = [0., 36/255, 72/255, 108/255, 1.]


def cra(predictions, labels, radius, correction=sqrt(2)):
    labels_oh = one_hot(labels, predictions.shape[-1])
    penalized_predictions = predictions - radius * labels_oh * correction
    return torch.eq(penalized_predictions.argmax(dim=1), labels).float()


class CRA(Metric):
    def __init__(self, radius: float, correction=sqrt(2)):
        self.radius = radius
        self.correction = correction
        super().__init__()

    def __call__(self, score_batch, label_batch):
        return cra(score_batch, label_batch, self.radius, self.correction)

    def __repr__(self):
        return f"CRA({self.radius:.2f})"
