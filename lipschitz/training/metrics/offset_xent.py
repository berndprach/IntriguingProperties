import math
import torch

from .metric import Metric


class OffsetCrossEntropy(Metric):
    def __init__(self, offset=math.sqrt(2), temperature=1/4, **kwargs):
        super().__init__()
        self.offset = offset
        self.temperature = temperature
        self.xent = torch.nn.CrossEntropyLoss(**kwargs)

    def __call__(self, score_batch, label_batch):
        label_batch = to_one_hot(label_batch, score_batch.shape[-1])
        offset_scores = score_batch - self.offset * label_batch
        offset_scores /= self.temperature
        return self.xent(offset_scores, label_batch) * self.temperature

    def __repr__(self):
        return f"OX({self.offset:.2g}, {self.temperature:.2g})"


def to_one_hot(label_batch, num_classes, dtype=torch.float32):
    label_batch = torch.nn.functional.one_hot(
        label_batch.to(torch.int64),
        num_classes=num_classes,
    )
    label_batch = label_batch.to(dtype)
    return label_batch
