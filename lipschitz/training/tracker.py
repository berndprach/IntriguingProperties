
import torch
from lipschitz.training.metrics import Metric


class MetricTracker:
    def __init__(self, metrics: list[Metric]):
        self.metrics = metrics
        self.tracker = ValueTracker()

    def add_batch(self, output, label):
        statistics = [metric(output, label) for metric in self.metrics]
        self.tracker.add(statistics)

    def reset(self):
        self.tracker = ValueTracker()

    def start_epoch(self):
        self.reset()

    def get_averages(self) -> list[float]:
        return self.tracker.get_averages()

    def get_values(self) -> list[list[float]]:
        return self.tracker.get_values()

    @property
    def results_dict(self):
        return {
            repr(metric): average
            for metric, average in zip(self.metrics, self.get_averages())
        }


class ValueTracker:
    def __init__(self):
        self.all_values = None

    def add(self, value_batches: list[torch.Tensor]):
        if self.all_values is None:
            self.all_values = [[] for _ in value_batches]

        for i, value_batch in enumerate(value_batches):
            value = with_batch_dimension(value_batch.detach())
            self.all_values[i].append(value)

    def get_averages(self) -> list[float]:
        return [get_average(values) for values in self.all_values]

    def get_values(self) -> list[list[float]]:
        return [torch.cat(values).tolist() for values in self.all_values]


def get_average(values: list[torch.Tensor]) -> float:
    return torch.cat(values).mean().item()


def with_batch_dimension(value: torch.Tensor) -> torch.Tensor:
    if is_scalar(value):
        value = value[None]
    return value


def is_scalar(value: torch.Tensor) -> bool:
    return len(value.shape) == 0
