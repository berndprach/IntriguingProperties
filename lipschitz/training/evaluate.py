from functools import partial
from typing import Protocol, Any

import torch


from lipschitz.data.typing import TrainEval, DataLoader
from .metrics import LIST_OF_METRICS
from .tracker import MetricTracker
from lipschitz.io_functions.key_not_found_error import KeyNotFoundError
from ..algorithms import randomized_smoothing

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model: torch.nn.Module,
             dl: TrainEval[DataLoader],
             name: str = "Accuracy",
             partitions: list[str] = ("train", "eval"),
             **kwargs) -> dict[str, Any]:
    try:
        evaluation_function = evaluators[name]
    except KeyError:
        raise KeyNotFoundError(name, evaluators, "Evaluation function")

    evaluation_function = partial(evaluation_function, **kwargs)

    model.eval()
    model = torch.no_grad()(model)

    result = {}
    for partition in partitions:
        loader = getattr(dl, partition)
        result[partition] = evaluation_function(model, loader)
    return result


class EvaluateFunction(Protocol):
    def __call__(self, m: torch.nn.Module, dl: DataLoader) -> dict[str, Any]:
        """
        Assumption:
        - model is in eval mode (model.eval())
        - no_grad context is applied (model = torch.no_grad()(model))
        - data loader yield batches on device (dl = dl.to_device(DEVICE))
        """


def evaluate_metrics(model, data_loader, metrics_name: str = "Accuracy"):
    """ Requires model.eval() and no_grad() context. """
    tracker = MetricTracker(LIST_OF_METRICS[metrics_name])
    for x_batch, y_batch in data_loader:
        predictions = model(x_batch)
        tracker.add_batch(predictions, y_batch)
    return tracker.results_dict


def noisy_evaluate(model, data_loader, metrics_name="Accuracy", sigma=1., n=5):
    """ Requires model.eval() and no_grad() context. """
    tracker = MetricTracker(LIST_OF_METRICS[metrics_name])
    for x_batch, y_batch in data_loader:
        x_batch = x_batch.repeat((n, 1, 1, 1))
        noise = torch.randn_like(x_batch) * sigma
        predictions = model(x_batch + noise)  # n*b x 10
        predictions = predictions.reshape(n, -1, predictions.shape[-1])
        predictions = predictions.mean(dim=0)
        tracker.add_batch(predictions, y_batch)
    return tracker.results_dict


evaluators: dict[str, EvaluateFunction] = {  # Extended in other files.
    "Metrics": evaluate_metrics,
    "Noisy": noisy_evaluate,
    "Lipschitz": partial(evaluate_metrics, metrics_name="Lipschitz"),
    "Accuracy": partial(evaluate_metrics, metrics_name="Accuracy"),
    "RandomizedSmoothing": randomized_smoothing.evaluate_cra,
}


# def add_evaluator(name: str, f: EvaluateFunction):
#     if name in evaluators:
#         raise KeyError(f"Evaluator \"{name}\" already exists.")
#     evaluators[name] = f

