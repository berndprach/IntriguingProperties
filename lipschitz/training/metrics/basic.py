import torch

from lipschitz.training.metrics import Metric


class Accuracy(Metric):
    f = ".1%"

    def __call__(self, prediction_batch, label_batch):
        predictions = prediction_batch.argmax(dim=1)
        if len(label_batch.shape) == 1:
            return torch.eq(predictions, label_batch).float()
        else:  # one-hot or soft labels
            return torch.eq(predictions, label_batch.argmax(dim=1)).float()


class Margin(Metric):
    def __call__(self, prediction_batch, label_batch):
        if len(label_batch.shape) == 1:  #
            label_batch = torch.nn.functional.one_hot(
                label_batch.to(torch.int64),
                num_classes=prediction_batch.shape[-1]
            )
            label_batch = label_batch.to(prediction_batch.dtype)
        true_score = (prediction_batch * label_batch).sum(dim=-1)
        best_other = (prediction_batch - label_batch * 1e6).max(dim=-1)[0]
        return true_score - best_other
