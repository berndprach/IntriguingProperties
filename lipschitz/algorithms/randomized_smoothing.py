from collections import Counter
from statistics import NormalDist
from typing import Callable

import numpy as np
import torch
from torch import tensor
from torch.nn.functional import one_hot

from lipschitz.data.typing import DataLoader

# https://arxiv.org/abs/1902.02918

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate_cra(model: Callable[[tensor], tensor],  # assumes eval % no_grad.
                 data_loader: DataLoader,  # assumes on device.
                 sample_size=1_000,
                 sigma=0.5,
                 targets=(0., 36 / 255, 72 / 255, 108 / 255, 1.),
                 ) -> dict[str, float]:
    results = {}
    train_radii = approximate_radii(model, data_loader, sample_size, sigma)
    for target in targets:
        cra = sum(r >= target for r in train_radii) / len(train_radii)
        results[f"CRA({target:.2f})"] = cra
    results["Margin()"] = sum(train_radii) / len(train_radii)
    return results


def un_batch(data_loader: DataLoader):
    for x_batch, y_batch in data_loader:
        for x, y in zip(x_batch, y_batch):
            yield x, y


def approximate_radii(model, data_loader, sample_size, sigma) -> list[float]:
    radii = []
    for x, y in un_batch(data_loader):
        r = approximate_robustness_radius(model, sample_size, sigma, x, y)
        radii.append(r)
    return radii


def approximate_robustness_radius(model, num_samples, sigma, x, y):
    x = x[None, :, :, :].repeat((num_samples, 1, 1, 1))
    noise = torch.randn_like(x) * sigma
    predictions = model(x + noise).argmax(1)

    counts = Counter(predictions.cpu().numpy())
    p1 = counts[y.item()] / num_samples
    c2 = max((v for c, v in counts.items() if c != y.item()), default=0)
    p2 = c2 / num_samples
    p1 = clip(p1, 0.01, 0.99)
    p2 = clip(p2, 0.01, 0.99)
    radius = sigma / 2 * (inv_cdf(p1) - inv_cdf(p2))

    return radius


def clip(x, lower, upper):
    return max(lower, min(upper, x))


inv_cdf = NormalDist().inv_cdf


class Smooth(torch.nn.Module):
    def __init__(self, model, sigma, num_samples, clip_at=(0.01, 0.99)):
        super().__init__()
        self.model = model  # image -> class scores
        self.sigma = sigma
        self.num_samples = num_samples
        self.clip_at = clip_at

    def forward(self, x):  # image -> class scores
        """ Asure >> with torch.no_grad() to protect memory."""
        x = x.repeat((self.num_samples, 1, 1, 1))
        noise = torch.randn_like(x) * self.sigma
        class_scores = self.model(x + noise)
        class_count = class_scores.shape[1]
        predictions = class_scores.argmax(1)
        predictions = predictions.reshape(self.num_samples, -1)
        one_hot_predictions = one_hot(predictions, num_classes=class_count)
        scores = one_hot_predictions.float().mean(dim=0)  # (b, class_count)
        return scores

    def lipschitz_scores(self, x) -> np.ndarray:
        scores = self(x)
        scores = scores.cpu().numpy()
        scores = np.clip(scores, *self.clip_at)
        # # New:
        # scores = scores / scores.sum(axis=1, keepdims=True)
        # scores = np.array([inv_cdf(s) for s in scores])
        scores = np.vectorize(inv_cdf)(scores)
        return scores * self.sigma
