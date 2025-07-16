from collections.abc import Sequence
from typing import Iterable

import numpy as np
import torch


class PCA:
    """
    Maps data matrix X to T = XW, where W is the matrix of eigenvectors XtX.
    Assuming SVD is given as X = U S Vt, then W = V.
    """

    def __init__(self):
        self.mean = None
        self.w = None

    def fit(self, data_matrix: np.ndarray):
        self.mean = np.mean(data_matrix, axis=0)
        u, s, vt = np.linalg.svd(data_matrix - self.mean, full_matrices=False)
        # shapes: x: (n, d), u: (n, n), s: (n,), vt: (d, d)
        self.w = vt.transpose()
        return u, s, vt

    @property
    def components(self):
        if self.w is None:
            raise ValueError("PCA has not been fitted yet!")
        return [self.w[:, i] for i in range(self.w.shape[1])]

    def transform(self, data_matrix, components: Iterable[int]):
        if self.mean is None or self.w is None:
            raise ValueError("PCA has not been fitted yet!")
        return (data_matrix - self.mean) @ self.w[:, components]

    def head_transform(self, data_matrix, components: int):
        return self.transform(data_matrix, range(components))

    def tail_transform(self, data_matrix, components: int):
        """ Remove the first components. """
        return self.transform(data_matrix, range(components, self.w.shape[1]))

    def inverse_transform(self, m: np.ndarray, components: Iterable[int]):
        wt = self.w.transpose()[components, :]
        return m @ wt + self.mean

    def inverse_head_transform(self, m: np.ndarray):
        component_count = m.shape[1]
        return self.inverse_transform(m, range(component_count))

    def reconstruct(self, original_image: np.ndarray, components: list[int]):
        original_shape = original_image.shape
        flat_image = original_image.reshape(1, -1)
        transformed = self.transform(flat_image, components)
        reconstructed = self.inverse_transform(transformed, components)
        return reconstructed.reshape(original_shape)


XY = tuple[torch.Tensor, torch.Tensor]


def get_data_matrix(train_ds: Sequence[XY], ds_size=None):
    if ds_size is None:
        ds_size = len(train_ds)

    xys = [train_ds[i] for i in range(ds_size)]
    xs = [x.numpy() for x, _ in xys]
    flat_xs = [x.flatten() for x in xs]
    return np.stack(flat_xs)
