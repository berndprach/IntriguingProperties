from torch import nn, Tensor
from torch.nn.functional import linear

from lipschitz.models.layers.train_val_cache_decorator import train_val_cached
from lipschitz.models.layers.basic.simple_linear import SimpleLinear


def bjorck_bowie_orthonormalize(w, beta=0.5, iterations=20, order=1):
    """
    Bjorck, Ake, and Clazett Bowie. "An iterative algorithm for computing
    the best estimate of an orthogonal matrix."
    SIAM Journal on Numerical Analysis 8.2 (1971): 358-364.
    """

    assert order == 1, "only first order Bjorck is supported"

    if w.shape[-2] < w.shape[-1]:
        w_tp = w.transpose(-1, -2)
        w_tp_orth = bjorck_bowie_orthonormalize(w_tp, beta, iterations, order)
        return w_tp_orth.transpose(-1, -2)

    for _ in range(iterations):
        wt_w = w.transpose(-1, -2) @ w
        w = (1 + beta) * w - beta * w @ wt_w
    return w


class BnBLinear(SimpleLinear):
    def __init__(self,
                 *args,
                 initializer=nn.init.eye_,
                 train_iterations=3,
                 val_iterations=100,
                 **kwargs):
        super().__init__(*args, initializer=initializer, **kwargs)
        self.train_iterations = train_iterations
        self.val_iterations = val_iterations

    def forward(self, x: Tensor) -> Tensor:
        weight = self.get_weight()
        return linear(x, weight, self.bias)

    @train_val_cached
    def get_weight(self):
        iterations = self.get_iterations(self.training)
        return bjorck_bowie_orthonormalize(self.weight, iterations)

    def get_iterations(self, training: bool):
        if training:
            return self.train_iterations
        else:
            return self.val_iterations
