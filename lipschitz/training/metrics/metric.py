from torch import Tensor


class Metric:
    f = ".2f"

    def __call__(self, scores: Tensor, labels: Tensor) -> Tensor:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    def format_value(self, value: float) -> str:
        return f"{value:{self.f}}"
