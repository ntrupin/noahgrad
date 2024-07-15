from . import Module
from .. import Tensor

class MSELoss(Module):
    reduction: str

    def __init__(self, reduction: str = "mean"):
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        L = (input - target) ** 2
        return self.reduce(L)

    def reduce(self, x: Tensor) -> Tensor:
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        else:
            return x
