from typing import Optional

from . import Module
from .. import Tensor

class BCELoss(Module):
    weight: Tensor
    reduction: str

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = "mean"):
        self.weight = weight or Tensor.from_scalar(1)
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.clamp()
        target = target.clamp()
        L = -self.weight * (target * input.log() + (1 - target) * (1 - input).log())
        return self.reduce(L)

    def reduce(self, x: Tensor) -> Tensor:
        if self.reduction == "mean":
            return x.mean()
        elif self.reduction == "sum":
            return x.sum()
        else:
            return x
