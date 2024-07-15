import numpy as np
from typing import Optional

from . import functional as F
from . import Module
from .. import Tensor

class CrossEntropyLoss(Module):
    weight: Tensor
    reduction: str

    def __init__(self, weight: Optional[Tensor] = None, reduction: str = "mean"):
        self.weight = weight or Tensor.from_scalar(1)
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = F.softmax(input).log()
        L = -self.weight * input[np.arange(len(input.data)), target.data.astype(int)]
        return self.reduce(L)

    def reduce(self, x: Tensor) -> Tensor:
        if self.reduction == "mean":
            return x.sum() / len(x.data)
        elif self.reduction == "sum":
            return x.sum()
        else:
            return x
