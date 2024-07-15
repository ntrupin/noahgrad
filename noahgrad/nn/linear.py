import numpy as np
from typing import Optional

from . import Module
from .. import Tensor

class Linear(Module):
    in_dim: int
    out_dim: int
    w: Tensor
    b: Optional[Tensor]

    def __init__(self, in_dim: int, out_dim: int, /, bias: bool = True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = Tensor.random((out_dim, in_dim)) / np.sqrt(in_dim + out_dim)
        self.b = Tensor.zeros(out_dim) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.w.T()
        if self.b:
            x += self.b
        return x
