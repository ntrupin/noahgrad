import numpy as np
from typing import List, Optional
from noahgrad.core import Tensor


class Module:
    def parameters(self) -> List[Tensor]:
        params = []
        for _, v in self.__dict__.items():
            if isinstance(v, Tensor):
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
        return list(set(params))

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(f"forward not implemented on {type(self)}")

    def train(self):
        for p in self.parameters():
            p.requires_grad = True

    def eval(self):
        for p in self.parameters():
            p.requires_grad = False

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

class Linear(Module):
    in_dim: int
    out_dim: int
    w: Tensor
    b: Optional[Tensor]

    def __init__(self, in_dim: int, out_dim: int, /, bias: bool = True):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = Tensor.random((out_dim, in_dim)) / np.sqrt(in_dim + out_dim)
        self.b = Tensor.zeros(out_dim) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.w.T()
        if self.b:
            x += self.b
        return x
