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
    weights: Tensor
    bias: Optional[Tensor]

    def __init__(self, in_dim: int, out_dim: int, /, bias: bool = True):
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.weights = Tensor.random((out_dim, in_dim), op="weights") / np.sqrt(in_dim + out_dim)
        self.bias = Tensor.zeros(out_dim, op="bias") if bias else None

    def forward(self, x: Tensor) -> Tensor:
        x = x @ self.weights.T()
        if self.bias:
            x += self.bias
        return x

def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + (-1 * x).exp())
