import numpy as np

from .. import Tensor
from .._tensor import _ReLU

def relu(x: Tensor) -> Tensor:
    return _ReLU.apply(x)

def softmax(x: Tensor, dim: int = -1) -> Tensor:
    max = Tensor(np.max(x.data, axis=dim, keepdims=True))
    expr = (x - max).exp()
    return expr / expr.sum(dim=dim, keepdims=True)

def sigmoid(x: Tensor) -> Tensor:
    return 1 / ((-x + 1).exp())
