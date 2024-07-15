import numpy as np

from . import functional as F
from . import Module
from .._tensor import Tensor, Function, OptionalArrayTuple

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)

class Softmax(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.softmax(x)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.sigmoid(x)
