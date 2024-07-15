from typing import Iterable
from . import Optimizer
from .. import Tensor

class SGD(Optimizer):
    def __init__(self, parameters: Iterable[Tensor], lr: float):
        super().__init__(parameters, lr)

    def step(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr
