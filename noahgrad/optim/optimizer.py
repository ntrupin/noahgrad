from typing import Iterable, List

from .. import Tensor

class Optimizer:
    parameters: List[Tensor]

    def __init__(self, parameters: Iterable[Tensor], lr: float):
        self.parameters = [p for p in parameters]
        self.lr = lr

    def step(self):
        raise NotImplementedError(f"step not implemented on {type(self)}")

    def zero_grad(self):
        for p in self.parameters:
            p.zero_grad()
