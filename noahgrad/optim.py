import numpy as np

from noahgrad.core import Tensor

class Optimizer:
    def __init__(self, parameters: list[Tensor], learning_rate: float):
        self.parameters = parameters
        self.lr = learning_rate

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros(p.shape)

class SGD(Optimizer):
    def __init__(self, parameters: list[Tensor], learning_rate: float):
        super().__init__(parameters, learning_rate)

    def step(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr
