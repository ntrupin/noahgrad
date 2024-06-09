import numpy as np
from typing import List, Tuple

from noahgrad.core import Tensor

class Optimizer:
    def __init__(self, parameters: List[Tensor], learning_rate: float):
        self.parameters = parameters
        self.lr = learning_rate

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for p in self.parameters:
            p.grad = np.zeros(p.shape)

class SGD(Optimizer):
    def __init__(self, parameters: List[Tensor], learning_rate: float):
        super().__init__(parameters, learning_rate)

    def step(self):
        for p in self.parameters:
            p.data -= p.grad * self.lr

class Adam(Optimizer):
    def __init__(self, parameters: List[Tensor], learning_rate: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(parameters, learning_rate)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in parameters]
        self.v = [np.zeros_like(p.data) for p in parameters]

    def step(self):
        self.t += 1
        for (i, p) in enumerate(self.parameters):
            self.m[i] = self.betas[0] * self.m[i] + \
                    (1 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + \
                    (1 - self.betas[1]) * p.grad
            mhat = self.m[i] / (1 - self.betas[0] ** self.t)
            vhat = self.v[i] / (1 - self.betas[1] ** self.t)
            p.data -= self.lr * (mhat / (np.sqrt(vhat) + self.eps))
