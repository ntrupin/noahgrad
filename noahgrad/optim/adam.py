import numpy as np
from typing import Iterable, Tuple

from . import Optimizer
from .. import Tensor

class Adam(Optimizer):
    def __init__(self, parameters: Iterable[Tensor], lr: float = 1e-3,
                 betas: Tuple[float, float] = (0.9, 0.999), eps: float = 1e-8):
        super().__init__(parameters, lr)
        self.betas = betas
        self.eps = eps
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.parameters]
        self.v = [np.zeros_like(p.data) for p in self.parameters]

    def step(self):
        self.t += 1
        for (i, p) in enumerate(self.parameters):
            if p.grad is None or not p.requires_grad:
                continue

            self.m[i] = self.betas[0] * self.m[i] + \
                    (1 - self.betas[0]) * p.grad
            self.v[i] = self.betas[1] * self.v[i] + \
                    (1 - self.betas[1]) * p.grad ** 2
            mhat = self.m[i] / (1 - self.betas[0] ** self.t)
            vhat = self.v[i] / (1 - self.betas[1] ** self.t)
            p.data -= self.lr * (mhat / (np.sqrt(vhat) + self.eps))
