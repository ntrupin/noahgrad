import numpy as np

from noahgrad.core import Tensor

rng = np.random.default_rng()

class Module:
    def parameters(self):
        params = []
        for _, v in self.__dict__.items():
            if isinstance(v, Tensor):
                params.append(v)
            if isinstance(v, Module):
                params.extend(v.parameters())
            if isinstance(v, ModuleList):
                for module in v:
                    params.extend(module.parameters())
        return list(set(params))

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def eval(self):
        for p in self.parameters():
            p.requires_grad = False

    def train(self):
        for p in self.parameters():
            p.requires_grad = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

class ModuleList(Module, list):
    def __init__(self, modules=None):
        super(ModuleList, self).__init__()
        if modules is not None:
            self += modules

    def append(self, module: Module):
        super(ModuleList, self).append(module)

    def __setitem__(self, i, module):
        return super().__setitem__(i, module)

    def parameters(self):
        params = []
        for module in self:
            params.extend(module.parameters())
        return params

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, /, bias: bool = True):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = Tensor(
            rng.random((out_dim, in_dim)) / np.sqrt(in_dim + out_dim),
            label="LinearW")
        self.b = Tensor(np.zeros(out_dim), label="LinearB") if bias else None

    def forward(self, x):
        x = x @ self.w.T()
        if self.b:
            x += self.b
        return x

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = Tensor(np.maximum(0, x.data), prev=(x,), op="relu",
                     requires_grad=x.requires_grad)

        if x.requires_grad:
            def _relu_grad():
                x.grad += np.where(x.data > 0, 1, 0) * out.grad
            out.grad_fn = _relu_grad()

        return out

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        out = Tensor(np.tanh(x.data), prev=(x,), op="tanh",
                     requires_grad=x.requires_grad)

        if x.requires_grad:
            def _tanh_grad():
                x.grad += (1 - (np.tanh(x.data) ** 2)) * out.grad
            out.grad_fn = _tanh_grad

        return out

def tanh(x):
    return Tanh().forward(x)

def sigmoid(x):
    interm = (1 + (-1 * x).exp())
    interm._label = "FLAG"
    out = 1 / interm
    return out
