import numpy as np
from typing import Sequence, Union

Array = np.ndarray
Scalar = Union[float, int]

class Tensor:

    def __init__(self, data: Union[Array, list, Sequence], /, dtype = None, prev: tuple = (),
                 op: str | None = None, requires_grad: bool = False,
                 label: str | None = None):
        assert isinstance(data, Union[Array, list, Sequence]), "invalid type for Tensor"
        super(Tensor, self).__init__()

        self.dtype = dtype or np.float32
        self.data = (
            np.array(data) if not isinstance(data, Array) else data.astype(dtype=self.dtype))
        self._prev = prev
        self._op = op
        self.requires_grad = requires_grad
        self._label = label

        self.grad = np.zeros_like(self.data, dtype=self.dtype)
        self.grad_fn = None

    def __add__(self, other):
        if isinstance(other, Scalar):
            # scalar addition
            out = Tensor(self.data + other, prev=(self,), op="+", 
                         requires_grad=self.requires_grad)

            if self.requires_grad:
                def _scalar_add_grad():
                    self.grad += out.grad
                out.grad_fn = _scalar_add_grad

            return out

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data + other.data, prev=(self,other), op="+",
                     requires_grad=self.requires_grad or other.requires_grad)

        if not self.requires_grad and not other.requires_grad:
            return out

        if self.shape == other.shape:
            def _add_grad():
                if self.requires_grad:
                    self.grad += out.grad
                if other.requires_grad:
                    other.grad += out.grad
            out.grad_fn = _add_grad
        else:
            laxis, raxis = Tensor._broadcast_axes(self.shape, other.shape)

            def _bcast_add_grad():
                if self.requires_grad:
                    self.grad += np.reshape(
                        np.sum(out.grad, axis=laxis),
                        self.shape)
                if other.requires_grad:
                    other.grad += np.reshape(
                        np.sum(out.grad, axis=raxis),
                        other.shape)
            out.grad_fn = _bcast_add_grad

        return out

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (other * -1)

    def __rsub__(self, other):
        return (self * -1) + other

    def __mul__(self, other):
        if isinstance(other, Scalar):
            out = Tensor(self.data * other, prev=(self,), op="*",
                         requires_grad=self.requires_grad)

            if self.requires_grad:
                def _scalar_mul_grad():
                    self.grad += other * out.grad
                out.grad_fn = _scalar_mul_grad

            return out

        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data * other.data, prev=(self,other), op="*",
                     requires_grad=self.requires_grad or other.requires_grad)

        if not self.requires_grad and not other.requires_grad:
            return out

        if self.shape == other.shape:
            def _mul_grad():
                if self.requires_grad:
                    self.grad += other.data * out.grad
                if other.requires_grad:
                    other.grad += self.data * out.grad
            out.grad_fn = _mul_grad
        else:
            laxis, raxis = Tensor._broadcast_axes(self.shape, other.shape)

            def _bcast_mul_grad():
                if self.requires_grad:
                    self.grad += np.reshape(
                        np.sum(other.data * out.grad, axis=laxis),
                        self.shape)
                if other.requires_grad:
                    other.grad += np.reshape(
                        np.sum(self.data * out.grad, axis=raxis),
                        other.shape)
            out.grad_fn = _bcast_mul_grad

        return out

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return (self ** -1) * other

    def __pow__(self, other):

        def _npow(a, b):
            return 1 / (a ** abs(b)) if b < 0 else a ** b

        out = Tensor(_npow(self.data, other), prev=(self,), op="pow",
                     requires_grad=self.requires_grad)

        if self.requires_grad:
            def _pow_grad():
                self.grad += other * _npow(self.data, other - 1) * out.grad
            out.grad_fn = _pow_grad

        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)

        out = Tensor(self.data @ other.data, prev=(self,other), op="@",
                     requires_grad=self.requires_grad or other.requires_grad,
                     label="MatmulOUT")

        if not self.requires_grad and not other.requires_grad:
            return out

        l_expand = (0,) if self.ndim == 1 else ()
        r_expand = (-1,) if other.ndim == 1 else ()
        expand = l_expand + r_expand

        laxis, raxis = Tensor._broadcast_axes(self.shape[:-2], other.shape[:-2])

        def _matmul_grad():
            if self.requires_grad:
                self.grad += np.reshape(np.sum(
                    np.expand_dims(out.grad, axis=expand) @
                    np.expand_dims(other.data, axis=r_expand).swapaxes(-1, -2),
                    axis=laxis), self.shape)
            if other.requires_grad:
                other.grad += np.reshape(np.sum(
                    np.expand_dims(self.data, axis=l_expand).swapaxes(-1, -2) @
                    np.expand_dims(out.grad, axis=expand),
                    axis=raxis), other.shape)
        out.grad_fn = _matmul_grad

        return out

    def __neg__(self):
        return self * -1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data, dtype=self.dtype)
        for v in reversed(topo):
            if v.grad_fn is not None:
                v.grad_fn()

    def sum(self, /, axis = None, keepdims: bool = False):
        out = Tensor([np.sum(self.data, axis=axis, keepdims=keepdims)],
                     prev=(self,), op="sum", requires_grad=self.requires_grad)

        if self.requires_grad:
            expand = axis if axis and not keepdims else None

            def _sum_grad():
                if expand:
                    self.grad += np.ones_like(self.grad) * np.expand_dims(out.grad, axis=expand)
                else:
                    self.grad += np.ones_like(self.grad) * out.grad
            out.grad_fn = _sum_grad

        return out

    def exp(self):
        out = Tensor(np.exp(self.data), prev=(self,), op="exp",
                     requires_grad=self.requires_grad)

        if self.requires_grad:
            def _exp_grad():
                self.grad += out.data * out.grad
            out.grad_fn = _exp_grad

        return out

    def mean(self, /, axis = None, keepdims: bool = False):
        out = self.sum(axis=axis, keepdims=keepdims)
        return out * (np.prod(out.shape) / np.prod(self.shape))

    def clip(self, min: float = 1e-7, max: float = 1 - 1e-7):
        self.data = np.clip(self.data, min, max)
        return self

    def log(self):
        out = Tensor(np.log(self.data), prev=(self,), op="log",
                     requires_grad=self.requires_grad)

        if self.requires_grad:
            def _log_grad():
                self.grad += (out.grad / self.data)
            out.grad_fn = _log_grad

        return out

    def softmax(self, /, dim: int = 0):
        return self.exp() / self.exp().sum(axis=dim)

    def T(self, axes: Union[tuple[int], list[int], None] = None):
        out = Tensor(np.transpose(self.data, axes=axes), prev=(self,),
                     op="T", requires_grad=self.requires_grad)

        if self.requires_grad:
            def _t_grad():
                self.grad += np.transpose(out.grad, axes=axes)
            out.grad_fn = _t_grad

        return out

    @staticmethod
    def _broadcast_axes(a, b):
        len_a, len_b = len(a), len(b)
        max_len = max(len_a, len_b)

        a = (1,) * (max_len - len_a) + a
        b = (1,) * (max_len - len_b) + b

        laxis = []
        raxis = []

        for i in range(max_len):
            if a[i] != b[i]:
                if a[i] == 1: laxis.append(i)
                if b[i] == 1: raxis.append(i)

        return tuple(laxis), tuple(raxis)

    @property
    def shape(self) -> tuple:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def ndim(self) -> int:
        return self.data.ndim

    def item(self):
        if self.size != 1:
            raise RuntimeError("invalid tensor for item")
        return self.data.sum()

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"
