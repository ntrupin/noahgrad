import unittest
from typing import Any, List, Optional, Self, Set, Tuple, Union
import numpy as np

Array = np.ndarray
ArrayLike = Union[List, Tuple]
Scalar = Union[int, float]

class Tensor:
    _rng = np.random.Generator = np.random.default_rng()

    def __init__(self, data: Union[Array, ArrayLike], /,
                 requires_grad: bool = False, ctx: Optional['Function'] = None,
                 dtype = np.float32, op: Optional[str] = None,
                 label: Optional[str] = None):
        self.dtype = dtype
        self.data = data.astype(dtype) if isinstance(data, Array) \
                else np.array(data, dtype=dtype)
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data, dtype=dtype)
        self._ctx = ctx
        self._op = op
        self._label = label

    @classmethod
    def from_scalar(cls, x: Scalar, **kwargs) -> Self:
        return Tensor([x], **kwargs)

    @classmethod
    def _ensure_tensor(cls, x: Union[Self, Array, ArrayLike, Scalar],
                       **kwargs) -> Self:
        if isinstance(x, Union[Array, ArrayLike]): return Tensor(x, **kwargs)
        elif isinstance(x, Scalar): return Tensor.from_scalar(x, **kwargs)
        else: return x

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    def __add__(self, y): return Add.apply(self, Tensor._ensure_tensor(y))
    def __radd__(self, y): return self + y

    def backward(self):
        if not self.requires_grad or self._ctx is None:
            return

        def _toposort(node: Self, visited: Set[Self], ordering: List[Self]) -> List[Self]:
            if node in visited:
                return ordering
            visited.add(node)
            if node._ctx is not None:
                for child in node._ctx.args:
                    _toposort(child, visited, ordering)
            ordering.append(node)
            return ordering
        ordering = _toposort(self, set(), [])

        for node in reversed(ordering):
            if not node.requires_grad or node._ctx is None:
                continue

            grads = node._ctx.backward(node)

            for arg, grad in zip(node._ctx.args, grads):
                if grad is None:
                    continue
                if arg.grad is None:
                    arg.grad = np.zeros_like(self.data, dtype=node.dtype)
                arg.grad += grad

    def __repr__(self) -> str:
        return f"Tensor({self.shape}{', requires_grad' if self.requires_grad else ''})"

class Function:
    def __init__(self, *tensors: Tensor, **kwargs):
        self.requires_grad = any([t.requires_grad for t in tensors])
        self.args = tuple(tensors) if self.requires_grad else ()

        self.kwargs = kwargs

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(f"forward not implemented on {type(self)}")

    def backward(self, *args, **kwargs) -> Any:
        raise NotImplementedError(f"backward not implemented on {type(self)}")

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        ctx = cls(*args, **kwargs)
        out = ctx.forward(*args)
        if ctx.requires_grad:
            out.requires_grad = True
            out._ctx = ctx
        return out

class Add(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data + y.data)

    def backward(self, out: Tensor) -> Tuple[np.ndarray, np.ndarray]:
        # d/dL x + y = x' + y'
        return out.grad, out.grad

if __name__ == "__main__":
    a = Tensor([1, 2, 3])
    b = Tensor([[0, 0, 0], [10, 10, 10]])
    c = a + b
    print(c)
