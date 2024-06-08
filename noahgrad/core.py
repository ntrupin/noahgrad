from typing import Any, List, Optional, Self, Set, Tuple, Union
import numpy as np

Array = np.ndarray
ArrayLike = Union[List, Tuple]
Scalar = Union[int, float]

class Tensor:
    _rng = np.random.Generator = np.random.default_rng()

    data: Array
    requires_grad: bool
    grad: Array
    _ctx: Optional['Function']
    _op: Optional[str]
    _label: Optional[str]

    def __init__(self, data: Union[Array, ArrayLike], /,
                 requires_grad: bool = False, ctx: Optional['Function'] = None,
                 dtype: np.dtype = np.float32, op: Optional[str] = None,
                 label: Optional[str] = None):
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
    def random(cls, shape: Tuple[int, ...], **kwargs) -> Self:
        data = Tensor._rng.random(shape)
        return Tensor(data, **kwargs)

    @classmethod
    def zeros(cls, n: int, **kwargs) -> Self:
        data = np.zeros(n)
        return Tensor(data, **kwargs)

    @classmethod
    def ones(cls, n: int, **kwargs) -> Self:
        data = np.ones(n)
        return Tensor(data, **kwargs)

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

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    def __add__(self, y): return Add.apply(self, Tensor._ensure_tensor(y))
    def __radd__(self, y): return self + y

    def __sub__(self, y): return Sub.apply(self, Tensor._ensure_tensor(y))
    def __rsub__(self, y): return -self + y

    def __mul__(self, y): return Mul.apply(self, Tensor._ensure_tensor(y))
    def __rmul__(self, y): return self * y

    def __truediv__(self, y): return self * (y ** -1)
    def __rtruediv__(self, y): return (self ** -1) * y

    def __pow__(self, y): return Pow.apply(self, Tensor._ensure_tensor(y))

    def __matmul__(self, y): return MatMul.apply(self, Tensor._ensure_tensor(y))
    def __rmatmul__(self, y): return MatMul.apply(Tensor._ensure_tensor(y), self)

    def __neg__(self): return -1 * self

    def T(self, axes: Optional[Tuple[int, ...]] = None):
        return Transpose.apply(self, axes=axes)

    def exp(self): return Exp.apply(self)

    def log(self): return Log.apply(self)

    def sum(self, axis: Optional[Tuple[int, ...]] = None):
        return Sum.apply(self, axis=axis)

    def mean(self, axis: Optional[Tuple[int, ...]] = None):
        out = self.sum(axis=axis)
        return out * (np.prod(out.shape) / np.prod(self.shape))

    @staticmethod
    def _unbroadcast(x: Array, y: Array) -> Array:
        while len(x.shape) != len(y.shape):
            y = y.sum(axis=0, keepdims=(len(y.shape) == 1))

        for idx, (s1, s2) in enumerate(zip(x.shape, y.shape)):
            if s1 < s2:
                y = y.sum(axis=idx, keepdims=True)

        return y

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

        self.grad = np.ones_like(self.data, dtype=self.dtype)

        for node in reversed(ordering):
            if not node.requires_grad or node._ctx is None:
                continue

            grads = node._ctx.backward(node)

            for arg, grad in zip(node._ctx.args, grads):
                if grad is None:
                    continue
                grad = Tensor._unbroadcast(arg.data, grad)
                if arg.grad is None:
                    arg.grad = np.zeros_like(self.data, dtype=node.dtype)
                arg.grad += grad

    def item(self) -> Scalar:
        return np.sum(self.data)

    def __repr__(self) -> str:
        return f"Tensor({self.data}{', requires_grad' if self.requires_grad else ''})"

class Function:
    args: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor, **kwargs):
        self.requires_grad = any([t.requires_grad for t in tensors])
        self.args = tuple(tensors) if self.requires_grad else ()

        self.kwargs = kwargs

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(f"forward not implemented on {type(self)}")

    def backward(self, *args, **kwargs) -> Tuple[Optional[np.ndarray], ...]:
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

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        # d/dL (x + y) = x' + y'
        return out.grad, out.grad

class Sub(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data - y.data)

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        # d/dL (x - y) = x' - y'
        return out.grad, -out.grad

class Mul(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data * y.data)

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        x, y = self.args
        # d/dL xy = x'y + xy'
        return out.grad * y.data, x.data * out.grad

class Pow(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data ** y.data)

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        # d/dL x^y = yx^(y-1)x'
        x, y = self.args
        return y.data * x.data ** (y.data - 1) * out.grad, None

class MatMul(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data @ y.data)

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        # d/dX XY = (XY)' Y^T
        # d/dY XY = X^T (XY)'
        x, y = self.args
        return out.grad @ np.swapaxes(y.data, -1, -2), \
               np.swapaxes(x.data, -1, -2) @ out.grad

class Transpose(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.transpose(x.data, axes=self.kwargs["axes"]))

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        # d/dL X^T = X'^T
        return out.grad.transpose(self.kwargs["axes"]), None

class Exp(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.exp(x.data))

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        # d/dL e^x = e^x x'
        return out.data * out.grad, None

class Log(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.log(x.data))

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        # d/dL log(x) = 1 / x
        (x,) = self.args
        return (1 / x.data) * out.grad, None

class Sum(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.sum(x.data, axis=self.kwargs["axis"], keepdims=True))

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        (x,) = self.args
        return np.broadcast_to(out.grad, x.shape), None

class ReLU(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(x.data, 0))

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        # d/dL max(x, 0) = (1 if x > 0 else 0)
        (x,) = self.args
        return np.where(x.data > 0, 1, 0) * out.grad, None

class Tanh(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.tanh(x.data))

    def backward(self, out: Tensor) -> Tuple[Optional[np.ndarray], ...]:
        # d/dL tanh(x) = 1 - tanh^2(x)
        return (1 - (out.data ** 2)) * out.grad, None

def tanh(x: Tensor) -> Tensor:
    return Tanh.apply(x)

def sigmoid(x: Tensor) -> Tensor:
    return 1 / (1 + (-1 * x).exp())
