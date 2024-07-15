"""
tensor.py
Noah Trupin

tensor module for noahgrad. supports myriad tensor operations,
autodiff, and backpropagation. provides a no_grad context manager.
"""

from typing import Any, List, Optional, Self, Set, Tuple, Union
import numpy as np

ArrayLike = Union[np.ndarray, List]
Scalar = Union[int, float, np.integer, np.floating]

DEFAULT_MIN = 1e-6
DEFAULT_MAX = 1 - DEFAULT_MIN

class Tensor:
    _rng: np.random.Generator = np.random.default_rng()
    _grad_enabled: bool = True

    data: np.ndarray
    grad: np.ndarray
    _ctx: Optional["Function"]
    requires_grad: bool

    _is_buffer: bool
    persistent: bool

    def __init__(self, data: ArrayLike, dtype: np.dtype | None = None,
                 requires_grad: bool = False):
        dtype = dtype or (data.dtype if isinstance(data, np.ndarray) else np.dtype(np.float32))

        self.data = data.astype(dtype) if isinstance(data, np.ndarray) \
            else np.array(data, dtype=dtype)

        self.grad = np.zeros_like(self.data, dtype=dtype)
        self.requires_grad = requires_grad

        self._ctx = None
        self._is_buffer = False
        self.persistent = True

    @classmethod
    def from_scalar(cls, x: Scalar, **kwargs) -> Self:
        return cls([x], **kwargs)

    @classmethod
    def uniform(cls, loc: Union[float, ArrayLike] = 0.0, scale: Union[float, ArrayLike] = 1.0, \
        size: Optional[Union[int, Tuple[int, ...]]] = None, **kwargs) -> Self:
        data = Tensor._rng.uniform(loc, scale, size=size)
        return cls(data, **kwargs)

    @classmethod
    def random(cls, size: Optional[Union[int, Tuple[int, ...]]] = None, **kwargs) -> Self:
        data = Tensor._rng.random(size)
        if size is None or isinstance(data, float):
            data = np.array(data, dtype=float)
        return cls(data, **kwargs)

    @classmethod
    def zeros(cls, shape: Union[int, Tuple[int, ...]], **kwargs) -> Self:
        data = np.zeros(shape)
        return cls(data, **kwargs)

    @classmethod
    def _ensure_tensor(cls, x: Union[Self, ArrayLike, Scalar], **kwargs) -> Self:
        if isinstance(x, ArrayLike): return cls(x, **kwargs)
        elif isinstance(x, Scalar): return cls.from_scalar(x, **kwargs)
        else: return x

    @property
    def shape(self) -> Tuple[int, ...]:
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

    def __add__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return Add.apply(self, Tensor._ensure_tensor(y))

    def __radd__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return self + y

    def __sub__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return Sub.apply(self, Tensor._ensure_tensor(y))

    def __rsub__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return -self + y

    def __mul__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return Mul.apply(self, Tensor._ensure_tensor(y))

    def __rmul__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return self * y

    def __truediv__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return Div.apply(self, Tensor._ensure_tensor(y))

    def __rtruediv__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return (self ** -1) * y

    def __neg__(self) -> Self:
        return -1 * self

    def __pow__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return Pow.apply(self, Tensor._ensure_tensor(y))

    def __matmul__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return MatMul.apply(self, Tensor._ensure_tensor(y))

    def __rmatmul__(self, y: Union[Self, ArrayLike, Scalar]) -> Self:
        return MatMul.apply(Tensor._ensure_tensor(y), self)

    def __getitem__(self, args: Any) -> Self:
        return Slice.apply(self, args=args)

    def __setitem__(self, args: Any, y: Union[Self, ArrayLike, Scalar]):
        y = Tensor._ensure_tensor(y)
        self.data[args] = y.data.astype(self.dtype).copy()
        self.grad[args] = y.grad.astype(self.dtype).copy()

    def T(self, *dim: int):
        return Transpose.apply(self, dim=dim)

    def transpose(self, *dim: int):
        return Transpose.apply(self, dim=dim)

    def exp(self) -> Self:
        return Exp.apply(self)

    def log(self) -> Self:
        return Log.apply(self)

    def sum(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Self:
        return Sum.apply(self, dim=dim, keepdims=keepdims)

    def mean(self, dim: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False) -> Self:
        out = self.sum(dim=dim, keepdims=keepdims)
        return out * (np.prod(out.shape) / np.prod(self.shape))

    def clamp(self, min: float = DEFAULT_MIN, max: float = DEFAULT_MAX) -> Self:
        self.data = np.minimum(np.maximum(self.data, min), max)
        return self

    def broadcast_to(self, shape: Tuple[int, ...]) -> Self:
        return Broadcast.apply(self, shape=shape)

    def item(self) -> Scalar:
        return np.sum(self.data)

    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)

    @staticmethod
    def _unbroadcast(grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        while len(grad.shape) > len(shape):
            grad = grad.sum(axis=0)
        for i, dim in enumerate(shape):
            if dim == 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad

    def backward(self):
        if not Tensor._grad_enabled or not self.requires_grad or self._ctx is None:
            return

        def _toposort(node: Self, visited: Set[Self], ordering: List[Self]) -> List[Self]:
            if node in visited:
                # exhausted graph
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
                grad = Tensor._unbroadcast(grad, arg.grad.shape)
                arg.grad += grad

    def __repr__(self):
        return f"Tensor([{self.data}]{', requires_grad' if self.requires_grad else ''})"

OptionalArrayTuple = Tuple[Optional[np.ndarray], ...]

class Function:
    args: Tuple[Tensor, ...]

    def __init__(self, *tensors: Tensor, **kwargs):
        self.requires_grad = any([t.requires_grad for t in tensors])
        self.args = tuple(tensors) if self.requires_grad else ()
        self.kwargs = kwargs

        #if len(self.args) == 2:
        #    # handle broadcasting for binop
        #    x, y = self.args
        #    shape = np.broadcast_shapes(x.shape, y.shape)
        #    self.args = (x.broadcast_to(shape), y.broadcast_to(shape))

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(f"forward not implemented on {type(self)}")

    def backward(self, *args, **kwargs) -> Tuple[Optional[np.ndarray], ...]:
        raise NotImplementedError(f"backward not implemented on {type(self)}")

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    @classmethod
    def apply(cls, *args, **kwargs) -> Tensor:
        ctx = cls(*args, **kwargs)
        out = ctx.forward(*args)
        if ctx.requires_grad:
            out.requires_grad = True
            out._ctx = ctx
        return out

class Broadcast(Function):
    def forward(self, x: Tensor) -> Tensor:
        shape = self.kwargs["shape"]
        data = np.broadcast_to(x.data, shape)
        return Tensor(data)

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        shape = self.kwargs["shape"]
        x = self.args[0]
        return np.sum(out.grad, axis=Broadcast.get_axes(x.shape, shape)[0]), None

    @staticmethod
    def get_axes(left: Tuple[int, ...], right: Tuple[int, ...]) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        ldim = len(left)
        rdim = len(right)
        maxdim = max(ldim, rdim)

        lshape = (1, ) * (maxdim - ldim) + left
        rshape = (1, ) * (maxdim - rdim) + right

        laxes, raxes = [], []

        for i in range(len(lshape)):
            if lshape[i] > rshape[i]:
                raxes.append(i)
            elif rshape[i] > lshape[i]:
                laxes.append(i)

        return tuple(laxes), tuple(raxes)

class Add(Function):
    """
    addition

    L = x + y
    dL = 1, 1
    """
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data + y.data)

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        # sum rule
        return out.grad, out.grad

class Sub(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data - y.data)

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        # sum rule
        return out.grad, -1 * out.grad

class Mul(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data * y.data)

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        x, y = self.args
        # product rule
        return out.grad * y.data, x.data * out.grad

class Div(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data / y.data)

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        x, y = self.args
        # quotient rule
        return (out.grad * y.data) / (y.data ** 2), -1 * (x.data * out.grad) / (y.data ** 2)

class Pow(Function):
    """
    L = x^y
    dL = y x^(y - 1) x', x^y log(x) y'
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data ** y.data)

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        x, y = self.args
        return y.data * x.data ** (y.data - 1) * out.grad, \
               out.data * np.log(x.data) * out.grad

class MatMul(Function):
    """
    L = x y
    dL = L y^T, x^T L
    """

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data @ y.data)

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        x, y = self.args

        # if lhs is vector, prepend an empty dimension
        laxis = (0,) if x.ndim == 1 else ()

        # if rhs is vector, append an empty dimension
        raxis = (-1,) if y.ndim == 1 else ()

        # expand along new axes and transpose
        xval = np.expand_dims(x.data, axis=laxis).swapaxes(-1, -2)
        yval = np.expand_dims(y.data, axis=raxis).swapaxes(-1, -2)
        grad = np.expand_dims(out.grad, axis=laxis+raxis)

        # get tensor broadcast shapes (for summing gradient along common axes)
        l, r = Broadcast.get_axes(x.shape[:-2], y.shape[:-2])

        return np.reshape(np.sum(grad @ yval, axis=l), x.shape), \
               np.reshape(np.sum(xval @ grad, axis=r), y.shape)

class Transpose(Function):
    def forward(self, x: Tensor) -> Tensor:
        dim = self.kwargs["dim"]
        return Tensor(x.data.transpose(*dim))

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        dim = self.kwargs["dim"]
        return out.grad.transpose(*dim), None

class Exp(Function):
    """
    L = e^x
    L = e^x x'
    """

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.exp(x.data))

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        return out.data * out.grad, None

class Log(Function):
    """
    L = log(x)
    dL = x' / x
    """

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.log(x.data))

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        x = self.args[0]
        return out.grad / x.data, None

class Sum(Function):
    def forward(self, x: Tensor) -> Tensor:
        dim, keepdims = self.kwargs["dim"], self.kwargs["keepdims"]
        return Tensor(np.sum(x.data, axis=dim, keepdims=keepdims))

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        dim, keepdims = self.kwargs["dim"], self.kwargs["keepdims"]
        x = self.args[0]
        if dim and not keepdims:
            return np.ones_like(x.grad) * np.expand_dims(out.grad, axis=dim), None
        else:
            return np.ones_like(x.grad) * out.grad, None

class Slice(Function):
    def forward(self, x: Tensor) -> Tensor:
        args = self.kwargs["args"]
        return Tensor(x.data[args])

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        x, args = self.args[0], self.kwargs["args"]
        grad = np.zeros_like(x.data)
        grad[args] = out.grad
        return grad, None

class _ReLU(Function):
    """
    rectified linear unit

    L = x if x > 0 else 0
    dL = 1 if x > 0 else 0
    """

    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.maximum(x.data, 0))

    def backward(self, out: Tensor) -> OptionalArrayTuple:
        x = self.args[0]
        return np.where(x.data > 0, 1, 0) * out.grad, None
