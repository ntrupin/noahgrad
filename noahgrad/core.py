import numpy as np

from typing import Union, List, Tuple, Self, Optional, Callable, Dict, Any, Set

Array = np.ndarray
Scalar = Union[int, float]

class Tensor:
    _rng: np.random.Generator = np.random.default_rng()

    data: Array
    requires_grad: bool
    grad: Array
    _ctx: Optional['Function']
    _op: Optional[str]
    _label: Optional[str]

    def __init__(self, data: Union[Array, List], /,
                 ctx: Optional['Function'] = None,
                 requires_grad: bool = False, dtype = None,
                 op: Optional[str] = None, label: Optional[str] = None):

        # data
        dtype = dtype or np.float32
        self.data = data.astype(dtype) if isinstance(data, Array) \
                    else np.array(data, dtype=dtype)

        # gradients
        self.requires_grad = requires_grad
        self.grad = np.zeros_like(self.data)
        self._ctx = ctx if requires_grad else None

        # debug info
        self._op = op
        self._label = label

    @classmethod
    def from_scalar(cls, x: Scalar, **kwargs) -> Self:
        return cls([x], **kwargs)

    @classmethod
    def zeros(cls, n: int, **kwargs) -> Self:
        return cls(np.zeros(n), **kwargs)

    @classmethod
    def ones(cls, n: int, **kwargs) -> Self:
        return cls(np.ones(n), **kwargs)

    @classmethod
    def random(cls, size: Optional[Union[Tuple[int, ...], int]] = None, /,
               dtype = np.float32, **kwargs):
        rand = Tensor._rng.random(size, dtype=dtype)
        if isinstance(size, Optional[int]):
            return cls.from_scalar(rand, **kwargs)
        return cls(rand, **kwargs)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    def __add__(self, other: Union[Self, Array, Scalar]) -> Self:
        return Add.apply(self, Tensor._ensure_tensor(other))

    def __radd__(self, other: Union[Self, Array, Scalar]) -> Self:
        return self + other

    def __sub__(self, other: Union[Self, Array, Scalar]) -> Self:
        return Sub.apply(self, Tensor._ensure_tensor(other))

    def __rsub__(self, other: Union[Self, Array, Scalar]) -> Self:
        return -self + other

    def __mul__(self, other: Union[Self, Array, Scalar]) -> Self:
        return Mul.apply(self, Tensor._ensure_tensor(other))

    def __rmul__(self, other: Union[Self, Array, Scalar]) -> Self:
        return self * other

    def __truediv__(self, other: Union[Self, Array, Scalar]) -> Self:
        return self * (other ** -1)

    def __rtruediv__(self, other: Union[Self, Array, Scalar]) -> Self:
        return (self ** -1) * other

    def __neg__(self) -> Self:
        return -1 * self

    def __pow__(self, other: Scalar) -> Self: 
        return Pow.apply(self, Tensor._ensure_tensor(other))

    def __matmul__(self, other: Union[Self, Array, Scalar]) -> Self:
        return MatMul.apply(self, Tensor._ensure_tensor(other))

    def T(self, axes: Optional[Union[Tuple[int, ...], List[int]]] = None) -> Self:
        return Transpose.apply(self, axes=axes)

    def exp(self) -> Self:
        return Exp.apply(self)

    def log(self) -> Self:
        return Log.apply(self)

    def sum(self, /, **kwargs) -> Self:
        return Sum.apply(self, **kwargs)

    def mean(self, /, **kwargs) -> Self:
        out = self.sum(**kwargs)
        return out * (np.prod(out.shape) / np.prod(self.shape))

    @classmethod
    def _ensure_tensor(cls, x: Union[Self, Array, Scalar]) -> Self:
        if isinstance(x, Array): return cls(x, op="coerce")
        elif isinstance(x, Scalar): return cls.from_scalar(x, op="coerce")
        else: return x

    def item(self) -> Scalar:
        return np.sum(self.data)

    def _undo_broadcast(self, tensor: Self, grad: Array) -> Array:
        # tinytorch

        data = tensor.data
        print(data, grad)

        while len(data.shape) != len(grad.shape):
            grad = grad.sum(axis=0, keepdims=(len(grad.shape) == 1))

        for idx, (s1, s2) in enumerate(zip(data.shape, grad.shape)):
            if s1 < s2:
                grad = grad.sum(axis=idx, keepdims=True)

        return grad

    def backward(self):
        if self._ctx is None:
            return

        # thanks tinytorch
        def _toposort(node: Tensor, visited: Set[Tensor],
                      topo: List[Tensor]) -> List[Tensor]:
            if node in visited:
                return topo
            visited.add(node)
            if node._ctx is None:
                topo.append(node)
                return topo
            for child in node._ctx.args:
                _toposort(child, visited, topo)
            topo.append(node)
            return topo

        nodes = reversed(_toposort(self, set(), []))
        #self.grad = np.ones_like(self.data, dtype=self.dtype)
        for node in nodes:
            if node._ctx is None:
                continue
            grads = node._ctx.backward(node)
            if len(node._ctx.args) == 1:
                grads = [grads]

            for tensor, grad in zip(node._ctx.args, grads):
                if grad is None:
                    continue
                grad = self._undo_broadcast(tensor, grad)
                if tensor.grad is None:
                    tensor.grad = np.zeros_like(self.data, dtype=np.float32)
                tensor.grad += grad

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

class Function:
    tensors_need_grad: List[bool]
    requires_grad: bool
    args: Tuple[Tensor, ...]
    kwargs: Dict[str, Any]

    def __init__(self, *tensors: Tensor, **kwargs):
        self.tensors_need_grad = [t.requires_grad for t in tensors]
        self.requires_grad = any(self.tensors_need_grad)
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
        return Tensor(x.data + y.data, op="add")

    def backward(self, out: Tensor) -> Tuple[Array, Array]:
        return out.grad, out.grad

class Sub(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data - y.data, op="sub")

    def backward(self, out: Tensor) -> Tuple[Array, Array]:
        return out.grad, -out.grad

class Mul(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data * y.data, op="mul")

    def backward(self, out: Tensor) -> Tuple[Array, Array]:
        x, y = self.args
        return y.data * out.grad, x.data * out.grad

class Pow(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data ** y.item(), op="pow")

    def backward(self, out: Tensor) -> Array:
        x, y = self.args
        return y.data * x.data ** (y.data - 1) * out.grad

class MatMul(Function):
    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return Tensor(x.data @ y.data, op="matmul")

    def backward(self, out: Tensor) -> Tuple[Array, Array]:
        x, y = self.args
        return out.grad @ y.T().data, x.T().data @ out.grad

class Transpose(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.transpose(x.data, axes=self.kwargs["axes"]), 
                      op="transpose")

    def backward(self, out: Tensor):
        x = self.args[0]
        return np.transpose(out.grad, axes=self.kwargs["axes"])

class Exp(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.exp(x.data), op="exp")

    def backward(self, out: Tensor):
        # d/dL e^x = e^x * L' = L * L'
        return out.data * out.grad

class Log(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.log(x.data), op="log")

    def backward(self, out: Tensor):
        x = self.args[0]
        # d/dL log(x) = L' / x
        return out.grad / x.data

class Sum(Function):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor(np.sum(x.data, **self.kwargs, keepdims=True), op="sum")

    def backward(self, out: Tensor):
        x = self.args[0]
        return np.broadcast_to(out.grad, x.shape)
