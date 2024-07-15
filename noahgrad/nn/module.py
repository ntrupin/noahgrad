from typing import Dict, Iterable
from .. import Tensor

class Module:
    """
    neural net base class
    """

    is_training: bool

    def __init__(self, /, is_training: bool = True):
        self.is_training = is_training

    def forward(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError(f"forward not implemented on {type(self)}")

    def train(self):
        for p in self.parameters():
            p.requires_grad = True

    def eval(self):
        for p in self.parameters():
            p.requires_grad = False

    def zero_grad(self):
        for p in self.parameters():
            p.zero_grad()

    def __call__(self, *args, **kwargs) -> Tensor:
        return self.forward(*args, **kwargs)

    def parameters(self, /, only_trainable: bool = False) -> Iterable[Tensor]:
        # params = []
        for _, v in self.__dict__.items():
            if isinstance(v, Tensor) and not v._is_buffer:
                if only_trainable and not v.requires_grad:
                    continue
                # params.append(v)
                yield v
            elif isinstance(v, Module):
                # params.extend(v.parameters())
                for v in v.parameters():
                    yield v
            elif isinstance(v, list):
                for l in v:
                    # params.extend(l.parameters())
                    for p in l.parameters():
                        yield p
        # return list(set(params))

    def buffers(self, /, only_persistent: bool = False) -> Iterable[Tensor]:
        for _, v in self.__dict__.items():
            if isinstance(v, Tensor) and v._is_buffer:
                if only_persistent and not v.persistent:
                    continue
                yield v
        #return list(set(bufs))

    def register_buffer(self, name: str, tensor: Tensor, persistent: bool = True):
        self.__dict__[name] = tensor
        tensor._is_buffer = True
        tensor.persistent = persistent

    def get_param(self, name: str) -> Tensor:
        if name not in self.__dict__ or self.__dict__[name]._is_buffer:
            raise ValueError(f"{name} not in parameters")
        return self.__dict__[name]

    def get_buffer(self, name: str) -> Tensor:
        if name not in self.__dict__ or not self.__dict__[name]._is_buffer:
            raise ValueError(f"{name} not in buffers")
        return self.__dict__[name]

    def state_dict(self) -> Dict[str, Tensor]:
        state = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Tensor) and v.persistent:
                state[k] = v
        return state
