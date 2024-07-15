import functools
from typing import Any
from . import Tensor

def context_decorator(fn):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        return fn(*args, **kwargs)

    return wrapper

class no_grad:
    """
    gradient context manager. turns off gradient calculations within
    a function or `with` statement.
    """

    def __init__(self):
        self.prev = False

    def __new__(cls, fn=None):
        if fn is None:
            return super().__new__(cls)
        return cls()(fn)

    def __enter__(self):
        self.prev = Tensor._grad_enabled
        Tensor._grad_enabled = False

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any):
        Tensor._grad_enabled = self.prev

    def __call__(self, fn):
        return context_decorator(fn)
