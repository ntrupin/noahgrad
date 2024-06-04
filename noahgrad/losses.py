import numpy as np

from noahgrad.core import Tensor

def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return ((y_pred - y_true) ** 2).sum() / y_pred.size

def bce_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return (-y_true * y_pred.log() - (1 - y_true) * (1 - y_pred).log()).mean()
