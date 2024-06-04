import numpy as np

from noahgrad.core import Tensor

def mse_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return ((y_pred - y_true) ** 2).sum() / y_pred.size

def bce_loss(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return (-y_true * y_pred.log() - (1 - y_true) * (1 - y_pred).log()).mean()

def cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    idx = y_true.data.astype(int)
    onehot = np.zeros((idx.shape[0], y_pred.shape[1]))
    onehot[np.arange(len(onehot)), idx] = 1
    onehot = Tensor(onehot, prev=(y_true,), requires_grad=False)
    return (-y_pred.softmax().log() * onehot).mean()
