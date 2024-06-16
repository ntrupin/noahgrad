import numpy as np
from noahgrad.core import Tensor, one_hot, softmax

def mean_square_error(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return ((y_pred - y_true) ** 2).sum() / y_pred.size

def binary_cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    return (-y_true * y_pred.log() - (1 - y_true) * (1 - y_pred).log()).mean()

def cross_entropy(y_pred: Tensor, y_true: Tensor) -> Tensor:
    #y_true = one_hot(y_true, y_pred.shape[-1])
    probs = softmax(y_pred, axis=1)
    log_probs = -(probs[np.arange(probs.shape[0]), y_true.data.astype(int)]).log()
    return log_probs.mean().sum()
