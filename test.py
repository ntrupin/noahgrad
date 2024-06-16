import torch
import numpy as np

from noahgrad import core
from noahgrad import losses

rng = np.random.default_rng(seed=1121)

def test_sum():
    x = rng.random((2, 4))

    x_ng = core.Tensor(x, requires_grad=True)
    x_pt = torch.tensor(x, requires_grad=True)

    y_ng = x_ng.sum()
    y_pt = x_pt.sum()

    y_ng.backward()
    y_pt.backward()

    assert np.allclose(y_ng.data, y_pt.detach().numpy()), \
        "mismatched sum results"
    assert np.allclose(x_ng.grad, x_pt.grad.numpy()), \
        "mismatched sum gradients"

    print("sum passed tests")

def test_max():
    x = rng.random((2, 3))
    axis = 1

    x_ng = core.Tensor(x, requires_grad=True)
    x_pt = torch.tensor(x, requires_grad=True)

    y_ng = x_ng.max(axis=axis)
    y_pt, _ = torch.max(x_pt, dim=axis)

    loss_ng = y_ng.sum()
    loss_pt = y_pt.sum()

    loss_ng.backward()
    loss_pt.backward()

    assert np.allclose(y_ng.data, y_pt.detach().numpy()), \
        f"mismatched max results"
    assert np.allclose(x_ng.grad, x_pt.grad.numpy()), \
        "mismatched max gradients"

    print("max passed tests")

def test_slice():
    x = rng.random((10, 5))
    indices = (np.arange(10), [1, 3, 2, 4, 2, 4, 4, 2, 1, 2])

    x_ng = core.Tensor(x, requires_grad=True)
    x_pt = torch.tensor(x, requires_grad=True)

    y_ng = x_ng[indices[0], indices[1]].sum()
    y_pt = x_pt[indices[0], indices[1]].sum()

    y_ng.backward()
    y_pt.backward()

    assert np.allclose(y_ng.data, y_pt.detach().numpy()), \
        "mismatched slice results"
    assert np.allclose(x_ng.grad, x_pt.grad.numpy()), \
        "mismatched slice gradients"

    print("slice passed tests")

def test_cross_entropy():
    pred = rng.random((256, 10))
    true = rng.integers(0, high=9, size=(256))

    pred_ng = core.Tensor(pred, requires_grad=True)
    true_ng = core.Tensor(true)

    pred_pt = torch.tensor(pred, requires_grad=True)
    true_pt = torch.tensor(true, dtype=torch.long)

    y_ng = losses.cross_entropy(pred_ng, true_ng)
    y_pt = torch.nn.CrossEntropyLoss()(pred_pt, true_pt)

    y_ng.backward()
    y_pt.backward()

    assert np.allclose(y_ng.data, y_pt.detach().numpy()), \
        f"mismatched cross_entropy results ({y_ng.data} != {y_pt.detach().numpy()})"
    assert np.allclose(pred_ng.grad, pred_pt.grad.numpy()), \
        f"mismatched cross_entropy gradients ({pred_ng.grad} != {pred_pt.grad.numpy()})"

    print("crossentropy passed tests")


if __name__ == "__main__":
    test_sum()
    test_max()
    test_slice()
    test_cross_entropy()
