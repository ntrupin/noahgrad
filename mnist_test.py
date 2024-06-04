import numpy as np

import noahgrad.core as core
import noahgrad.nn as nn
import noahgrad.losses as losses
import noahgrad.optim as optim

import torch

import mnist

class MLP(nn.Module):
    def __init__(
        self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int
    ):
        super().__init__()

        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.relu = nn.ReLU()

    def forward(self, x):
        for l in self.layers[:-1]:
            x = self.relu(l(x))
        return self.layers[-1](x)

num_layers = 2
hidden_dim = 32
num_classes = 10
batch_size = 256
num_epochs = 10
learning_rate = 1e-1

train_images, train_labels, test_images, test_labels = map(
    np.array, mnist.mnist()
)

def batch_iterate(batch_size, X, y):
    perm = np.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield core.Tensor(X[ids]), core.Tensor(y[ids])

model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
optimizer = optim.SGD(model.parameters(), learning_rate=learning_rate)

for e in range(num_epochs):
    loss = core.Tensor([0])
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        pred = model(X)
        loss = losses.cross_entropy(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"epoch {e + 1}, loss: {loss.item():.4f}")
