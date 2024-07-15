import noahgrad
import noahgrad.nn as nn
from noahgrad.nn import functional as F

import numpy as np

import mnist

class MLP(nn.Module):
    def __init__(self, num_layers: int, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()

        layer_sizes = [input_dim] + [hidden_dim] * num_layers + [output_dim]
        self.layers = [
            nn.Linear(idim, odim)
            for idim, odim in zip(layer_sizes[:-1], layer_sizes[1:])
        ]

    def forward(self, x):
        for l in self.layers[:-1]:
            x = F.relu(l(x))
        return self.layers[-1](x)

num_layers = 2
hidden_dim = 32
num_classes = 10
batch_size = 256
num_epochs = 10
learning_rate = 1e-3

train_images, train_labels, test_images, test_labels = map(
    np.array, mnist.mnist()
)

def batch_iterate(batch_size, X, y):
    perm = np.array(np.random.permutation(y.size))
    for s in range(0, y.size, batch_size):
        ids = perm[s : s + batch_size]
        yield nn.Tensor(X[ids]), nn.Tensor(y[ids])

def evaluate(model, X, y):
    return np.mean(np.argmax(model(nn.Tensor(X)).data, axis=1) == y.data)

model = MLP(num_layers, train_images.shape[-1], hidden_dim, num_classes)
optimizer = noahgrad.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

model.train()
for e in range(num_epochs):
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        pred = model(X)
        loss = loss_fn(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    accuracy = evaluate(model, test_images, test_labels)
    print(f"epoch {e + 1}, accuracy: {accuracy.item():.4f}")
model.eval()
