import noahgrad.core as ng
import noahgrad.losses as losses
import noahgrad.nn as nn
import noahgrad.optim as optim

class XorNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(2, 10)
        self.l2 = nn.Linear(10, 1)

    def forward(self, x):
        x = ng.sigmoid(self.l1(x))
        x = ng.sigmoid(self.l2(x))
        return x

lr = 1e-1

model = XorNet()
optimizer = optim.SGD(model.parameters(), learning_rate=lr)

x = ng.Tensor([
    [0, 0],
    [1, 0],
    [0, 1],
    [1, 1],
])
y = ng.Tensor([
    [0],
    [1],
    [1],
    [0],
])


epochs = 20000

pred = model(x)
print(pred)
model.train()
for epoch in range(epochs):
    pred = model(x)
    loss = losses.bce_loss(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 1000 == 0:
        print(f"epoch [{epoch + 1}/{epochs}], loss: {loss.item():.4f}")
model.eval()
y_pred = model(x)
print("Predicted outputs:")
print(y_pred)
