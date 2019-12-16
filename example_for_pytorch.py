### NumPy: cannot utilize GPUs to accelerate its numerical computations###
import numpy as np

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

w1 = np.random.randn(D_in, H)  # weight for layer 1 (hidden layer)
w2 = np.random.randn(H, D_out)  # weight for layer 2 (output layer)

learning_rate = 1e-6
for t in range(500):
    h = x.dot(w1)  # z1 = x * w1
    h_relu = np.maximum(h, 0)  # a1 = max(z1, 0)
    y_pred = h_relu.dot(w2)  # z2 = a1 * w2

    loss = np.square(y_pred - y).sum()
    # print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)  # dz2 = 2*(z2-y)
    grad_w2 = h_relu.T.dot(grad_y_pred)  # dw2 = a1.T * dz2
    grad_h_relu = grad_y_pred.dot(w2.T)  # da1 = dz2 * w2.T
    grad_h = grad_h_relu.copy()  # dz1 = da1
    grad_h[h < 0] = 0  # dz1 = max(dz1, 0)
    grad_w1 = x.T.dot(grad_h)  # dw1 = x.T * dz1

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

### Torch ###
import torch

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)  # , device=device, dtype=dtype)
y = torch.randn(N, D_out)  # , device=device, dtype=dtype)

w1 = torch.randn(D_in, H)  # , device=device, dtype=dtype)
w2 = torch.randn(H, D_out)  # , device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    h = x.mm(w1)  # make multiplication, z1 = w1 * x
    h_relu = h.clamp(min=0)  # torch.clamp(input, min, max, out=None), make value in (min, max), a1 = max(0, z1)
    y_pred = h_relu.mm(w2)  # z2 = w2 * a1
    loss = (y_pred - y).pow(2).sum().item()  # loss = (y_pred - y)^2.sum()
    if t % 100 == 99:
        print(t, loss)

    grad_y_pred = 2.0 * (y_pred - y)  # dz2 = 2 * (z2 - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)  # dw2 = da1.T * dz2
    grad_h_relu = grad_y_pred.mm(w2.t())  # da1 = dz2 * w2.T
    grad_h = grad_h_relu.clone()  # dz1 = da1
    grad_h[h < 0] = 0  # dz1 = max(dz1, 0)
    grad_w1 = x.t().mm(grad_h)  # dw1 = x.T * dz1

    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

### PyTorch: Tensors and autograd (can custom) ###
# use automatic differentiation to automate the computation of backward passes in neural networks. (autograd package)
import torch

dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    loss = (y_pred - y).pow(2).sum()

    if t % 100 == 99:
        print(t, loss.item())

    loss.backward()

    with torch.no_grad():  # Wrap in torch.no_grad() because weights have requires_grad=True
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        w1.grad.zero_()  # Manually zero the gradients
        w2.grad.zero_()


### PyTorch: nn (can custom) ###
import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# A sequential container, used to set the structure of model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-4
for t in range(500):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

### PyTorch: optim ###
import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# A sequential container, used to set the structure of model
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MSELoss(reduction="sum")

learning_rate = 1e-4
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for t in range(500):
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

### PyTorch: Control Flow + Weight Sharing ###
import random
import torch


class DynamicNet(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(DynamicNet, self).__init__()
        self.input_linear = torch.nn.Linear(D_in, H)
        self.middle_linear = torch.nn.Linear(H, H)
        self.output_linear = torch.nn.Linear(H, D_out)

    def forward(self, x):
        h_relu = self.input_linear(x).clamp(min=0)
        for _ in range(random.randint(0, 3)):  # get a random int in [0, 3]
            h_relu = self.middle_linear(h_relu).clamp(min=0)
        y_pred = self.output_linear(h_relu)
        return y_pred


N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = DynamicNet(D_in, H, D_out)

criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for t in range(500):
    y_pred = model(x)
    loss = criterion(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
