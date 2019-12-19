from pathlib import Path
import requests

import pickle
import gzip

from matplotlib import pyplot
import numpy as np

import torch

import math

from IPython.core.debugger import set_trace

import torch.nn.functional as F

from torch import nn
from torch.nn import Module

from torch import optim

from torch.utils.data import TensorDataset

from torch.utils.data import DataLoader



DATA_PATH = Path("data")  # instantiate concrete paths
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)  # Create a new directory at this given path, ignore errors when True

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"
if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:  # must be open as rb to use pickle.load()
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

# pyplot.imshow(x_train[0].reshape((28, 28),), cmap="gray")  # reshape flattened row to 2d image

x_train, y_train, x_valid, y_valid = map(torch.tensor,
                                         (x_train, y_train, x_valid, y_valid))  # turn numpy arrays to tensor
n, c = x_train.shape


### Neural net from scratch ###
weights = torch.randn(784, 10) / math.sqrt(784)  # initialize the weights with Xavier initialisation (1/sqrt(n))
weights.requires_grad_()  # a trailling _ in PyTorch signifies that the operation is performed in-place
bias = torch.zeros(10, requires_grad=True)


def log_softmax(x):
    return x - x.exp().sum(-1).log().unsqueeze(-1)


def model(xb):
    return log_softmax(xb @ weights + bias)


bs = 64  # batch size
xb = x_train[0:bs]
preds = model(xb)
preds[0], preds.shape
print(preds[0], preds.shape)


def nll(input, target):
    return -input[range(target.shape[0]), target].mean()


loss_func = nll

yb = y_train[0:bs]


def accuracy(out, yb):
    preds = torch.argmax(out, dim=1)
    return (preds == yb).float().mean()


lr = 0.5
epochs = 2

for epoch in range(epochs):
    for i in range((n - 1) // bs + 1):  # number of batches
        start_i = i * bs
        end_i = start_i + bs
        xb = x_train[start_i:end_i]
        yb = y_train[start_i:end_i]
        preds = model(xb)
        loss = loss_func(preds, yb)

        loss.backward()
        with torch.no_grad():
            weights -= weights.grad * lr
            bias -= bias.grad * lr
            weights.grad.zero_()
            bias.grad.zero_()
print(loss_func(model(xb), yb), accuracy(model(xb), yb))


### Using torch.nn.functional:function ###
loss_func = F.cross_entropy  # replacing our hand-written activation and loss functions


def model(xb):
    return xb @ weights + bias


print(loss_func(model(xb), yb), accuracy(model(xb), yb))


### Refactor using nn.Module: model and update parameters###
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(784, 10) / math.sqrt(784), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(10), requires_grad=True)

    def forward(self, xb):
        return xb @ self.weights + self.bias


lr = 0.5
epochs = 2


def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):  # number of batches
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            model = Mnist_Logistic()
            preds = model(xb)
            loss = loss_func(preds, yb)

            loss.backward()
            with torch.no_grad():
                for p in model.parameters():
                    p -= p.grad * lr
                model.zero_grad()


### Refactor using nn.Linear ###
class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10)

    def forward(self, xb):
        return self.lin(xb)


print(loss_func(model(xb), yb))


### Refactor using optim ###
def get_model():
    model = Mnist_Logistic()
    return model, optim.SGD(model.parameters(), lr=lr)


model, opt = get_model()
loss_func = F.cross_entropy


def fit():
    for epoch in range(epochs):
        for i in range((n - 1) // bs + 1):  # number of batches
            start_i = i * bs
            end_i = start_i + bs
            xb = x_train[start_i:end_i]
            yb = y_train[start_i:end_i]
            model, opt = get_model()
            preds = model(xb)
            loss = loss_func(preds, yb)

            loss.backward()

            opt.step()
            opt.zero_grad()


fit()
print(loss_func(model(xb), yb))


### Refactor using Dataset and DataLoader and add validation ###
train_ds = TensorDataset(x_train, y_train)
train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True)
# Shuffling the training data is important to prevent correlation between batches and overfitting.
valid_ds = TensorDataset(x_valid, y_valid)
valid_dl = DataLoader(valid_ds, batch_size=bs * 2)  # use a larger batch size and compute the loss more quickly

model, opt = get_model()

for epoch in range(epochs):
    model.train()  # always call model.train() before training, and model.eval() before inference
    # these are used by layers such as nn.BatchNorm2d and nn.Dropout to ensure appropriate behaviour for these phases.
    for xb, yb in train_dl:
        pred = model(xb)
        loss = loss_func(pred, yb)

        loss.backward()
        opt.step()
        opt.zero_grad()

    model.eval()
    with torch.no_grad():
        valid_loss = sum(loss_func(model(xb), yb) for xb, yb in valid_dl)

    print(epoch, valid_loss / len(valid_dl))


### Create fit() and get_data() ###
def loss_batch(model, loss_func, xb, yb, opt=None):  # computes the loss for one batch
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):  # use zip() along with the unpacking operator * to unzip
    for epoch in range(epochs):
        model.train()
        for xb, yb in train_dl:
            loss_batch(model, loss_func, xb, yb, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(*[loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])

        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print(epoch, val_loss)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True),
        DataLoader(valid_ds, batch_size=bs * 2)  # use a larger batch size and compute the loss more quickly
    )


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
model, opt = get_model()
fit(epochs, model, loss_func, opt, train_dl, valid_dl)


### Switch to CNN ###
class Mnist_CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1)

    def forward(self, xb):
        xb = xb.view(-1, 1, 28, 28)
        xb = F.relu(self.conv1(xb))
        xb = F.relu(self.conv2(xb))
        xb = F.relu(self.conv3(xb))
        xb = F.avg_pool2d(xb, 4)  # xb.size() == torch.Size([64, 10, 1, 1])
        return xb.view(-1, xb.size(1))


lr = 0.1
epochs = 2

train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
loss_func = F.cross_entropy

model = Mnist_CNN()
opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)


### nn.Sequential ###
class Lambda(nn.Module):  # create a view layer for the network
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)


def preprocess(x):
    return x.view(-1, 1, 28, 28)


# A Sequential object runs each of the modules contained within it, in a sequential manner.
model = nn.Sequential(
    Lambda(preprocess),
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AvgPool2d(4),
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)


### Wrapping DataLoader: to work with any 2d single channel image ###
def preprocess(x, y):  # x.size() == torch.Size([64, 784])
    return x.view(-1, 1, 28, 28), y  # x.size() == torch.Size([64, 1, 28, 28]) n, c, h, w


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


train_dl, valid_dl = get_data(train_ds, valid_ds, bs)
train_dl = WrappedDataLoader(train_dl, preprocess)
valid_dl = WrappedDataLoader(valid_dl, preprocess)

model = nn.Sequential(
    nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(16, 10, kernel_size=3, stride=2, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),  # define the size of the output tensor we want
    Lambda(lambda x: x.view(x.size(0), -1)),
)

opt = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

fit(epochs, model, loss_func, opt, train_dl, valid_dl)
