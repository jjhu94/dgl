
#####import#####
from __future__ import print_function
import torch
import numpy as np

####start####
# start
x = torch.rand(5, 3)
print(x)
x = x.new_ones(5,3,dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)  # get the same shape
print(x)
print(x[:, 1])  # all rows and 2nd column
# reshape
y = torch.rand(5, 3)
print("y=", y)
print("y.size=", y.size())
x = y.view(15)  # reshape
print(x)
print(x.size())
x = y.view(15, 1)
print(x)
x = y.view(1, 15)
print(x)
x = y.view(-1, 3)  # the size -1 is inferred from other dimensions
print(x.size())
# calculation
y.add_(x)  # addition

# exchange between NumPy and Tensor
# Tensor to NumPy
a = torch.ones(5)
b = a.numpy()
# NumPy to Tensor
a = np.ones(5)
b = torch.from_numpy(a)


####autograd package#####
# Tensor
x = torch.ones(2, 2, requires_grad=True)  # Create a tensor and set requires_grad=True to track computation with it
a = torch.randn(2, 2)
print(a.requires_grad)  # requires_grad defaults to False if not given
a.requires_grad_(True)
print(a.requires_grad)
a.requires_grad_(False)  # turn requires_grad back to False
print(a.requires_grad)
with torch.no_grad():  # also works
    print((a ** 2).requires_grad)  # stop autograd from tracking history on Tensors

# Gradients
# calculate directly
y = x + 2
z = y * y * 3
out = z.mean()
out.backward()
print(x.grad)  # Print gradients d(out)/dx
# calculate by provide the vector to backward as argument
x = torch.randn(3, requires_grad=True)
y = x * 2
while y.data.norm() < 1000:  # the L2 norm (a.k.a Euclidean norm) of the tensor
    y = y * 2
print(y)
v = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)  # provide the vector: how much it changes
y.backward(v)
print(x.grad)
