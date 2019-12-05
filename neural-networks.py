import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

###NEURAL NETWORKS####
# STEP1: Define the network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # inherit all the variables defined in parent class
        self.conv1 = nn.Conv2d(1, 6, 3)  # 1 sample, set to 6 3*3 filters output
        self.conv2 = nn.Conv2d(6, 16, 3)  # 1 input filters from last layer, set to 16 3*3 filters output
        self.fc1 = nn.Linear(16 * 6 * 6,
                             120)  # 6*6 is the H and W of the result picture after 2 conv, set to 120 length
        self.fc2 = nn.Linear(120, 84)  # 120 input length of linear, 84 output
        self.fc3 = nn.Linear(84, 10)  # 10 output

    def forward(self, values):  # just have to define the forward , backward one is automatically defined by autograd.
        values = F.max_pool2d(F.relu(self.conv1(values)), 2)  # layer1: convolution, relu, pool
        values = F.max_pool2d(F.relu(self.conv2(values)), 2)  # layer2: convolution, relu, pool
        values = values.view(-1, self.num_flat_features(values))  # flatten: turn to linear, defined below;
        # the size -1 means it is inferred from other dimensions of the torch
        values = F.relu(self.fc1(values))  # layer3: full connection, linear -> relu
        values = F.relu(self.fc2(values))  # layer4: full connection, linear -> relu
        values = self.fc3(values)  # layer5: have activation function such as softmax or sigmoid to classify the outputs
        return values

    def num_flat_features(self, values):  # to get the total features
        size = values.size()[1:]  # get all dimensions except the batch one ([0])
        num_features = 1
        for s in size:
            num_features *= s  # to multiple all features and get the result
        return num_features


net = Net()
print(net)  # get the structure of the NN

# get learnable parameters: net.parameters()
params = list(net.parameters())  # The learnable parameters of a model are returned by net.parameters()
print(len(params))
for i in range(0, 9):
    print(params[i].size())  # for example:  conv1's .weight:torch.Size([6, 1, 3, 3])
    # 6 filters, 1 input, 3*3 for H and W

# Example: Processing inputs: try a random 32x32 input.
input = torch.randn(1, 1, 32, 32)
out = net(input)
print(out)

# Example: Zero the gradient buffers of all parameters and backprops with random gradients
net.zero_grad()  # set the gradient to 0
out.backward(torch.randn(1, 10))  # backprops with random gradients

# Example: for Loss Function and backward
# Loss Function
output = net(input)
target = torch.randn(1, 10)
criterion = nn.MSELoss()
loss = criterion(output, target)
print(loss)
# backward
"""
Now, if you follow loss in the backward direction, using its .grad_fn attribute, 
you will see a graph of computations that looks like this
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
So, when we call loss.backward(), the whole graph is differentiated w.r.t. the loss, and all Tensors in the graph 
that has requires_grad=True will have their .grad Tensor accumulated with the gradient.
For illustration, let us follow a few steps backward:
"""
print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# Backprop
net.zero_grad()  # zeroes gradient buffers of all parameters: else gradients will be accumulated to existing gradients.
loss.backward()

# Update the weights: use torch.optim
optimizer = optim.SGD(net.parameters(), lr=0.01)  # lr: learning rate
# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)  # see STEP1: define a neural network
loss = criterion(output, target)  # STEP2: compute loss
loss.backward()
optimizer.step()    # STEP3: Does the update
