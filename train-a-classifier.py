import torch
import torchvision  # the package has data loaders for common datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt  # conda install -n env_name matploylib
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


### Loading and normalizing CIFAR10 ###
# class Compose is used to manage transform, it do loops on input images for all transform actions
transform = transforms.Compose([
    # transforms.Resize((32, 32)),  # used when the image is not 32*32
    transforms.ToTensor(),  # turn data to [0,1] and to Tensor
    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])  # normalize: to [-1.0, 1.0] (by (x-0.5)/0.5)
trainset = torchvision.datasets.CIFAR10(root="./data", train=True,
                                        download=True, transform=transform)
# download train and transform it as set, root means the place to save data
trainloader = torch.utils.data.DataLoader(dataset=trainset, batch_size=4,
                                          shuffle=True, num_workers=2)
# shuffle set to True for trainset and set to False for testset; reshuffle it in each epoch
# num_workers: use subprocesses to read data; batch_size: 4 in 1 batch
testset = torchvision.datasets.CIFAR10(root="./data", train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # name classes


# show some examples
def imshow(img):  # turn Tensor type back to picture
    img = img / 2 + 0.5  # unnormalize: turn back to [0,1]
    npimg = img.numpy()  # Tensor back to NumPy (C, H, W)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    # np.transpose is used to reverse the dimensions, imshow (H, W, C)
    plt.show()


"""
dataiter = iter(trainloader)  # get an iterator using iter()
images, labels = dataiter.next()  # iterate through it using next()
imshow(torchvision.utils.make_grid(images))  # Make a grid of images: Tensor type
print(' '.join('%s' % classes[labels[j]] for j in range(4)))  # print labels
"""


### Define a Convolutional Neural Network ###
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3-channel images, 6 filters with 5*5 H & W
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # the result image with H and W at 5*5
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

### Define a Loss function and optimizer ####
criterion = nn.CrossEntropyLoss()  # define loss function: classification cross-entropy loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # optimizer: use SGD with momentum update the weights
# create an optimizer object, maintain current parameter status and update them based on calculated gradients

### Train the network ###
for epoch in range(1):  # loop times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)  # forward
        loss.backward()  # backward: calculate gradients
        optimizer.step()  # optimize: update all parameters based on calculated gradients
        # simplied version, notice that it is only used once in one mini-batch

        running_loss += loss.item()  # accumulate the loss; .item() is used to get the values of the object
        if i % 2000 == 1999:
            print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 2000))  # print the mean loss
            running_loss = 0  # set to 0 for another loop
print("finished")

# save trained model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)

### Test the network on the test data ###
# examples

dataiter = iter(testloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
print("Truth:", " ".join("%s" % classes[labels[j]] for j in range(4)))
net = Net()
net.load_state_dict(torch.load(PATH))  # reload saved model
outputs = net(images)  # put data into model ,output size(4,10)
_, predicted_label = torch.max(outputs, 1)  # find predicted label for each row; 0 means for each column
print("Predicted:", " ".join("%s" % classes[predicted_label[j]] for j in range(4)))

# on the whole test set
correct = 0
total = 0
with torch.no_grad():  # do not need gradient
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted_label = torch.max(outputs.data, 1)
        total += labels.size(0)  # use 0 to get the value of batchsize
        correct += (predicted_label == labels).sum().item()  # (predicted_label == labels) returns 1 when True
        # use .sum() to sum up all hits, use item() to get the value of Tensor
print('Accuracy of the network on the 10000 test images: %d %%' % (100 * correct / total))
# on the whole test set and show accuracy for each class
class_correct = list(0. for i in range(10))  # what the . mean?
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = net(images)
        _, predicted_label = torch.max(outputs, 1)
        a = (predicted_label == labels)
        c = (predicted_label == labels).squeeze()  # delect all dimensions with only 1 object
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()  # plus 0 or 1 for the specific class
            class_total[label] += 1
for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))


### Training on GPU ###
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
net.to(device)  # use net on GPU
inputs, labels = data[0].to(device), data[1].to(device)  # send the inputs and targets to the GPU AT EVERY STEP




