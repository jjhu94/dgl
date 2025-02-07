import torch
import torchvision  # the package has data loaders for common datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=str)
    args = parser.parse_args()
    print(args)
    device = torch.device(args if torch.cuda.is_available() else "cpu")

print(device)

start = time.time()

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
                                       download=False, transform=transform)  # only need to download once
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')  # name classes


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
net.to(device)  # use net on GPU

### Define a Loss function and optimizer ####
criterion = nn.CrossEntropyLoss()  # define loss function: classification cross-entropy loss
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # optimizer: use SGD with momentum update the weights
# create an optimizer object, maintain current parameter status and update them based on calculated gradients

### Train the network ###
for epoch in range(2):  # loop times
    running_loss = 0.0
    for i, data in enumerate(trainloader):
        inputs, labels = data[0].to(device), data[1].to(device)  # send the inputs and targets to the GPU AT EVERY STEP
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
class_correct = list(0. for i in range(10))  # what the . mean?
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
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

end = time.time()
print("cost %.5f seconds" % (end-start))
