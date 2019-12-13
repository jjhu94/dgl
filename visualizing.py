# visualize by TensorBoard
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# transforms
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])  # normalize: to [-1.0, 1.0] (by (x-0.5)/0.5)

# datasets
trainset = torchvision.datasets.FashionMNIST("./data",
                                             download=True,
                                             train=True,
                                             transform=transform
                                             )
testset = torchvision.datasets.FashionMNIST("./data",
                                            download=False,
                                            train=False,
                                            transform=transform
                                            )

# dataloaders
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=4)

# constant for classes
classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')


# helper function to show an image
def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)  # dim=0 means get mean for each column
    img = img / 2 + 0.5
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.show(np.transpose(npimg, (1, 2, 0)))


# define networks
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*4*4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*4*4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()  # Loss Function
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)  # Update the weights


### TensorBoard setup ###
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/fashion_mnist_experiment_1")  # creates a runs/fashion_mnist_experiment_1 folder


### Writing to TensorBoard ###
dataiter = iter(trainloader)
images, labels = dataiter.next()

img_grid = torchvision.utils.make_grid(images)

matplotlib_imshow(img_grid, one_channel=True)

writer.add_image('four_fashion_mnist_images', img_grid)  # write single image data to tensorboard

# run: tensorboard --logdir=runs from the command line and then navigating to https://localhost:6006 to show results

### Inspect the model using TensorBoard ###
writer.add_graph(net, images)  # to visualize a network
writer.close()
plt.close()


### Adding a “Projector” to TensorBoard ###
def select_n_random(data, labels, n=100):

    assert len(data) == len(labels)

    perm = torch.randperm(len(data))  # return a tensor with int for 0 to n-1 in a random order
    return data[perm][:n], labels[perm][:n]  # data[perm].size() = torch.Size([60000, 28, 28]), get first n in perm


images, labels = select_n_random(trainset.data, trainset.targets)

classlabels = [classes[lab] for lab in labels]

features = images.view(-1, 28*28)
writer.add_embedding(features,  # visualize the lower dimensional representation of higher dimensional data
                     metadata=classlabels,
                     label_img=images.unsqueeze(1))  # add a dimension
writer.close()


### Tracking model training with TensorBoard ###
def images_to_probs(net, images):
    output = net(images)
    _, preds_tensor = torch.max(output, 1)  # return max value and its corresponding index
    preds = np.squeeze(preds_tensor.numpy())  # predicted labels
    return preds, [F.softmax(el, dim=0) for i, el in zip(preds, output)]  # softmax for each image
    # softmax() returns probabilities for the given matrix


def plot_classes_preds(net, images, labels):
    preds, probs = images_to_probs(net, images)
    fig = plt.figure(figsize=(12, 48))  # create a blank figure
    for idx in np.arange(4):  # like range()
        ax = fig.add_subplot(1, 4, idx+1, xticks=[], yticks=[])  # xticks: labels on x axis
        matplotlib_imshow(images[idx], one_channel=True)  # show pictures in grey
        ax.set_title("{0}, {1:.1f}%\n(label:{2})".format(
            classes[preds[idx]],
            probs[idx][preds[idx]] * 100.0,
            classes[labels[idx]]),
            color=("green" if preds[idx] == labels[idx].item() else "red"))
    return fig


running_loss = 0.0
for epoch in range(1):
    for i, data in enumerate(trainloader, start=0):
        inputs, labels = data  # get data, data is a list of [inputs, labels]

        optimizer.zero_grad()  # set gradient to 0

        outputs = net(inputs)  # get outcome
        loss = criterion(outputs, labels)  # calculate loss
        loss.backward()  # backward
        optimizer.step()  # update weight

        running_loss += loss.item()

        if i % 1000 == 999:
            writer.add_scalar('training loss',
                              running_loss / 1000,
                              epoch * len(trainloader) + i)
            writer.add_figure('predictions vs. actuals',
                              plot_classes_preds(net, inputs, labels),
                              global_step=epoch * len(trainloader) + i)
            running_loss = 0.0

print("Finished training")

### Assessing trained models with TensorBoard ###
class_probs = []
class_preds = []
with torch.no_grad():  # testset needn't gradient
    for data in testloader:
        images, labels = data
        output = net(images)  # 4 * 10, 10 values for 4 images
        class_probs_batch = [F.softmax(el, dim=0) for el in output]  # get probabilities
        _, class_preds_batch = torch.max(output, 1)  # get predicted label

        class_probs.append(class_probs_batch)  # len(class_probs)=2500
        class_preds.append(class_preds_batch)  # len(class_preds)=2500

# torch.cat(tensors, dim=0, out=None) → Tensor: Concatenates the given sequence of seq tensors in the given dimension.
# torch.stack(tensors, dim=0, out=None) → Tensor: Concatenates sequence of tensors along a new dimension.
test_probs = torch.cat([torch.stack(batch) for batch in class_probs])  # test_probs.size()=torch.Size([10000, 10])
test_preds = torch.cat(class_preds)  # test_preds.size()=torch.Size([10000])


# Adds precision recall curve.
# you provide the ground truth labeling (T/F) and prediction confidence (usually the output of model) for each target.
def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
    tensorboard_preds = test_preds == class_index  # T/F
    tensorboard_probs = test_probs[:, class_index]  # prediction confidence for specific class
    # add_pr_curve(tag, labels, predictions, global_step=None, num_thresholds=127, weights=None, walltime=None)
    writer.add_pr_curve(classes[class_index],
                        tensorboard_preds,
                        tensorboard_probs,
                        global_step=global_step)
    writer.close()


for i in range(len(classes)):
    add_pr_curve_tensorboard(i, test_probs, test_preds)
