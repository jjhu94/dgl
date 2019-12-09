from __future__ import print_function, division
import os
import torch
import pandas as pd
from matplotlib.transforms import Transform
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import warnings

warnings.filterwarnings("ignore")  # Ignore warnings
plt.ion()  # interactive mode, does not need plt.show()

### a simple example ###
landmarks_frame = pd.read_csv("data/faces/face_landmarks.csv")
n = 65
img_name = landmarks_frame.iloc[n, 0]  # Purely integer-location based indexing for selection by position.
# here means the values of row n and column 0
landmarks = landmarks_frame.iloc[n, 1:].as_matrix()
landmarks = landmarks.astype("float").reshape(-1, 2)  # turn to 2 column matrix
print('Image name: {}'.format(img_name))  # the same as print("Image name: %s" % (img_name))


def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")  # x, y; s: size , c: color of the plot
    # plt.pause(0.001)


plt.figure()  # create a new figure
path = os.path.join("data/faces/", img_name)  # get the path of the specific image
show_landmarks(io.imread(path), landmarks)  # io.imread(): read the image
plt.show()


### Dataset class ###
def show_landmarks(image, landmarks):
    plt.imshow(image)
    plt.scatter(landmarks[:, 0], landmarks[:, 1], s=10, marker=".", c="r")  # x, y; s: size , c: color of the plot
    # plt.pause(0.001)


class FaceLandmarksDataset(Dataset):  # to get sample
    def __init__(self, csv_file, root_dir, transform=None):  # csv_file, root_dir are the path of csv and pictures
        self.landmarks_frame = pd.read_csv(csv_file)  # read the csv in __init__ so as to store them in the memory
        self.root_dir = root_dir
        self.transform = transform  # the usefulness of transform will be discussed then

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):  # return the value related to specific index
        if torch.is_tensor(idx):  # turn idx to list
            idx = idx.tolist()
        img_name = self.landmarks_frame.iloc[idx, 0]
        img_path = os.path.join(self.root_dir, img_name)
        image = io.imread(img_path)
        landmarks = self.landmarks_frame.iloc[idx, 1:]
        landmarks = np.array([landmarks])
        landmarks = landmarks.astype("float").reshape(-1, 2)
        sample = {"image": image, "landmarks": landmarks}  # dict

        if self.transform:
            sample = self.transform(sample)

        return sample


face_dataset = FaceLandmarksDataset(csv_file="data/faces/face_landmarks.csv", root_dir="data/faces/")
fig = plt.figure()

for i in range(len(face_dataset)):
    sample = face_dataset[i]
    print(i, sample["image"].shape, sample["landmarks"].shape)
    ax = plt.subplot(1, 4, i+1)  # print multiple pictures in whole picture. H=1 and w=4, i means the number.
    plt.tight_layout()  # adjust subplot to fill the entire image area
    ax.set_title("sample #{}".format(i))  # set title for each subplot
    ax.axis("off")  # do not show the axis
    show_landmarks(**sample)  # turn dict to parameters:  {'a':1,'b':2,'c':3} -> a=1,b=2,c=3

    if i == 3:
        plt.show()
        break


### Transforms ###
# do prepocess to make the images with different sizes a fixed size
class Rescale(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        # Returns a Boolean stating whether the object is an instance or subclass of another object
        self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]
        h, w = image.shape[:2]  # image.shape: (h,w,c)
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size  # if is tuple

        new_h, new_w = int(new_h), int(new_w)
        img = transform.resize(image, (new_h, new_w))  # resize the image
        landmarks = landmarks * [new_w / w, new_h / h]  # relocate the spot
        return {"image": img, "landmarks":landmarks}


class RandomCrop(object):
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            self.output_size = output_size

    def __call__(self, sample):
        image, landmarks = sample["image"], sample["landmarks"]

        h, w = image.shape[:2]

        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)  # get an int ranging [0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h,  # H from [top, top + new_h); W from [left, left + new_w)
                left: left + new_w]

        landmarks = landmarks - [left, top]
        return {'image': image, 'landmarks': landmarks}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']

        image = image.transpose((2, 0, 1))  # H x W x C in numpy; C X H X W in torch
        return {'image': torch.from_numpy(image),
                'landmarks': torch.from_numpy(landmarks)}


### Compose transforms ###
scale = Rescale(256)
crop = RandomCrop(128)
composed = transforms.Compose([Rescale(256),
                               RandomCrop(224)])  # compose 2 transforms
fig = plt.figure()
sample = face_dataset[65]
for i, tsfrm in enumerate([scale, crop, composed]):
    transformed_sample = tsfrm(sample)

    ax = plt.subplot(1, 3, i + 1)
    plt.tight_layout()
    ax.set_title(type(tsfrm).__name__)
    show_landmarks(**transformed_sample)

plt.show()


### Iterating through the dataset ###
transformed_dataset = FaceLandmarksDataset(csv_file='data/faces/face_landmarks.csv',
                                           root_dir='data/faces/',
                                           transform=transforms.Compose([
                                               Rescale(256),
                                               RandomCrop(224),
                                               ToTensor()
                                           ]))

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)


def show_landmarks_batch(sample_batched):
    image_batch, landmarks_batch = sample_batched["image"], sample_batched["landmarks"]
    batch_size = len(image_batch)
    im_size = image_batch.size(2)  # image_batch.size(): n, c, h (transformed), w (transformed)
    grid_border_size = 2

    grid = utils.make_grid(image_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    for i in range(batch_size):
        plt.scatter(landmarks_batch[i, :, 0].numpy() + i * im_size + (i + 1) * grid_border_size,
                    landmarks_batch[i, :, 1].numpy() + grid_border_size,
                    s=10, marker='.', c='r')  # x, y

        plt.title("Batch from dataloader")


for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['image'].size(),
          sample_batched['landmarks'].size())

    if i_batch == 3:
        plt.figure()
        show_landmarks_batch(sample_batched)
        plt.axis('off')
        plt.ioff()  # used in interactive mode
        plt.show()
        break

