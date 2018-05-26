import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision import transforms


class MNISTDataset(Dataset):
    """
    MNIST Dataset
    """

    def __init__(self, mode, frac=0.95, random_state=22, transform=None, target_transform=None):
        """
        Constructor
        :param csv_file: string. Path to the csv file with image pixels
        :param transform: callable. Optional transformation to be applied on a sample (Optional)
        """

        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        self.frac = frac
        self.random_state = random_state

        # Load the dataset and subset
        dataset = pd.read_csv("data/train.csv")
        train_dataset = dataset.sample(frac=frac, random_state=random_state)
        eval_dataset = dataset.drop(train_dataset.index)

        if self.mode == "train":
            self.data = train_dataset
        elif self.mode == "eval":
            self.data = eval_dataset
        elif self.mode == "test":
            self.data = pd.read_csv("data/test.csv")

    def __len__(self):
        """
        Override method
        :return:
        """
        return len(self.data)

    def __getitem__(self, idx):

        if self.mode != "test":

            image = self.data.iloc[idx, 1:].as_matrix().astype('float').reshape(1, 28, 28)
            label = self.data.iloc[idx, 0]

            if self.transform:
                image = self.transform(image)

            # if self.target_transform:
            #     label = self.target_transform(label)

            return image.float(), label

        else:
            image = self.data.iloc[idx, 0:].as_matrix().astype('float').reshape(1, 28, 28)

            if self.transform:
                image = self.transform(image)

            return image.float()


class ToTensor(object):
    """
    Transformation class that convert ndarrays in sample to Tensors
    """

    def __call__(self, target):
        return torch.from_numpy(target)


class Normalization(object):
    """
    Normalization class that convert image into normalized image
    """

    def __call__(self, target):
        return target.div(255)






