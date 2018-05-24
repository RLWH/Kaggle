import torch
import pandas as pd

from torch.utils.data import Dataset


class MNISTDataset(Dataset):
    """
    MNIST Dataset
    """

    def __init__(self, csv_file, transform=None):
        """
        Constructor
        :param csv_file: string. Path to the csv file with image pixels
        :param transform: callable. Optional transformation to be applied on a sample (Optional)
        """

        self.data = pd.read_csv(csv_file)
        self.transform = transform