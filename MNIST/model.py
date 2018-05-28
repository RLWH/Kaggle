import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):

    def __init__(self):
        # Override the Net object
        super(LeNet5, self).__init__()

        # Implement a LeNet-5 Network as an example
        # Conv1 (nc=6, f=5x5, s=1, p=0)- BN - RELU - Pool1 (f=2x2, s=2)
        # - Conv2 (nc=16, f=5x5, s=1, p=0) - BN - RELU - Pool2 (f=2x2, s=2)
        # - Fc (120) - Fc (84) - Fc (10)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)

        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(6)
        self.bn2 = nn.BatchNorm2d(16)

        # FC Layers
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)


    def forward(self, x):

        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=2)

        # FC Layers
        x = F.relu(self.fc1(x.view(-1, self.num_flat_features(x))))
        x = F.relu(self.fc2(x))
        out = self.out(x)

        return out


    def num_flat_features(self, x):
        """
        A function to calculate what is the flat dimension if the conv layer is flattened
        :param x:
        :return:
        """
        size = x.size()[1:]     # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features


class Net2(nn.Module):

    def __init__(self):
        # Override the Net object
        super(Net2, self).__init__()

        # Implement a LeNet-5 Network as an example
        # Conv1 (nc=6, f=5x5, s=1, p=0)- BN - RELU - Pool1 (f=2x2, s=2)
        # - Conv2 (nc=16, f=5x5, s=1, p=0) - BN - RELU - Pool2 (f=2x2, s=2)
        # - Fc (120) - Fc (84) - Fc (10)

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        # BatchNorm layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)

        # FC Layers
        self.fc1 = nn.Linear(64 * 28 * 28, 120)
        self.fc2 = nn.Linear(120, 84)
        self.out = nn.Linear(84, 10)

        # Dropout layers
        self.dropout1 = nn.Dropout2d(p=0.5)

    def forward(self, x):

        # Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)
        x = F.max_pool2d(x, kernel_size=2)

        # Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.bn2(x)
        x = F.max_pool2d(x, kernel_size=2)

        # FC Layers
        x = F.relu(self.fc1(x.view(-1, self.num_flat_features(x))))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        out = self.out(x)

        return out


    def num_flat_features(self, x):
        """
        A function to calculate what is the flat dimension if the conv layer is flattened
        :param x:
        :return:
        """
        size = x.size()[1:]     # All dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s

        return num_features









