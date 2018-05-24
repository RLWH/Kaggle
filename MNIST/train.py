import torch

from model import LeNet5


def train():
    model = LeNet5()
    print(model)

    # Print model parameters
    params = list(model.parameters())
    print(len(params))
    print([param.size() for param in params])





if __name__ == "__main__":
    train()