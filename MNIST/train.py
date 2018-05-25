import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import LeNet5
from input import MNISTDataset, ToTensor, Normalization
from torchvision import transforms


def train():

    # Check GPU Status
    print("Checking GPU status")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    # Assume that we are on a CUDA machine, then this should print a CUDA device:
    print("Using device %s" % device)

    # Load the dataset
    transform = transforms.Compose([ToTensor(), Normalization()])
    target_transform = transforms.Compose([ToTensor()])

    train_dataset = MNISTDataset(mode="train", transform=transform,target_transform=target_transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)

    eval_dataset = MNISTDataset(mode="eval", transform=transform, target_transform=target_transform)
    evalloader = torch.utils.data.DataLoader(eval_dataset)

    test_dataset = MNISTDataset(mode="test", transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)

    print("Train Dataset size: %s" % len(train_dataset))
    print("Eval Dataset size: %s" % len(eval_dataset))
    print("Test Dataset size: %s" % len(test_dataset))

    # Try some samples
    # dataiter = iter(trainloader)
    # images, labels = dataiter.next()
    # plt.imshow(images[0].numpy())
    # print("The label is %s" % labels[0])
    # plt.show()
    # print(images[0].size(), labels[0].size())

    # Load Model
    print("Loading Model...")
    net = LeNet5()
    net.to(device)
    print(net)

    # Define Loss
    criteria = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.003)

    print("Start training...")
    for epoch in range(10):

        print("Epoch %s started..." % epoch)

        running_loss = 0.0
        correct_count = 0.0
        labels_count = 0.0

        for i, data in enumerate(trainloader, 0):

            # Fetch the inputs
            inputs, labels = data

            # Send the inputs and labels to device
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # Forward prop and back prop, then update
            output = net(inputs)
            loss = criteria(output, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            labels_count += labels.size(0)
            correct_count += (predicted == labels).sum().item()

            if i % 100 == 99:
                print('[%d, %5d] loss: %.3f - Training Accuracy:%.3f' % (epoch + 1, i + 1, running_loss/500,
                                                                correct_count/labels_count * 100))
                running_loss = 0
                labels_count = 0
                correct_count = 0


    # Print model parameters
    params = list(net.parameters())
    print(len(params))
    print([param.size() for param in params])


if __name__ == "__main__":
    train()