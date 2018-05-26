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

    train_dataset = MNISTDataset(mode="train", transform=transform)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2)

    print("Train Dataset size: %s" % len(train_dataset))

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
    net.train()
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
        best_accuracy = 0.0

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

            if i % 50 == 49:
                print('[%d, %5d] loss: %.3f - Training Accuracy:%.3f' % (epoch + 1, i + 1, running_loss/500,
                                                                correct_count/labels_count * 100))
                running_loss = 0
                labels_count = 0
                correct_count = 0

        # Evaluate accuracy
        acc = correct_count/labels_count * 100
        print("Epoch %s finished. Accuracy: %s" % (epoch, acc))

    print("Finished training. Saving model...")
    torch.save(net.state_dict(), 'MNIST_le_5.pt')

if __name__ == "__main__":
    train()