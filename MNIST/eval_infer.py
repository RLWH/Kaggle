import torch
import argparse
import numpy as np

from model import LeNet5
from input import MNISTDataset, ToTensor, Normalization
from torchvision import transforms


parser = argparse.ArgumentParser(description="Evaluate or Infer a model")

parser.add_argument("--mode", help="eval mode or infer mode", default="eval", type=str)
parser.add_argument("path", help="Saved model path", default="eval", type=str)

args = parser.parse_args()

transform = transforms.Compose([ToTensor(), Normalization()])

def eval(path):

    #Import the dataset
    eval_dataset = MNISTDataset(mode="eval", transform=transform)
    evalloader = torch.utils.data.DataLoader(eval_dataset, batch_size=128, num_workers=2)
    print("Eval Dataset Loaded. Size: %s" % len(eval_dataset))

    net = LeNet5()
    net.load_state_dict(torch.load(path))
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in evalloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print('Accuracy of the network on the %s test images: %d %%' % (len(eval_dataset), acc))


def infer(path):
    test_dataset = MNISTDataset(mode="test", transform=transform)
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=128, num_workers=2)
    print("Test Dataset Loaded. Size: %s" % len(test_dataset))

    net = LeNet5()
    net.load_state_dict(torch.load(path))
    net.eval()

    all_predictions = []

    with torch.no_grad():
        for data in testloader:
            images = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.append(predicted)

    all_predictions = torch.cat(all_predictions, dim=0)
    np.savetxt("output.csv", all_predictions.int().numpy(), fmt='%d', delimiter=',')


def load_model(model_path):
    cuda = torch.cuda.is_available()
    if cuda:
        net = torch.load(model_path)
    else:
        # Load GPU model on CPU
        net = torch.load(model_path,map_location=lambda storage, loc: storage)

    return net

if __name__ == "__main__":

    if str(args.mode).lower() == "eval":
        print("Evaluate mode")

        eval(args.path)

    elif str(args.mode).lower() == "infer":
        print("Infer mode")

        infer(args.path)

    else:
        raise ValueError("mode should be either 'eval' or 'infer'. ")



