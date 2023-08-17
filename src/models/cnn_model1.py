import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 

from colorama import Style

from data import NUM_CLASSES, IMG_HEIGHT, IMG_WIDTH
from data import log_success

class CNN_1(nn.Module):
    """
    first convnet model
    """

    def __init__(self):
        super(CNN_1, self).__init__()
        
        log_success('CNN_1 created')

        # dim text for logging purposes
        print(Style.DIM)
        print(f"input size: [3, {IMG_WIDTH}, {IMG_HEIGHT}]")

        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=(8, 16),
        )
        output_size = self.conv1(torch.zeros(3, IMG_WIDTH, IMG_HEIGHT)).size()
        print(f"size after conv1 {output_size}")

        self.pool1 = nn.MaxPool2d(
            kernel_size=(5, 5),
            stride=10,
        )
        output_size = self.pool1(torch.zeros(output_size)).size()
        print(f"size after pool1 {output_size}")

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(5, 5),
        )
        output_size = self.conv2(torch.zeros(output_size)).size()
        print(f"size after conv2 {output_size}")

        self.pool2 = nn.MaxPool2d(
            kernel_size=(2, 2),
            stride=5,
        )
        output_size = self.pool2(torch.zeros(output_size)).size()
        print(f"size after pool2 {output_size}")

        output_size = (output_size[0]*output_size[1]*output_size[2])

        self.fc1 = nn.Linear(output_size, 100)
        output_size = self.fc1(torch.zeros(output_size)).size()
        print(f"size after fc1 {output_size}")


        self.fc2 = nn.Linear(100, NUM_CLASSES)
        output_size = self.fc2(torch.zeros(output_size)).size()
        print(f"size after fc2 {output_size}")

        # disable dim text
        print(Style.RESET_ALL)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)

        # flatten input
        x = x.view(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)

        return x


    def summary(self):
        """
        summarizes the model
        """

        print(self.__str__())

