import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # defining max pooling fuction with kernel_size=2 and stride=2
        self.max_pooling = nn.MaxPool2d(2, 2)

        # defining dropout functions with different probabilities
        self.dropout025 = nn.Dropout2d(p=0.25)
        self.dropout03 = nn.Dropout2d(p=0.3)
        self.dropout04 = nn.Dropout2d(p=0.4)

        # defining first convolution with in_channels=1, out_channels=8 and kernel_size=3
        self.convolution1 = nn.Conv2d(1, 8, 3)

        # defining second convolution with in_channels=8, out_channels=16 and kernel_size=3
        self.convolution2 = nn.Conv2d(8, 16, 3)

        # defining linear transformation with in_features=5*5*16 and out_features=128
        self.linear1 = nn.Linear(400, 128)

        # defining linear transformation with in_features=128 and out_features=10 (number of classes)
        self.linear2 = nn.Linear(128, 10)

    # defining forward function
    def forward(self, x):
        x = self.dropout025(x)

        # first layer: convolution, activation, max pooling
        x = self.max_pooling(F.relu(self.convolution1(x)))
        x = self.dropout025(x)

        # second layer: convolution, activation, max pooling
        x = self.max_pooling(F.relu(self.convolution2(x)))
        x = self.dropout03(x)

        # flattening the tensor
        x = x.view(x.size(0), -1)

        # linear functions
        x = F.relu(self.linear1(x))
        x = self.dropout04(x)
        x = self.linear2(x)

        return x