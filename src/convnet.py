# Modified from https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tsoftmax import LogTSoftmax, Quadratic

class ConvNet(nn.Module):
    def __init__(self, num_classes, nu=0, channels=1, penultimate_dim=500):
        super(ConvNet, self).__init__()
        self.penultimate_dim = penultimate_dim
        self.conv1 = nn.Conv2d(channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, self.penultimate_dim)
        self.nu = nu
        if self.nu == 0:
            self.fc2 = nn.Linear(self.penultimate_dim, num_classes)
            self.out = nn.LogSoftmax(dim=1) 
        else:
            self.fc2 = Quadratic(self.penultimate_dim, num_classes)
            self.out = LogTSoftmax(nu=nu)

    def penultimate_layer(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*4*50)
        x = self.fc1(x)
        return x

    def last_layer(self, x):
        x = self.penultimate_layer(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

    def forward(self, x):
        y = self.last_layer(x)
        return self.out(y)

if __name__ == '__main__':
    for channels in [1,3]:
        net=ConvNet(10,channels=channels)
        x = torch.randn(5,channels,28,28)
        y = net(x)
        print(y.size())

        net=ConvNet(10,channels=channels, nu=1.0)
        x = torch.randn(5,channels,28,28)
        y = net(x)
        print(y.size())
