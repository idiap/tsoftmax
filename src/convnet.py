# Modified from https://github.com/pytorch/examples/blob/master/mnist/main.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from tsoftmax import Quadratic, TSoftmax, TLogSoftmax  

class ConvNet(nn.Module):
    def __init__(self, num_classes, channels=1, nu=0.0 ):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4*4*50, 500)
        self.fc2 = nn.Linear(500, 100)
        self.nu = nu
        if self.nu == 0:
            self.fc3 = nn.Linear(100, num_classes)
            self.out = nn.LogSoftmax(dim=1) 
        else:
            self.fc3 = Quadratic(100, num_classes)
            self.out = TLogSoftmax(nu=nu)

    def last_layer(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 4*4*50)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y

    def forward(self, x):
        y = self.last_layer(x)
        return self.out(y)

if __name__ == '__main__':
    for channels in [1,3]:
        net=ConvNet(10,channels=channels)
        x = torch.randn(5,channels,28,28)
        y = net(x)
        print(y.size())

        net=ConvNet(10, nu=1.0, channels=channels)
        x = torch.randn(5,channels,28,28)
        y = net(x)
        print(y.size())
