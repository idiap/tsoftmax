# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Niccolo Antonello <nantonel@idiap.ch>,
# Philip N. Garner <pgarner@idiap.ch>

import torch
import torch.nn as nn
import torch.nn.functional as F
from tsoftmax import Quadratic, LogTSoftmax  

class TNet(nn.Module):
    def __init__(self, Nx, Ny, Nh=10, nu=1.0):
        super(TNet, self).__init__()
        self.fc = nn.Linear(Nx, Nh)
        self.q  = Quadratic(Nh, Ny)
        self.nu = nu
        self.out = LogTSoftmax(nu=nu) 

    def forward(self, x):
        x = F.relu(x)
        x = F.relu(self.fc(x))
        x = self.q(x)
        return self.out(x)

if __name__ == '__main__':
    net=TNet(10,30)
    x = torch.randn(5,10)
    y = net(x)
    print(y.size())
