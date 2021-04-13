# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Niccolo Antonello <nantonel@idiap.ch>,
# Philip N. Garner <pgarner@idiap.ch>

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

class Quadratic(nn.Module):
    
    def __init__(self, D, Nc):
        super(Quadratic, self).__init__()
        self.U = nn.Parameter(torch.randn(D, Nc))
        nn.init.kaiming_uniform_(self.U, a=math.sqrt(5))
        
    def forward(self, x):
        x_xt = (x ** 2).sum(-1)
        xt_U = x.mm(self.U)
        U_Ut = ( self.U ** 2 ).sum(0)

        x = 2*xt_U + U_Ut
        x = (x.t() + x_xt.t()).t()
        return x

class TSoftmax(nn.Module):
    
    def __init__(self, nu=1.0):
        super(TSoftmax, self).__init__()
        self.register_buffer('nu', torch.tensor(nu))
        self.register_buffer('beta', -(self.nu+1)/2)
        
    def forward(self, x):
        x = 1.0 + x
        x = x.pow(self.beta)
        x = F.normalize(x, p=1.0, dim=1)
        return x

class LogTSoftmax(nn.Module):
    
    def __init__(self, nu=1.0):
        super(LogTSoftmax, self).__init__()
        self.register_buffer('nu', torch.tensor(nu))
        self.register_buffer('beta', -(self.nu+1)/2)
        
    def forward(self, x):
        x = 1.0 + x
        x = x.pow(self.beta)
        x = F.normalize(x, p=1.0, dim=1)
        return x.log()
