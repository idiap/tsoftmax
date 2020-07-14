# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Niccolo Antonello <nantonel@idiap.ch>,
# Philip N. Garner <pgarner@idiap.ch>
# 
# This file is part of tsoftmax.
# 
# tsoftmax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License version 3 as
# published by the Free Software Foundation.
# 
# tsoftmax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with tsoftmax. If not, see <http://www.gnu.org/licenses/>.

import numpy as np
import math
import torch
from torchvision import datasets
import torch.utils.data as torchdata
import torchvision.transforms as trn
from PIL import Image as PILImage

def get_eer(fpr,tpr,threshold):
    fnr = 1 - tpr
    idx = np.nanargmin(np.absolute((fnr - fpr)))
    OP = threshold[idx]
    EER = fpr[idx]
    return EER, OP, idx

def get_fprtpr95(fpr,tpr,threshold):
    idx = (tpr >= 0.95).argmax()
    OP = threshold[idx]
    TPR95 = tpr[idx]
    FPRTPR95 = fpr[idx]
    DE = 0.5*(1-TPR95) + 0.5*FPRTPR95
    return FPRTPR95, DE, OP, idx

def get_normal(train_data):
    if train_data == 'cifar10':
        mean=[x / 255.0 for x in [125.3, 123.0, 113.9]]
        std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
    elif train_data == 'svhn':
        mean = [x / 255.0 for x in [109.9, 109.7, 113.8]]
        std  = [x / 255.0 for x in [50.1,  50.6,  50.8 ]]
    elif train_data == 'fmnist':
        mean=[0.5]
        std=[0.5]
    elif train_data == 'kmnist':
        mean=[0.5]
        std=[0.5]
    else:
        print('{} means and std unknown! Add in utils.py'.format(train_data))
        quit()
    return mean, std
