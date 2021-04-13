# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Niccolo Antonello <nantonel@idiap.ch>,
# Philip N. Garner <pgarner@idiap.ch>

from __future__ import print_function
import os
import argparse
import numpy as np
import torch
import pandas as pd
from torchvision import datasets
import torchvision.transforms as trn
from convnet import ConvNet
from densenet import DenseNet
from utils import get_normal
import torch.nn.functional as F
import time
from tnet import TNet

def confidence(args, data, model, tnet, mode):
    if mode == 'msp' or mode[:4] == 'tmsp':
        with torch.no_grad():
            output = model(data)
            C, pred = output.max(dim=1, keepdim=True) # get the index of the max log-probability
    elif mode[:4] == 'tnet':
        with torch.no_grad():
            y = model.penultimate_layer(data)
            output = tnet(y)
            C, pred = output.max(dim=1, keepdim=True) # get the index of the max log-probability
    elif mode == 'odin':
        data.requires_grad = True
        model.zero_grad()
        y = model.last_layer(data)
        y = y / args.T
        output = model.out(y)

        labels = torch.argmax(output, dim=1)
        loss = F.nll_loss(output, labels)
        loss.backward()

        with torch.no_grad():
            data = data - args.eps*torch.sign(data.grad)
            y = model.last_layer(data)
            y = y / args.T
            output = model.out(y)
            C, pred = output.max(dim=1, keepdim=True) # get the index of the max log-probability
    return C, pred
    
def test(args, model, tnet, device, test_loader):

    N = len(test_loader.dataset)
    score = torch.zeros(N) 
    is_hit = torch.zeros(N, dtype=torch.bool) 

    model = model.eval()
    delta_t = torch.zeros(len(test_loader))
    correct = 0
    idx = 0
    for i,(data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        t0 = time.time()
        C, pred = confidence(args, data, model, tnet, args.mode)
        delta_t[i] = time.time()-t0
        correct += pred.eq(target.view_as(pred)).sum().item()

        score[idx:idx+data.size(0)] = C.squeeze().cpu()
        if args.data == args.test_data:
            is_hit[idx:idx+data.size(0)] = pred.eq(target.view_as(pred)).squeeze().cpu()
        idx += data.size(0)

    if args.print_time:
        print("Elapsed time per batch mean: {} ms median {} ms".format(
            round(delta_t.mean().item()  *1000, 3),
            round(delta_t.median().item()*1000, 3)
            ))
    return score, is_hit

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Obtain confidence scores from trained networks')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 200)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    
    parser.add_argument('--arch', type=str, default='wrn', metavar='ARC',
                        help='neural network arch')
    parser.add_argument('--data', type=str, default='cifar10', metavar='DT',
                        help='dataset the classifier was trained with')
    parser.add_argument('--test-data', type=str, default='fakedata', metavar='DT',
                        help='dataset used to test the classifier')
    parser.add_argument('--lsun-path', type=str,
                        help='path to LSUN dataset')
    parser.add_argument('--save_path', type=str, default='data', metavar='SP',
                        help='path to save the produced confidence')

    parser.add_argument('--print-time', type=bool, default=False, metavar='PT',
                        help='print elapsed time per batch')
    parser.add_argument('--mode', type=str, default='msp', metavar='M',
            help='available modes: msp (maximum softmax probability), tmsp (t-softmax msp), tmsp0.5, tmsp5, tmsp10 (where number indicates nu value) odin. default: msp')
    parser.add_argument('--T', type=float, default=1000.0, metavar='T',
                        help='ODIN temperature scaling')
    parser.add_argument('--eps', type=float, default=0.001, metavar='EPS',
                        help='ODIN epsilon value')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mean, std = get_normal(args.data)

    if args.test_data == "cifar10": 
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', download=True, train=False, transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.test_data == "cifar10_bw": 
        test_transform = trn.Compose([ 
            trn.Grayscale(),
            trn.Resize((28,28)),
            trn.ToTensor(),trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', download=True, train=False, transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.test_data == "svhn": 
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN('../data', download=True, split='test', transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.test_data == "fakedata":
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
             datasets.FakeData(size=10000,image_size=(3,32,32),transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.test_data == "fakedata_bw":
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
             datasets.FakeData(size=10000,image_size=(1,28,28),transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.test_data == "fakedata_wm":
        # wrong mean normalization
        mean=(50,50,50)
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
             datasets.FakeData(size=10000,image_size=(3,32,32),transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.test_data=="fmnist":
        test_transform = trn.Compose([
            trn.ToTensor(),
            trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
             datasets.FashionMNIST('../data', download=True, train=False, transform=test_transform),
             batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.test_data=="kmnist":
        test_transform = trn.Compose([
            trn.ToTensor(),
            trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
             datasets.KMNIST('../data', download=True, train=False, transform=test_transform),
             batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.test_data=="mnist":
        test_transform = trn.Compose([
            trn.ToTensor(),
            # lambda x : x.repeat(3,1,1) * torch.rand(3,1,1),
            trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
             datasets.MNIST('../data', download=True, train=False, transform=test_transform),
             batch_size=args.test_batch_size, shuffle=True, **kwargs)
    elif args.test_data=="lsun":
        test_transform=trn.Compose([trn.Resize((32,32)), #trn.CenterCrop(32),
            trn.ToTensor(), trn.Normalize(mean, std)])
        test_loader = torch.utils.data.DataLoader(
             datasets.LSUN(args.lsun_path, classes='test', transform=test_transform),
             batch_size=args.test_batch_size, shuffle=True, **kwargs)

    if args.data == "cifar10": 
        Nc = 10
        channels=3
    elif args.data == "svhn":
        Nc = 10
        channels=3
    elif args.data == "fmnist":
        Nc = 10
        channels=1
    elif args.data == "kmnist":
        Nc = 10
        channels=1

    model_path= 'models'
    model_path += '/' + args.data

    if args.mode[:4] == 'tmsp':
        nu = float(args.mode[4:])
    else:
        nu = 0.0

    if args.arch == 'densenet':
        densenet_depth=100
        model = DenseNet(densenet_depth, Nc, nu=nu).to(device)
    elif args.arch == 'densenet_small':
        densenet_depth=10
        model = DenseNet(densenet_depth, Nc, nu=nu).to(device)
    elif args.arch == 'convnet':
        model = ConvNet(Nc, channels=channels, nu=nu).to(device)

    if model.nu != 0:
        suffix = "_nu{}".format(model.nu)
    else:
        suffix = ""

    suffix += "_best"
    model_name = args.arch+suffix

    model.load_state_dict(torch.load(model_path+'/'+model_name+'.pt', 
        map_location=device))

    if args.mode[:4] == 'tnet':
        nu = float(args.mode[4:])
        tnet = TNet(model.penultimate_dim,Nc,
                Nh = (3*model.penultimate_dim)//4,
                nu=nu).to(device)
        tnet.load_state_dict(torch.load(model_path+'/'+'tnet_{}'.format(nu)+model_name+'.pt', 
        map_location=device))
    else:
        tnet = None

    score, is_hit = test(args, model, tnet, device, test_loader)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    df = pd.DataFrame(data={'score' : score, 'is_hit' : is_hit})
    df.to_csv('{}/{}_train{}_test{}_{}.csv'.format(args.save_path, 
        args.arch, 
        args.data, 
        args.test_data, 
        args.mode))

if __name__ == '__main__':
    main()
