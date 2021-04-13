# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Niccolo Antonello <nantonel@idiap.ch>,
# Philip N. Garner <pgarner@idiap.ch>

from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as trn
from convnet import ConvNet
from densenet import DenseNet
from tnet import TNet
from utils import get_normal

def save_model(model, tnet, arch, data, save_path='models', best=False):
    # Make save directory
    save_path += '/' + data
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if model.nu != 0:
        suffix = "_nu{}".format(model.nu)
    else:
        suffix = ""

    if best:
        suffix += "_best"

    if tnet == None:
        md = model.state_dict()
        torch.save(md,"{}/{}{}.pt".format(save_path, arch, suffix))
    else:
        md = tnet.state_dict()
        torch.save(md,"{}/tnet_{}{}{}.pt".format(save_path, tnet.nu, arch, suffix))

def train(args, model, tnet, device, train_loader, optimizer, epoch):
    if tnet == None:
        model.train()
    else:
        model.eval()
        tnet.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        if tnet == None:
            output = model(data)
        else:
            with torch.no_grad():
                y = model.penultimate_layer(data)
            output = tnet(y)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

def test(args, model, tnet, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if tnet == None:
                output = model(data)
            else:
                y = model.penultimate_layer(data)
                output = tnet(y)

            test_loss += F.nll_loss(output, target, reduction='sum').item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = correct / len(test_loader.dataset)

    if args.verbose > 0:
        print('Test set: Average loss: {:.4f}, Test Accuracy: {}/{} ({:.0f}%)'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * accuracy)
            )
    
    return accuracy

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='CIFAR classifiers')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=200, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
    parser.add_argument('--milestones', default=True, metavar='LRM',
                        help='utilizes scheduler learning rate')
    parser.add_argument('--verbose', type=int, default=1, metavar='V',
                        help='verbosity')
    
    parser.add_argument('--data', type=str, default='cifar10', metavar='DT',
                        help='dataset')
    parser.add_argument('--arch', type=str, default='densenet', metavar='ARC',
                        help='neural network architecture')
    parser.add_argument('--use-tsoftmax', type=float, default=0.0,
                        help='train model with tsoftmax layer from scratch, if 0 train model without it')
    parser.add_argument('--use-tnet', type=float, default=0.0,
                        help='train tsoftmax layer using a pre-trained model, if 0 train model without it')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    device = torch.device("cuda" if use_cuda else "cpu")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    mean, std = get_normal(args.data)

    if args.data == "cifar10": 
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                           trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('../data', train=False, transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        Nc = 10
        channels=3

    elif args.data == "svhn": 
        train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                           trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN('../data', split='train', download=True, transform=train_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN('../data', split='test', download=True, transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        Nc = 10
        channels=3

    elif args.data == "fmnist":
        train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=True, download=True,
                           transform=train_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('../data', train=False, transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        Nc = 10
        channels=1

    elif args.data == "kmnist":
        train_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])
        train_loader = torch.utils.data.DataLoader(
            datasets.KMNIST('../data', train=True, download=True,
                           transform=train_transform),
            batch_size=args.batch_size, shuffle=True, **kwargs)
        test_loader = torch.utils.data.DataLoader(
            datasets.KMNIST('../data', train=False, transform=test_transform),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
        Nc = 10
        channels=1

    else:
        print('Training data not supported')
        quit()
        
    if args.arch == 'densenet':
        densenet_depth=100
        model = DenseNet(densenet_depth, Nc, nu=args.use_tsoftmax).to(device)
    elif args.arch == 'densenet_small':
        densenet_depth=10
        model = DenseNet(densenet_depth, Nc, nu=args.use_tsoftmax).to(device)
    elif args.arch == 'convnet':
        model = ConvNet(Nc, channels=channels, nu=args.use_tsoftmax).to(device)

    if args.use_tnet > 0:
        if args.use_tsoftmax != 0:
            suffix = "_nu{}".format(args.use_tsoftmax)
        else:
            suffix = ""
        suffix += "_best"
        # load model
        model.load_state_dict(torch.load('models/{}/{}{}.pt'.format(args.data,args.arch,suffix), 
            map_location=device))
        tnet = TNet(model.penultimate_dim,Nc,
                Nh = (3*model.penultimate_dim)//4,
                nu=args.use_tnet).to(device)
        if args.verbose > 0:
            print(tnet)
        ps = tnet.parameters()
    else:
        tnet = None
        if args.verbose > 0:
            print(model)
        ps = model.parameters()

    optimizer = optim.SGD(ps, 
            lr=args.lr, 
            momentum=args.momentum,
            weight_decay=args.decay, 
            nesterov=True
            )
    if args.milestones:
        if args.epochs == 300:
            milestones=[150,225]
            gamma=1/10
        elif args.epochs == 200:
            milestones=[60,120,160]
            gamma=1/5
        elif args.epochs == 100:
            milestones=[30,60,80]
            gamma=1/5
        else:
            milestones=[args.epochs // 3, 2 * args.epochs // 3]
            gamma=1/10

        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer=optimizer, 
            milestones=milestones, 
            gamma=gamma
        )

    best_accuracy = 0
    for epoch in range(1, args.epochs + 1):
        if args.verbose > 0:
            print('Epoch {}'.format(epoch))
        train(args, model, tnet, device, train_loader, optimizer, epoch)
        if args.milestones:
            scheduler.step()
        accuracy = test(args, model, tnet, device, test_loader)
        if accuracy > best_accuracy:
            print("save best!")
            save_model(model, tnet, args.arch, args.data, best=True)
            best_accuracy = accuracy
    save_model(model, tnet, args.arch, args.data)

if __name__ == '__main__':
    main()
