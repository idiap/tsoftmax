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

import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Plot figures of merit')
    
    parser.add_argument('--arch', type=str, default='convnet', metavar='ARC',
                        help='neural network architecture')
    parser.add_argument('--data', type=str, default='fmnist', metavar='DT',
                        help='dataset used to train the classifier')
    parser.add_argument('--save-path', type=str, default='data', metavar='SP',
                        help='dataset')
    parser.add_argument('--to-csv', action='store_true', default=False,
                        help='export to comma separated value format')
    parser.add_argument('--csv-dest', type=str, default='csv', 
            metavar='DEST', help='folder where to save csv files')
    args = parser.parse_args()

    f='{}/{}_fom_{}.csv'.format(args.save_path, args.arch, args.data)
    if os.path.exists(f):
        df = pd.read_csv(f, index_col=False) 
        #df.set_index('ood', inplace=True)
        oods = df['ood'].unique()
        modes = df['mode'].unique()
    else:
        print('{} not found!'.format(f))
        quit()

    if args.data == 'cifar10':
        modes=['msp','odin','tmsp0.5','tmsp','tmsp5.0','tmsp10.0']
        foms=['fprtpr95','de','rocauc','prauc']
        oods=['svhn', 'lsun', 'fakedata']
    elif args.data == 'fmnist':
        modes=['msp','odin','tmsp0.5', 'tmsp', 'tmsp5.0', 'tmsp10.0']
        foms=['fprtpr95','de','rocauc','prauc']
        oods=['mnist', 'kmnist', 'cifar10_bw', 'fakedata_bw']
    elif args.data == 'kmnist':
        modes=['msp','odin','tmsp0.5', 'tmsp', 'tmsp5.0', 'tmsp10.0']
        foms=['fprtpr95','de','rocauc','prauc']
        oods=['mnist', 'fmnist', 'cifar10_bw', 'fakedata_bw']

    x = np.arange(len(oods))  # the label locations
    width = 0.5  # the width of the bars
    pos = np.linspace(width/2,-width/2,num=len(modes))

    df.set_index(['mode'], inplace=True)
    
    fig, ax = plt.subplots(len(foms),1)
    for z in range(len(foms)):
        ymin,ymax=100,-100
        for k, m in enumerate(modes):
            y = df.loc[m].set_index(['ood'])[foms[z]]
            y = y[oods]
            ax[z].bar(x - pos[k], y, width/len(modes), label=modes[k])
            labels = y.index.tolist()
            if args.to_csv:
                print(y)
                if not os.path.exists(args.csv_dest):
                    os.makedirs(args.csv_dest)
                y[oods].reset_index().to_csv(args.csv_dest+
                        '{}_{}_{}_{}.csv'.format(args.arch,args.data,modes[k],foms[z]), 
                        header=['ood','fom'])
            y = y.to_numpy()
            ymin = min(y.min(),ymin)
            ymax = max(y.max(),ymax)
        ax[z].set_ylim((ymin*0.8,ymax*1.2))
        ax[z].set_title(foms[z])
        ax[z].set_xticks(x)
        if z == len(foms)-1:
            ax[z].set_xticklabels(labels)
        else:
            ax[z].set_xticks([])
        if z == 0:
            ax[z].legend()

    if not args.to_csv:
        plt.show()

if __name__ == '__main__':
    main()
