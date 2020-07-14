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
from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score, roc_auc_score
from utils import  get_eer, get_fprtpr95

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Obtain figure of merits from confidence scores')
    
    parser.add_argument('--arch', type=str, default='wrn', metavar='ARC',
                        help='neural network arch')
    parser.add_argument('--data', type=str, default='cifar10', metavar='DT',
                        help='dataset')
    parser.add_argument('--ood', type=str, default='', #nargs='+',
            help="OOD testsets.")
    parser.add_argument('--mode', type=str, default='msp', metavar='M',
            help='default: maximum softmax prob.')
    parser.add_argument('--save-path', type=str, default='data', metavar='SP',
                        help='dataset')
    parser.add_argument('--plot', action='store_true')

    args = parser.parse_args()

    df = pd.read_csv('{}/{}_train{}_test{}_{}.csv'.format(args.save_path, 
        args.arch, 
        args.data, 
        args.data,
        args.mode,
        ))
    is_hit, score_ind = df['is_hit'].to_numpy(), df['score'].to_numpy() 
    L = np.size(is_hit)

    if args.ood != '':
        df = pd.read_csv('{}/{}_train{}_test{}_{}.csv'.format(args.save_path, 
            args.arch, 
            args.data, 
            args.ood,
            args.mode,
            ))
        score_ood = df['score'].to_numpy() 
        L = min(L,np.size(score_ood))
        is_hit = np.concatenate((is_hit[:L], np.zeros(L, dtype=bool)))
        score  = np.concatenate((score_ind[:L],score_ood[:L]))
    else:
        score = score_ind

    score = np.exp(score)
    accuracy = np.sum(is_hit) / len(is_hit)
    fpr , tpr , threshold  = roc_curve(is_hit, score)
    EER , OP , idx  = get_eer(fpr , tpr , threshold )
    ROCAUC  = auc(fpr ,tpr )
    FPRTPR95 , DE , OP2 , _ = get_fprtpr95(fpr , tpr , threshold )
    precision , recall , _ = precision_recall_curve(is_hit, score)
    PRAUC  = auc(recall ,precision )

    if args.ood == '':
        print("Accuracy:{}, Test Error:{}%".format(accuracy, 
            round((1-accuracy)*100,2)
            ))
    else:
        print("OOD & FPRTPR95 & DE & ROCAUC & PRAUC \\\\")
        print("{0} & {1:0.3f} & {2:0.3f} & {3:0.3f} & {4:0.3f} \\\\".format(args.ood, FPRTPR95, DE, ROCAUC, PRAUC))
        FOM=[args.ood, args.mode, EER, FPRTPR95, DE, ROCAUC, PRAUC]
        columns=['ood','mode','eer','fprtpr95','de','rocauc','prauc']
        f='{}/{}_fom_{}.csv'.format(args.save_path, args.arch, args.data)
        if os.path.exists(f):
            fom = pd.read_csv(f, index_col=False) 
            fom.set_index(['ood', 'mode'], inplace=True)
            fom.loc[(args.ood,args.mode), columns[2:]] = FOM[2:]
            fom.reset_index(inplace=True)
        else:
            fom = pd.DataFrame((FOM,), columns=columns)
        fom.to_csv(f, index=False)

    if args.plot:
        fig, axs = plt.subplots(2,1)
        axs[0].plot([fpr[idx]], [tpr[idx]], 'bo' )
        axs[0].plot(fpr,  tpr , label="softmax  AUC={0:0.3f} FPR@TPR95={1:0.3f}".format(ROCAUC, FPRTPR95 ) )
        axs[0].set_xlabel("FPR")
        axs[0].set_ylabel("TPR")
        axs[0].set_title("ROC curve" )
        axs[0].legend()

        axs[1].plot(recall , precision , label="softmax  AUC={0:0.3f}".format(PRAUC)  )
        axs[1].set_ylabel("Precision")
        axs[1].set_xlabel("Recall")
        axs[1].legend()
        axs[1].set_title("PR curve")

        fig, axs = plt.subplots(2,1)
        bins = np.linspace(0,1,100)
        axs[0].set_title("softmax")
        indc, _, _  = axs[0].hist(score[:L], bins=bins, alpha = 0.5, label= "IND");
        if args.ood != '':
            oodc, _, _  = axs[0].hist(score[L:], bins=bins, alpha = 0.5, label= "OOD");
        axs[0].plot([OP ,OP ], [0,100], 'b:', label="EER OP");
        axs[0].plot([OP2,OP2], [0,100], 'r:', label="FPR95 OP");
        axs[0].set_yscale('log')
        axs[1].set_ylabel("Counts")
        axs[0].legend()
        plt.show()

if __name__ == '__main__':
    main()
