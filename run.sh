#!/bin/bash
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Niccolo Antonello <nantonel@idiap.ch>,
# Philip N. Garner <pgarner@idiap.ch>

. ./path.sh
echo "Using conda enviroment: $PYPATH"

data="cifar10"
arch="densenet"
declare -a nus=(0.5 1.0 5.0 10.0)

$PYPATH/python src/train.py --epochs 300 --decay 0.0001 --batch-size 64 --data $data --arch $arch
for nu in "${nus[@]}"; do
  $PYPATH/python src/train.py --epochs 20 --decay 0.01 --batch-size 128 --data $data --arch $arch \
    --use-tnet $nu
done
# qsub -cwd -l q_gpu -P shissm -o log.log -e log.log run.sh
