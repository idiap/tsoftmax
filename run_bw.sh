#!/bin/bash
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Niccolo Antonello <nantonel@idiap.ch>,
# Philip N. Garner <pgarner@idiap.ch>
set -e

. ./path.sh
echo "Using conda enviroment: $PYPATH"

data=$1 #kmnist or fmnist
arch="convnet"
epochs=20
declare -a nus=(0.5 1.0 5.0 10.0)

$PYPATH/python src/train.py --epochs $epochs --data $data --arch $arch
for nu in "${nus[@]}"; do
  $PYPATH/python src/train.py --epochs $epochs --data $data --arch $arch --use-tsoftmax $nu
  $PYPATH/python src/train.py --epochs 5 --data $data --arch $arch --use-tnet $nu
done
