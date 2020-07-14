#!/bin/bash
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

. ./path.sh
echo "Using conda enviroment: $PYPATH"

data="cifar10"
arch="densenet"

if [ "$arch" == "densenet" ]; then
  opt="--epochs 300 --decay 0.0001 --batch-size 64"
fi

$PYPATH/python src/train.py --nu 0.0 $opt --data $data --arch $arch 
$PYPATH/python src/train.py --nu 1.0 $opt --data $data --arch $arch 
$PYPATH/python src/train.py --nu 0.5 $opt --data $data --arch $arch 
$PYPATH/python src/train.py --nu 5.0 $opt --data $data --arch $arch 
$PYPATH/python src/train.py --nu 10.0 $opt --data $data --arch $arch 
# qsub -cwd -l q_gpu -o log.log -e log.log run.sh
