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

arch="convnet"
data="kmnist"
data_test="fakedata_bw"

for eps in 0.0001 0.001 0.01; do
  for T in 100000 10000 1000; do
    echo "ODIN eps=$eps T=$T"
    $PYPATH/python src/get_conf.py --arch $arch --data $data --test-data $data \
       --mode odin --T $T --eps $eps || exit 1
    $PYPATH/python src/get_conf.py --arch $arch --data $data --test-data $data_test \
       --mode odin --T $T --eps $eps || exit 1
    $PYPATH/python src/get_FOM.py --arch $arch --data $data --ood $data_test \
       --mode odin || exit 1
  done
done
