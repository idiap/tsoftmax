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

for data in kmnist fmnist; do
  echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
  echo " $data Classifier IND"
  echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
  for mode in msp tmsp0.5 tmsp tmsp5.0 tmsp10.0 odin; do
    echo "mode $mode"
    $PYPATH/python src/get_conf.py --arch $arch --data $data --test-data $data \
      --mode $mode || exit 1
    $PYPATH/python src/get_FOM.py --arch $arch --data $data \
      --mode $mode || exit 1
  done

  echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
  echo " $data Classifier OOD"
  echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
  for data_test in mnist fmnist kmnist cifar10_bw fakedata_bw; do
    echo "- - - -"
    if [ "$data" != "$data_test" ]; then
      for mode in msp tmsp0.5 tmsp tmsp5.0 tmsp10.0  odin; do
        if [[ "$data" == "fmnist" && "$mode" == "odin" ]]; then
          odin_opt="--T 10000 --eps 0.01" # got this from tune_odin.sh
        elif [[ "$data" == "fmnist" && "$mode" == "odin" ]]; then
          odin_opt="--T 1000 --eps 0.001" # got this from tune_odin.sh
        else
          odin_opt=""
        fi
        echo "mode $mode"
        $PYPATH/python src/get_conf.py --arch $arch --data $data --test-data $data_test \
          --mode $mode $odin_opt || exit 1
        $PYPATH/python src/get_FOM.py --arch $arch --data $data --ood $data_test \
          --mode $mode || exit 1
      done
    fi
  done
done
