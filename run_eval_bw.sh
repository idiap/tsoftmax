#!/bin/bash
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Niccolo Antonello <nantonel@idiap.ch>,
# Philip N. Garner <pgarner@idiap.ch>

. ./path.sh
echo "Using conda enviroment: $PYPATH"
arch="convnet"
declare -a modes=(msp tmsp0.5 tnet0.5 tmsp1.0 tnet1.0 tmsp5.0 tnet5.0 tmsp10.0 tnet10.0 odin)

for data in fmnist kmnist; do
  echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
  echo " $data Classifier IND"
  echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
  for mode in "${modes[@]}"; do
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
      for mode in "${modes[@]}"; do
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
