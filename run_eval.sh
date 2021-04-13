#!/bin/bash
# Copyright (c) 2020 Idiap Research Institute, http://www.idiap.ch/
# Written by Niccolo Antonello <nantonel@idiap.ch>,
# Philip N. Garner <pgarner@idiap.ch>

. ./path.sh
echo "Using conda enviroment: $PYPATH"

declare -a modes=(msp tnet0.5 tnet1.0 tnet5.0 tnet10.0 odin)
for data in cifar10; do
  for arch in densenet; do
    echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
    echo " $data Classifier IND net:$arch"
    echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
    for mode in msp tnet1.0 tnet5.0 tnet10.0 odin; do
      echo "mode $mode"
      $PYPATH/python src/get_conf.py --arch $arch --data $data --test-data $data \
        --mode $mode || exit 1
      $PYPATH/python src/get_FOM.py --arch $arch --data $data \
        --mode $mode || exit 1
    done

    echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
    echo " $data Classifier OOD net:$arch"
    echo "-:-:-:-:-:-:-:-:-:-:-:-:-:-:-:-"
    for data_test in cifar10 svhn lsun fakedata fakedata_wm; do
      echo "- - - -"
      if [ "$data" != "$data_test" ]; then
        for mode in msp tnet0.05 tnet0.1 tnet0.2 tnet0.4 tnet0.6 tnet1.0 tnet5.0 tnet10.0 odin; do
          # from ODIN tuning below
          if [[ "$data" == "cifar10" && "$mode" == "odin" ]]; then
            T=10000
            eps=0.001
          else
            T=0
            eps=0.0
          fi
          echo "mode $mode"
          $PYPATH/python src/get_conf.py --arch $arch --data $data --test-data $data_test \
            --mode $mode --T $T --eps $eps --lsun-path $LSUNPATH || exit 1
          $PYPATH/python src/get_FOM.py --arch $arch --data $data --ood $data_test \
            --mode $mode || exit 1
        done
      fi
    done
  done
done
