#!/bin/bash

scenes=()
for dir in $1*;
do
    scenes+=($(basename $dir))
done

index=$((LSB_JOBINDEX-1))
scene=${scenes[index]}
echo $scene

python scripts/process.py --scannet_path=../scannet/train --res="$2" --scene=$scene --results_path="./results/res_$2"
