#!/bin/bash
module load python_cpu/3.6.4 gcc/6.3.0

export OMP_NUM_THREADS=10

bsub -R "rusage[mem=$3] span[ptile=$2]" -n $2 -W 24:00  python scripts/process.py --scannet_path=../scannet/train --res=$1 --scene=$4 --results_path="./results/res_$1"

