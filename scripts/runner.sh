#!/bin/bash
module load python_cpu/3.6.4 gcc/6.3.0

export OMP_NUM_THREADS=10

scenes=()
for dir in $1*;
do
    scenes+=($(basename $dir))
done

bsub -R "rusage[mem=$4] span[ptile=$3]" -n $3 -J "scenes[1-${#scenes[@]}]" scripts/process.sh $1 $2

