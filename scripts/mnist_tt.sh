#!/bin/bash

module load python_gpu/3.7.1
module load gcc/6.3.0
module load cudnn/7.5
module load cuda/10.0.130
module load magma

for func in "truncate_qtt" "truncate_qtt2" "truncate_tt" "truncate_svd"
do
    for digit in 0 1 2 3 4 5 6 7 8 9
    do
        bsub -R "rusage[mem=5000,ngpus_excl_p=1]" -W 120:00 -J "${func}_${digit}[1-20]" python scripts/mnist_tt.py --truncation=$func --digit=$digit --data="single" --problem="regression" --rank=\$LSB_JOBINDEX
    done
done

for func in "truncate_qtt" "truncate_qtt2" "truncate_tt" "truncate_svd"
do
    for digit in 0 1 2 3 4 5 6 7 8 9
    do
        bsub -R "rusage[mem=5000,ngpus_excl_p=1]" -W 120:00 -J "${func}_${digit}[1-20]" python scripts/mnist_tt.py --truncation=$func --digit=$digit --data="class" --problem="regression" --rank=\$LSB_JOBINDEX
    done
done

for func in "truncate_qtt" "truncate_qtt2" "truncate_tt" "truncate_svd"
do
    bsub -R "rusage[mem=15000,ngpus_excl_p=1]" -W 120:00 -J "${func}[1-20]" python scripts/mnist_tt.py --truncation=$func --digit="-1" --data="all" --problem="classification" --rank=\$LSB_JOBINDEX
    bsub -R "rusage[mem=15000,ngpus_excl_p=1]" -W 120:00 -J "${func}[1-20]" python scripts/mnist_tt.py --truncation=$func --digit="-1" --data="all" --problem="regression" --rank=\$LSB_JOBINDEX
done

