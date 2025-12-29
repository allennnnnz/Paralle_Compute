#!/bin/bash
# 用法: bash run_experiments.sh

# 節點數固定
N=1

# 測試用的 n 值
procs=(1 2 3 4 5 6 7 8 9 10 11 12 )

# 你的程式與參數
script="./nsight.sh"
program="./hw1"
arg1=536869888
arg2="testcases/33.in"
arg3="./33.out"

for n in "${procs[@]}"; do
    echo "==== Running with -N $N -n $n ===="
    srun -N $N -n $n $script $program $arg1 $arg2 $arg3
    echo
done
