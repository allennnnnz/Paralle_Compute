#!/usr/bin/env bash
# SLURM
#SBATCH --job-name=hw1-bench
#SBATCH --nodes 4
#SBATCH --ntasks-per-node=12
#SBATCH --output=logs/%x/%x-%j.out

set -euo pipefail

ml purge
ml openmpi nsys
cd /home/pp25/pp25s121/Parallel-Computing/hw1
make clean && make



N=$(jq -r '.n' "./testcases/$1.txt")
strategy=${2:-block}

rm -rf ./logs/nsys_reports/$1
mkdir -p ./logs/nsys_reports/$1
for i in $(seq -w 1 12); do
    i_dec=$((10#$i))
    echo "Running with rank $i_dec\n"

    nodes=$(( ((i_dec + 11) / 12) ))
    mkdir -p ./logs/nsys_reports/$1/$i
    srun -N$nodes -n$i_dec --distribution=$strategy --ntasks-per-node=12 \
        ./scripts/nsight.sh ./logs/nsys_reports/$1/$i \
        ./hw1 $N ./testcases/$1.in ./$1.out
    rm -f $1.out

    # Post processing the reports
    ls ./logs/nsys_reports/$1/$i/*.nsys-rep \
        | parallel -j12 'nsys stats {} --force-export=true --report nvtx_sum --timeunit ms --format csv > {.}.csv'
done

# Parse the csvs and generate the plots
pixi run python scripts/parse_nsys.py --case $1 --root ./logs/nsys_reports --out ./logs/plots/$3 --img-dpi 600
