#! /bin/bash

# Default values
testcase_id=""
outfile="./nsys_reports/rank_${SLURM_PROCID}.nsys-rep"

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --testcase|-t)
      testcase_id=$2
      shift 2
      ;;
    *)
      args+=("$1")
      shift
      ;;
  esac
done

# If testcase_id is given, update outfile
if [[ -n "$testcase_id" ]]; then
    mkdir -p ./nsys_reports/testcase_$testcase_id
    outfile="./nsys_reports/testcase_$testcase_id/rank_${SLURM_PROCID}_t${testcase_id}.nsys-rep"
fi

# Output to ./nsys_reports/rank_$N.nsys-rep
nsys profile \
  -o "$outfile" \
  --mpi-impl openmpi \
  --trace mpi,ucx,osrt,nvtx \
  "${args[@]}"

# Usage: (Under /hw1)
# srun -N 1 -n 12 ../wrapper/nsys_wrap.sh ./hw1 536869888 /home/pp25/share/hw1/testcases/33.in ./testcases/33-mine.out
# srun -N 1 -n 12 ../wrapper/nsys_wrap.sh --testcase 33 ./hw1 536869888 /home/pp25/share/hw1/testcases/33.in ./testcases/33-mine.out