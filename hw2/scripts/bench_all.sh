#!/usr/bin/env bash
# SLURM
#SBATCH --job-name=hw2-bench
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail

if [[ ${SLURM_JOB_ID:-} == "" ]]; then
  echo "This script must be submitted via sbatch." >&2
  exit 2
fi

if [[ $# -lt 2 ]]; then
  echo "Usage: sbatch scripts/bench_all.sh <case_name> <out_dir_for_plots_and_csv>" >&2
  exit 2
fi

OUT_DIR="$2"
CASE_NAME="$1"
CASE_FILE="testcases/$CASE_NAME.txt"

ml purge
ml openmpi nsys || true

make -C src clean && make -C src

export PMIX_MCA_oob_tcp_if_include=ibp3s0

mkdir -p ./logs/nsys_reports ./logs/plots "$OUT_DIR"

run_and_profile() {
  local exe="$1"; shift
  local procs="$1"; shift
  local case_tag="$1"; shift

  local args
  args=$(cat "$CASE_FILE")

  for t in $(seq 1 12); do
    local N_eff=$((procs * t))
    local tpad
    tpad=$(printf "%02d" "$t")
    local run_root="./logs/nsys_reports/${case_tag}/${N_eff}/t${tpad}"
    mkdir -p "$run_root"

    export OMP_NUM_THREADS=$t
    srun -N "$procs" -n "$procs" -c "$t" --exclusive --cpu-bind=cores --distribution=cyclic \
      ./scripts/nsys_wrapper.sh "$run_root" \
      "./src/${exe}" "${case_tag}.${N_eff}.t${tpad}.out" ${args} &
  done
}

# hw2a: pthreads, procs=1, t=1..12
run_and_profile hw2a 1 "${CASE_NAME}_hw2a"

# hw2b baseline: n1c1 for proper speedup baseline
args=$(cat "$CASE_FILE")
mkdir -p "./logs/nsys_reports/${CASE_NAME}_hw2b/1/t01"
export OMP_NUM_THREADS=1
srun -N 1 -n 1 -c 1 --exclusive --cpu-bind=cores \
  ./scripts/nsys_wrapper.sh "./logs/nsys_reports/${CASE_NAME}_hw2b/1/t01" \
  "./src/hw2b" "${CASE_NAME}_hw2b.1.t01.out" ${args}

# Wait for hw2a and hw2b baseline and clean up
wait
rm *.out

# hw2b: OpenMP+MPI, procs=4 fixed, t=1..12
run_and_profile hw2b 4 "${CASE_NAME}_hw2b"
wait
rm *.out

# Convert all .nsys-rep to CSV with GNU parallel via srun on 1 task
find ./logs/nsys_reports -type f -name '*.nsys-rep' | \
  parallel -j48 'srun -N1 -n1 --mpi=none --exclusive --cpu-bind=cores \
    nsys stats {} --force-export=true --report nvtx_sum --timeunit ms --format csv > {.}.csv'

# Parse and plot using polars-based script
pixi run python scripts/parse_nsys.py --case "${CASE_NAME}_hw2a" --root ./logs/nsys_reports --out "${OUT_DIR}" --img-dpi 600

pixi run python scripts/parse_nsys_hw2b.py --case "${CASE_NAME}_hw2b" --root ./logs/nsys_reports --out "${OUT_DIR}" --img-dpi 600
