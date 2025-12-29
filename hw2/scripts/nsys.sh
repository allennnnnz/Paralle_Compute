#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./scripts/nsight.sh <report_dir> [run_label] -- <your_program> <args...>
#   (如果不加 -- 也可以，腳本會自動 shift 第一個/第二個參數後把剩下的當指令)
#
# Examples:
#   ./scripts/nsight.sh ./logs/nsys_reports runA ./hw2pool out.png 10000 -2 2 -2 2 800 800
#   ./scripts/nsight.sh ./logs/nsys_reports ./hw2seq out.png 10000 -2 2 -2 2 800 800

report_dir="$1"; shift

# 可選的 run label（沒有就用 "run"）
run_label="${1:-run}"
# 如果下一個參數看起來是程式（不是以 - 開頭），就視為 label；否則當 label 省略
if [[ "$run_label" == -* || "$#" -eq 0 ]]; then
  run_label="run"
else
  shift
fi

# 其餘參數就是實際要跑的命令
if [[ "$#" -lt 1 ]]; then
  echo "ERROR: missing command to profile." >&2
  exit 1
fi

mkdir -p "${report_dir}"

# 取得 rank：優先 SLURM，再來 PMIx，最後預設 0
rank="${SLURM_PROCID:-${PMIX_RANK:-0}}"

# 自動偵測是否為 MPI 環境（有 OMPI 或 PMIx 就當作 MPI）
is_mpi=0
if [[ -n "${OMPI_COMM_WORLD_SIZE:-}" || -n "${PMIX_RANK:-}" ]]; then
  is_mpi=1
fi

# 組合輸出檔名（不加副檔名，nsys 會產生 .qdrep）
timestamp="$(date +%Y%m%d_%H%M%S)"
base="rank_${rank}_${run_label}_${timestamp}"
out_path="${report_dir}/${base}"

# 依情境決定 trace 與 mpi 參數
# 依情境決定 trace 與 mpi 參數
trace_list="osrt,nvtx"
nsys_extra=()
# CPU 取樣 + 顯示 context switch（process-tree）+ 載入統計
nsys_extra+=( --sample=cpu --cpuctxsw=process-tree --stats=true --force-overwrite=true )

if [[ -n "${OMPI_COMM_WORLD_SIZE:-}" || -n "${PMIX_RANK:-}" ]]; then
  trace_list="mpi,ucx,osrt,nvtx"
  nsys_extra+=( --mpi-impl openmpi )
fi

echo "[nsight.sh] rank=${rank} label=${run_label} mpi=${is_mpi} -> ${out_path}.qdrep"
echo "[nsight.sh] trace=${trace_list}"
echo "[nsight.sh] cmd: $*"

# 真的開始 profile
nsys profile \
  -o "${out_path}" \
  --trace="${trace_list}" \
  "${nsys_extra[@]}" \
  "$@"