#!/usr/bin/env bash
#
# SLURM
#SBATCH --job-name=hw1
#SBATCH --nodes 4
#SBATCH --ntasks-per-node=12
#SBATCH --output=logs/%x/%x-%j.out

set -u  # fail on unset vars

# =================== config ===================
# Float comparison tolerance (can be overridden: EPS=1e-7 sbatch ...)
EPS="${EPS:-1e-6}"

# Simple color helper (colors only if stdout is a TTY)
if [[ -t 1 ]]; then
  RED=$'\033[31m'; GRN=$'\033[32m'; YLW=$'\033[33m'; BLD=$'\033[1m'; CLR=$'\033[0m'
else
  RED=""; GRN=""; YLW=""; BLD=""; CLR=""
fi
check_with_hw1_floats() {
  local expected="$1" got="$2" n="$3"
  local report
  report="$(hw1-floats "$expected" "$got")"

  awk -v eps="$EPS" -v RED="$RED" -v CLR="$CLR" -v N="$n" '
    function abs(x){return x<0?-x:x}
    function isnum(t){ return (t ~ /^[-+]?([0-9]*\.?[0-9]+|[0-9]+\.)([eE][-+]?[0-9]+)?$/) }
    function isdash(t){ return (t ~ /^-+$/) }

    BEGIN { mismatch=0; worst=0; samples=0; seen=0 }

    /^[[:space:]]*[0-9]+[[:space:]]+([-+]?([0-9]*\.?[0-9]+|[0-9]+\.)([eE][-+]?[0-9]+)?|-+)[[:space:]]+([-+]?([0-9]*\.?[0-9]+|[0-9]+\.)([eE][-+]?[0-9]+)?|-+)[[:space:]]*$/ {
      seen=1
      i=$1; a=$2; b=$3

      # Ignore sentinel or anything beyond N
      if (i > N) next

      # If both non-numeric, ignore
      if (!isnum(a) && !isnum(b)) next

      # If exactly one side missing, mismatch
      if (!isnum(a) || !isnum(b)) {
        mismatch++
        if (samples<10) printf("    %s[%d]%s %s | %s (missing)\n", RED, i, CLR, a, b) > "/dev/stderr"
        samples++
        next
      }

      # Numeric compare
      xa=a+0; xb=b+0
      d=xa-xb; ad=abs(d)
      if (ad>eps) {
        mismatch++
        if (ad>worst){worst=ad; worst_i=i; worst_a=xa; worst_b=xb}
        if (samples<10) printf("    %s[%d]%s %.8g | %.8g (Δ=%.3g)\n", RED, i, CLR, xa, xb, d) > "/dev/stderr"
        samples++
      }
      next
    }

    END {
      if (!seen) {
        print "FAIL no comparable lines found"
        exit 1
      } else if (mismatch==0) {
        print "PASS"
        exit 0
      } else {
        if (worst_i=="") { worst_i=""; worst_a=0; worst_b=0 }
        printf("FAIL %d mismatches (eps=%s). max|Δ|=%.6g at i=%s (exp=%.8g got=%.8g)\n",
               mismatch, eps, worst, worst_i, worst_a, worst_b)
        exit 1
      }
    }
  ' 2> >(sed "s/^/    /") <<<"$report"
}

mkdir -p "/tmp/$USER"

pass_count=0
fail_count=0

for i in $(seq -w 1 40); do
  echo "${BLD}===== Testcase #$i =====${CLR}"
  nodes=$(jq -r '.nodes' "testcases/$i.txt")
  procs=$(jq -r '.procs' "testcases/$i.txt")
  timelimit=$(jq -r '.time' "testcases/$i.txt")
  N=$(jq -r '.n' "testcases/$i.txt")
  input_path="testcases/$i.in"
  sample_out="testcases/$i.out"
  output_path="./$USER-$i.out"

  echo "Running: n=$N, world_size=$procs (time limit ${timelimit}s)"
  # Run the program; capture both stdout and stderr from the timed run.
  # (time builtin returns the command’s exit code; no need for '|| true')
  run_log=$(mktemp)
  { time -p srun -N "$nodes" -n "$procs" -t "0:$timelimit" /home/pp25/pp25s121/Parallel-Computing/hw1/hw1 "$N" "$input_path" "$output_path"; } \
      >"$run_log".out 2>"$run_log".err
  rc=$?

  # Give parallel FS a moment to flush metadata/data before reading output
  sleep 2   # 200ms pause (tune if needed)

  if [[ $rc -ne 0 ]]; then
    echo "${RED}RESULT: FAIL (program exit code $rc)${CLR}"
    echo "---- stdout ----"
    sed 's/^/    /' "$run_log".out | head -n 50
    echo "---- stderr (includes time) ----"
    sed 's/^/    /' "$run_log".err | head -n 50
    ((fail_count++))
    rm -f "$output_path" "$run_log".out "$run_log".err
    continue
  fi

  # Compare outputs numerically with tolerance via hw1-floats feed
  status="$(check_with_hw1_floats "$sample_out" "$output_path" "$N")"
  cmp_rc=$?

  if [[ $cmp_rc -eq 0 ]]; then
    echo "${GRN}RESULT: PASS${CLR}"
    ((pass_count++))
  else
    echo "${RED}RESULT: ${status}${CLR}"
    ((fail_count++))
  fi

  rm -f "$output_path" "$run_log".out "$run_log".err
done

echo
echo "============================"
echo "SUMMARY: ${GRN}${pass_count} passed${CLR}, ${RED}${fail_count} failed${CLR}"