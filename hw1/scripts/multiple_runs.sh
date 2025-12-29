#!/usr/bin/env bash
REPORT_DIR="$1"
OUTPUT_AGG_CSV="$2"

# TAG="send_all_N1n12_split"
# EXECUTABLE="./hw1"
# DEFINES="-DSEND_ALL"

# TAG="send_min_max_N1n12_split"
# EXECUTABLE="./hw1"
# DEFINES="-DSEND_MIN_MAX"

TAG="send_required_N1n12_split"
EXECUTABLE="./hw1"
DEFINES="-DSEND_REQUIRED"

make DEFINES="${DEFINES}"

for i in {0..2..1}; do
     echo "Experiment #$i"
     srun -N 1 -n 12 ./wrapper/nsys_wrap.sh --testcase "33_${TAG}_${i}" ${EXECUTABLE} 536869888 /home/pp25/share/hw1/testcases/33.in ./testcases/33-mine.out

     REPORT_DIR="./nsys_reports/testcase_33_${TAG}_${i}/"

     # Find all .nsys-rep files under REPORT_DIR
     find "${REPORT_DIR}" -type f -name "*.nsys-rep" | while read -r repfile; do
          # derive a basename for the output csv
          base=$(basename "${repfile}" .nsys-rep)
          nsys stats --report nvtx_sum --format csv \
               --force-export=true \
               -o "${REPORT_DIR}/${base}" \
               "${repfile}"
          nsys stats --report mpi_event_sum --format csv \
               --force-export=true \
               -o "${REPORT_DIR}/${base}" \
               "${repfile}"
          nsys stats --report osrt_sum --format csv \
               --force-export=true \
               -o "${REPORT_DIR}/${base}" \
               "${repfile}"
          
          rm -rf "${repfile}"
     done

     # Delete all files end with .sqlite
     find "${REPORT_DIR}" -type f -name "*.sqlite" -exec rm -rf {} \;
done

make clean

echo "Done."