#!/bin/bash

mkdir -p exp_result

# Output to ./nsys_reports/rank_$N.nsys-rep
nsys profile \
-o "./exp_result/exp0_$PMIX_RANK.nsys-rep" \
--mpi-impl openmpi \
--trace mpi,ucx,nvtx,osrt \
$@