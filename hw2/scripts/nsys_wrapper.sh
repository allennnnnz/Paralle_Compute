#!/bin/bash

set -u

profile_path=$1
shift

# Output to ./nsys_reports/rank_$N.nsys-rep
nsys profile \
--force-overwrite=true \
-o "$profile_path/rank_$PMIX_RANK.nsys-rep" \
--mpi-impl openmpi \
--trace mpi,ucx,nvtx,osrt \
$@