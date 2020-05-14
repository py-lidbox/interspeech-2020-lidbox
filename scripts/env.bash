#!/usr/bin/env bash
# Load dependencies in the Triton compute cluster
set -ue
module load cuda/10.1.243
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WRKDIR}/lib/cuda/lib64
export PYTHONUNBUFFERED=1
experiment_dir=/m/triton/scratch/elec/puhe/p/lindgrm1/exp
