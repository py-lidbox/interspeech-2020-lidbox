#!/usr/bin/env bash
# Load dependencies inside the Triton compute cluster
# You should update all of these to match your platform
set -ue

# CUDA toolkit
module load cuda/10.1.243
# cuDNN library
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WRKDIR}/lib/cuda/lib64
# Flush printed data to logs without buffering
export PYTHONUNBUFFERED=1
# Root directory for the experiments repository
experiment_dir=/m/triton/scratch/elec/puhe/p/lindgrm1/exp
