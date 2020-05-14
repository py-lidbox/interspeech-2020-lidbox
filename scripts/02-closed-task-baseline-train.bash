#!/usr/bin/env bash
# Train the baseline models of three datasets
set -ue

experiment_dir=
source scripts/env.bash
source scripts/utils.bash

config_files=(
models/ap19-olr/config.ap19olr-baseline.yaml
models/mgb3/config.mgb3-baseline.yaml
models/dosl/config.dosl-baseline.yaml
)
config_files=( "${config_files[@]/#/$experiment_dir/}" )

echo 'launching training tasks on slurm'

for config in ${config_files[*]}; do
    jobname=$(config_to_jobname $config)-train
    sbatch \
        --job-name=$jobname \
        --output=$experiment_dir/logs/${jobname}.out \
        --error=$experiment_dir/logs/${jobname}.err \
        --time=01-00 \
        --gres=gpu:1 \
        --mem=32G \
        $experiment_dir/scripts/lidbox-run.bash e2e $config
done
