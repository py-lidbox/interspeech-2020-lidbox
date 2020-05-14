#!/usr/bin/env bash
# Train all non-baseline models on the union of all three datasets
set -ue

experiment_dir=
source scripts/env.bash
source scripts/utils.bash

config_files=(
models/combined3/config.ap19olr-baseline.yaml
models/combined3/config.mgb3-baseline.yaml
models/combined3/config.dosl-baseline.yaml
models/combined3/config.spherespeaker.yaml
models/combined3/config.xvec-2d.yaml
models/combined3/config.xvec-channeldropout.yaml
models/combined3/config.xvec-extended.yaml
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
        --constraint=volta \
        --gres=gpu:1 \
        --mem=32G \
        $experiment_dir/scripts/lidbox-run.bash e2e $config
done
