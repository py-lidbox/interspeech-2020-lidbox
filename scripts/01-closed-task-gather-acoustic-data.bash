#!/usr/bin/env bash
# Gather all acoustic data into three binary files
set -ue

experiment_dir=/m/triton/scratch/elec/puhe/p/lindgrm1/exp

source $experiment_dir/scripts/env.bash
source $experiment_dir/scripts/utils.bash

config_files=(
models/ap19-olr/config.prepare.yaml
models/mgb3/config.prepare.yaml
models/dosl/config.prepare.yaml
)
# Prepend experiment_dir to config file paths
config_files=( "${config_files[@]/#/$experiment_dir/}" )

mkdir -pv $experiment_dir/logs

for cf in ${config_files[*]}; do
    echo "checking '$(basename $cf)' with lidbox JSON schema"
    lidbox utils -v --validate-config-file $cf || exit $?
done

echo 'launching preparation tasks on slurm'

for config in ${config_files[*]}; do
    jobname=$(get_first_dataset_key $config)-cache-signals
    sbatch \
        --job-name=$jobname \
        --output=$experiment_dir/logs/${jobname}.out \
        --error=$experiment_dir/logs/${jobname}.err \
        --cpus-per-task=5 \
        --time=00-04 \
        --mem=64G \
        $experiment_dir/scripts/lidbox-run.bash prepare $config
done
