#!/usr/bin/env bash
set -ue
experiment_dir=
source scripts/env.bash
echo "collecting C_avg results with scripts/collect_results.py"
python3 scripts/collect_results.py $experiment_dir $experiment_dir/out
