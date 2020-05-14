#!/usr/bin/env bash
# Executable wrapper for sbatch on Slurm, this has no other purpose
set -ue
subcommand=$1
config=$2
if [ $# -gt 2 ]; then
    kwargs="${@:3:$#}"
else
    kwargs='-v'
fi
lidbox $subcommand $config $kwargs
