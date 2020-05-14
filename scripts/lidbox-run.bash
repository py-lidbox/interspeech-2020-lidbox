#!/usr/bin/env bash
# Wrapper code for the Aalto compute cluster
# If you don't need to load the dependencies or do error checking, you can replace everything with just:
#   lidbox "$1" "$2" -v
set -ue

function print_env {
    echo "==== env begin ===="
    env | sort | uniq
    echo "===== env end ====="
    if [ ! -z "$(command -v module)" ]; then
        module list
    fi
    if [ ! -z "$(command -v nvidia-smi)" ]; then
        nvidia-smi
    fi
    echo
    echo "slurm job nodelist: $SLURM_JOB_NODELIST"
}

function echo_err {
    echo "$@" >> /dev/stderr
}

function env_has_error {
    local config="$1"
    local error=0
    if [ -z $(command -v lidbox) ]; then
        echo_err "command 'lidbox' not found, did you install the package?"
        error=1
    fi
    if [ ! -f $config ]; then
        echo_err "lidbox config file '$config' does not exist"
        error=1
    fi
    return $error
}

if [ $# -ne 2 ]; then
    echo_err "wrong number of args: $# out of 2"
    echo_err 'usage: lidbox-run.bash lidbox-subcommand config.yaml'
    exit 2
fi
subcommand="$1"
config=$(readlink --canonicalize-missing $2)

env_has_error $config || exit $?

module load cuda/10.1.243
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WRKDIR}/lib/cuda/lib64
export PYTHONUNBUFFERED=1
export LIDBOX_DEBUG=false
print_env >> /dev/stderr

cmd="lidbox $subcommand $config -v"
echo "running '$cmd'"
$cmd || exit $?
