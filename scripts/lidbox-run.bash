#!/usr/bin/env bash
# Wrapper code for the Aalto compute cluster
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

if [ $# -ne 2 -a $# -ne 3 ]; then
    echo_err "wrong number of args: $#"
    echo_err 'usage: lidbox-run.bash config.yaml lidbox-subcommand [--lidbox-kwargs ...]'
    exit 2
fi

config=$(readlink --canonicalize-missing $1)
subcommand="$2"
if [ $# -eq 3 ]; then
    kwargs="$3"
else
    kwargs=" "
fi

env_has_error $config || exit $?

module load cuda/10.1.243
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${WRKDIR}/lib/cuda/lib64
export PYTHONUNBUFFERED=1
export LIDBOX_DEBUG=false
print_env >> /dev/stderr

cmd="lidbox $subcommand $config $kwargs"
echo "running '$cmd'"
$cmd || exit $?
