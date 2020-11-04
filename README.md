# x-vector comparison with lidbox

Source code for the experiments described in INTERSPEECH 2020 paper "Releasing a toolkit and comparing the performance of language embeddings across various spoken language identification datasets".

All experiments were performed by running [`lidbox`](https://github.com/matiaslindgren/lidbox) on the [Triton](https://scicomp.aalto.fi/index.html) compute cluster at [Aalto University](https://aalto.fi/en).
The workload manager for Triton is [Slurm](https://slurm.schedmd.com/documentation.html), and some wrapper code has been included for Slurm in this repository under `scripts`.
In case you want to run the experiments without Slurm, or for some new dataset, please see [this](https://github.com/matiaslindgren/lidbox/tree/master/examples/common-voice) example on how to run `lidbox` for a generic experiment.


## Notes before running

It is unlikely the experiments work right away and you probably need to make some fixes first.

* Fix acoustic data prefix `/m/teamwork/t40511_asr/c/` in all `utt2path` files under `data`.
* Then make sure you have the acoustic data for all three datasets, e.g. by checking every path in every `utt2path` file.
* Fix experiment directory `/m/triton/scratch/elec/puhe/p/lindgrm1/exp` or `/scratch/elec/puhe/p/lindgrm1/exp` in all yaml configuration files. If you have cloned or downloaded this repository, its path is the experiment directory.
* Fix platform specific dependency loading in `scripts/env.bash`
* Install TensorFlow 2 and `lidbox v0.5.0`, for example:
```
pip install https://github.com/py-lidbox/lidbox/archive/v0.5.0.zip
```



## Recipe

Experiments can be reproduced on a Slurm cluster by running these numbered scripts from the `scripts` directory:

* `01-closed-task-gather-acoustic-data.bash` (must complete before other steps)
* `02-closed-task-baseline-train.bash`
* `03-closed-task-all-train.bash`
* `04-generate-backend-training-configs.bash`
* `05-closed-task-backend-train.bash`
* `06-open-task-combine-acoustic-data-caches.bash`
* `07-open-task-all-train.bash`
* `08-open-task-backend-train.bash`
* `09-collect-results.bash`


## If you do not have Slurm

In case you do not have Slurm, it **might** be enough to replace `sbatch` and all its arguments with simply `bash`.
E.g.
```
sbatch \
    --job-name=$jobname \
    --output=$experiment_dir/logs/${jobname}.out \
    --error=$experiment_dir/logs/${jobname}.err \
    --time=01-00 \
    --constraint=volta \
    --gres=gpu:1 \
    --mem=32G \
    $experiment_dir/scripts/lidbox-run.bash e2e $config
```
becomes
```
bash $experiment_dir/scripts/lidbox-run.bash e2e $config
```
Running the experiments sequentially like this will probably take several days.
