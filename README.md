# x-vector comparison with lidbox

Source code for the experiments described in INTERSPEECH 2020 paper "Releasing a toolkit and comparing the performance of x-vector language embeddings across various spoken language identification datasets".

All experiments were performed by running [`lidbox`](https://github.com/matiaslindgren/lidbox) on the [Triton](https://scicomp.aalto.fi/index.html) compute cluster at [Aalto University](https://aalto.fi/en).
The workload manager for Triton is [Slurm](https://slurm.schedmd.com/documentation.html), and some wrapper code has been included for Slurm in this repository under `scripts`.
In case you want to run the experiments without Slurm, please see [this](https://github.com/matiaslindgren/lidbox/tree/master/examples/common-voice) example on how to run `lidbox` for a generic experiment.

## Notes before running

It is unlikely the experiments work right away and you probably need to make some fixes first.

* Fix acoustic data prefix `/m/teamwork/t40511_asr/c/` in all `utt2path` files under `data`.
* Make sure you have the acoustic data for all three datasets.
* Fix experiment directory `/m/triton/scratch/elec/puhe/p/lindgrm1/exp` or `/scratch/elec/puhe/p/lindgrm1/exp` in all yaml configuration files.
* Install TensorFlow 2 and `lidbox`.
