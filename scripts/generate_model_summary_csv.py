import argparse
import importlib
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np
import lidbox.api
import lidbox.models.keras_utils


# Same order as enumerated in the paper
config_files = [
    "config.ap19olr-baseline.yaml",
    "config.mgb3-baseline.yaml",
    "config.dosl-baseline.yaml",
    "config.spherespeaker.yaml",
    "config.xvec-channeldropout.yaml",
    "config.xvec-extended.yaml",
    "config.xvec-2d.yaml",
]


def parse_epoch_seconds_from_logfile(experiment_dir, config):
    logname = "-".join((
        config["experiment"]["model"]["key"],
        config["experiment"]["name"],
        "train"))
    logfile = os.path.join(experiment_dir, "logs", logname + ".out")
    with open(logfile) as f:
        lines = [l.strip() for l in f]
    lines = [l for l in lines if l]
    epoch2line = [(lines[i-1], lines[i]) for i in range(1, len(lines)) if lines[i-1].startswith("Epoch")]
    seconds_per_epoch = np.array([int(l.split(" - ", 2)[1].rstrip('s')) for _, l in epoch2line], dtype=np.float32)
    logfile = os.path.join(experiment_dir, "logs", logname + ".err")
    with open(logfile) as f:
        lines = [l.strip() for l in f]
    device_line = [l for l in lines if "computeCapability" in l][0]
    return device_line.split("name: ", 1)[1], seconds_per_epoch


def iter_model_summaries(experiment_dir):
    config_prefix = os.path.join(experiment_dir, "models", "combined3")
    for model_num, config_file in enumerate(config_files, start=1):
        print(config_file)
        config_path = os.path.join(config_prefix, config_file)
        config = lidbox.api.load_yaml(config_path)
        keras_wrapper = lidbox.api.KerasWrapper.from_config(config)
        million_params = 1e-6 * keras_wrapper.count_params()
        model_module = importlib.import_module(
            lidbox.models.keras_utils.MODELS_IMPORT_PATH + config["experiment"]["model"]["key"])
        embedding_extractor = model_module.as_embedding_extractor(keras_wrapper.keras_model)
        embedding_dim = embedding_extractor.outputs[0].shape[1]
        device, epoch_seconds = parse_epoch_seconds_from_logfile(experiment_dir, config)
        yield (
            model_num,
            format(million_params, ".1f"),
            embedding_dim,
            format(epoch_seconds.mean()/60, ".1f"),
            device)

def main(experiment_dir, csv_out_path):
    seen_devices = set()
    with open(csv_out_path, "w") as f:
        print("Model", "$10^6$ params", "$D$", "mean min/epoch", sep=',', file=f)
        for summary in iter_model_summaries(experiment_dir):
            seen_devices.add(summary[-1])
            print(*summary[:-1], sep=',', file=f)
    print("wrote '{}'".format(csv_out_path))
    assert len(seen_devices) == 1, "Warning: multiple different devices in log files, the mean min/epoch might be meaningless, devices:\n{}".format("\n".join(seen_devices))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_out_path", type=str)
    parser.add_argument("experiment_dir", type=str)
    args = parser.parse_args()
    main(args.experiment_dir, args.csv_out_path)
