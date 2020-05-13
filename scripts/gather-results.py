"""
Gather all C_avg values from the metrics.json files for every model and write csv files.
"""
import argparse
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import lidbox.api

parser = argparse.ArgumentParser()
parser.add_argument("experiment_dir", type=str)
parser.add_argument("csv_out_dir", type=str)
args = parser.parse_args()

dataset2name = {
    "ap19-olr": "AP19-OLR",
    "mgb3": "MGB-3",
    "sbs": "SBS",
}
dataset_order = ("ap19-olr", "mgb3", "sbs")

config_files_by_model = [
    [
        "models/ap19-olr/config.baseline.yaml",
        "models/mgb3/config.ap19olr-baseline.yaml",
        "models/sbs/config.ap19olr-baseline.yaml",
    ],
    [
        "models/ap19-olr/config.mgb3-baseline.yaml",
        "models/mgb3/config.baseline.yaml",
        "models/sbs/config.mgb3-baseline.yaml",
    ],
    [
        "models/ap19-olr/config.sbs-baseline.yaml",
        "models/mgb3/config.sbs-baseline.yaml",
        "models/sbs/config.baseline.yaml",
    ],
    [
        "models/ap19-olr/config.spherespeaker.yaml",
        "models/mgb3/config.spherespeaker.yaml",
        "models/sbs/config.spherespeaker.yaml",
    ],
    [
        "models/ap19-olr/config.xvec-channeldropout.yaml",
        "models/mgb3/config.xvec-channeldropout.yaml",
        "models/sbs/config.xvec-channeldropout.yaml",
    ],
    [
        "models/ap19-olr/config.xvec-extended.yaml",
        "models/mgb3/config.xvec-extended.yaml",
        "models/sbs/config.xvec-extended.yaml",
    ],
    [
        "models/ap19-olr/config.xvec-2d.yaml",
        "models/mgb3/config.xvec-2d.yaml",
        "models/sbs/config.xvec-2d.yaml",
    ],
]
# CLOSED TASK BACKEND
# models/ap19-olr/config.baseline-backendNB.yaml
# models/ap19-olr/config.mgb3-baseline-backendNB.yaml
# etc
# OPEN TASK BACKEND
# models/ap19-olr/config.ap19olr-baseline-combined3backendNB.yaml
# models/ap19-olr/config.mgb3-baseline-combined3backendNB.yaml
# etc

def cavg_from_config(config):
    metrics = lidbox.api.load_metrics(config)
    cavg = next(m["result"] for m in metrics if "average_detection_cost" in m["name"])
    return cavg

data = {"closed-task-e2e": [], "closed-task-embed": [], "open-task-embed": []}
for model_num, config_files in enumerate(config_files_by_model, start=1):
    for v in data.values():
        v.append({"model_num": model_num})
    for config_file in config_files:
        dataset = config_file.split(os.sep)[1]
        config = lidbox.api.load_yaml(os.path.join(args.experiment_dir, config_file))
        data["closed-task-e2e"][-1][dataset] = cavg_from_config(config)
        config = lidbox.api.load_yaml(os.path.join(args.experiment_dir, config_file.replace(".yaml", "-backendNB.yaml")))
        data["closed-task-embed"][-1][dataset] = cavg_from_config(config)
        if "config.baseline" in config_file:
            config_file = config_file.replace("config.baseline", "config.{}-baseline".format(dataset.replace("-", '')))
        config = lidbox.api.load_yaml(os.path.join(args.experiment_dir, config_file.replace(".yaml", "-combined3backendNB.yaml")))
        data["open-task-embed"][-1][dataset] = cavg_from_config(config)

os.makedirs(args.csv_out_dir, exist_ok=True)
for task, results in data.items():
    csv_out_path = os.path.join(args.csv_out_dir, "{}-cavg.csv".format(task))
    print("writing '{}'".format(csv_out_path))
    with open(csv_out_path, "w") as f:
        print("Model", *[dataset2name[m] for m in dataset_order], sep=',', file=f)
        for result in results:
            model_num_str = format(result["model_num"], "d")
            cavg_by_dataset = [format(result[ds], ".3f") for ds in dataset_order]
            print(model_num_str, *cavg_by_dataset, sep=',', file=f)
