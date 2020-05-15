"""
Collect all C_avg values from the metrics.json files for every model and write csv files.
"""
import argparse
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import numpy as np
import lidbox.api

dataset2name = {
    "ap19-olr": "AP19-OLR",
    "mgb3": "MGB-3",
    "dosl": "DoSL",
}
dataset_order = ("ap19-olr", "mgb3", "dosl")
experiment_order = ("closed-task-e2e", "closed-task-embed", "open-task-embed")
config_files_by_model = [
    [
        "models/ap19-olr/config.ap19olr-baseline.yaml",
        "models/mgb3/config.ap19olr-baseline.yaml",
        "models/dosl/config.ap19olr-baseline.yaml",
    ],
    [
        "models/ap19-olr/config.mgb3-baseline.yaml",
        "models/mgb3/config.mgb3-baseline.yaml",
        "models/dosl/config.mgb3-baseline.yaml",
    ],
    [
        "models/ap19-olr/config.dosl-baseline.yaml",
        "models/mgb3/config.dosl-baseline.yaml",
        "models/dosl/config.dosl-baseline.yaml",
    ],
    [
        "models/ap19-olr/config.spherespeaker.yaml",
        "models/mgb3/config.spherespeaker.yaml",
        "models/dosl/config.spherespeaker.yaml",
    ],
    [
        "models/ap19-olr/config.xvec-channeldropout.yaml",
        "models/mgb3/config.xvec-channeldropout.yaml",
        "models/dosl/config.xvec-channeldropout.yaml",
    ],
    [
        "models/ap19-olr/config.xvec-extended.yaml",
        "models/mgb3/config.xvec-extended.yaml",
        "models/dosl/config.xvec-extended.yaml",
    ],
    [
        "models/ap19-olr/config.xvec-2d.yaml",
        "models/mgb3/config.xvec-2d.yaml",
        "models/dosl/config.xvec-2d.yaml",
    ],
]
# We assume names of the backend config files can be trivially generated from the above list
# E.g. closed task
# models/ap19-olr/config.ap19olr-baseline-backendNB.yaml
# models/ap19-olr/config.mgb3-baseline-backendNB.yaml
# etc, and open task
# models/ap19-olr/config.ap19olr-baseline-combined3backendNB.yaml
# models/ap19-olr/config.mgb3-baseline-combined3backendNB.yaml
# etc

def latex_bf(s):
    return r"\textbf{{{:s}}}".format(s)

def cavg_from_config(config):
    metrics = lidbox.api.load_metrics(config)
    cavg = next(m["result"] for m in metrics if "average_detection_cost" in m["name"])
    return cavg

def format_cavg(cavg, best, precision=3):
    format_str = ".{}f".format(precision)
    cavg_str = format(cavg, format_str)
    best_str = format(best, format_str)
    return latex_bf(cavg_str) if cavg_str == best_str else cavg_str

def collect_cavg(experiment_dir):
    data = {exp: [] for exp in experiment_order}
    for config_files in config_files_by_model:
        for v in data.values():
            v.append([])
        for config_file in config_files:
            dataset = config_file.split(os.sep)[1]
            data["closed-task-e2e"][-1].append({dataset: cavg_from_config(
                    lidbox.api.load_yaml(os.path.join(experiment_dir, config_file)))})
            data["closed-task-embed"][-1].append({dataset: cavg_from_config(
                    lidbox.api.load_yaml(os.path.join(experiment_dir, config_file.replace(".yaml", "-backendNB.yaml"))))})
            data["open-task-embed"][-1].append({dataset: cavg_from_config(
                    lidbox.api.load_yaml(os.path.join(experiment_dir, config_file.replace(".yaml", "-combined3backendNB.yaml"))))})
    return data

def write_experiment_results(experiment, results, csv_out_dir):
    assert (results.ndim == 2
            and results.shape[0] == len(config_files_by_model)
            and results.shape[1] == len(dataset_order)), "incorrect result dimensions, should be a matrix where rows = models and columns = datasets"
    if csv_out_dir != "/dev/stdout":
        csv_out_path = os.path.join(csv_out_dir, "{}-cavg.csv".format(experiment))
    else:
        csv_out_path = "/dev/stdout"
    print("writing '{}'".format(csv_out_path))
    with open(csv_out_path, "w") as f:
        print("Model", *[dataset2name[m] for m in dataset_order], "Avg", sep=',', file=f)
        min_model_avg = results.mean(axis=1).min()
        min_result = results.min(axis=0)
        for model_num, (result, model_avg) in enumerate(zip(results, results.mean(axis=1)), start=1):
            model_num_str = format(model_num, "d")
            cavg_by_dataset = [format_cavg(result[i], min_result[i]) for i, _ in enumerate(dataset_order)]
            print(model_num_str, *cavg_by_dataset, format_cavg(model_avg, min_model_avg), sep=',', file=f)
        print("Avg", *[format_cavg(results.mean(axis=0)[i], -1.0) for i, _ in enumerate(dataset_order)], '', sep=',', file=f)

def main(experiment_dir, csv_out_dir):
    experiment2results = collect_cavg(experiment_dir)
    lidbox.api.yaml_pprint(experiment2results)
    cavg = np.array(
            [[[results_by_model[i][ds]
               for i, ds in enumerate(dataset_order)]
              for results_by_model in experiment2results[exp]]
             for exp in experiment_order])
    assert len(experiment_order) == cavg.shape[0]
    print("cavg numpy", cavg, sep="\n")
    print("experiment means", cavg.mean(axis=0), sep="\n")
    print("dataset means", cavg.mean(axis=1), sep="\n")
    print("model means", cavg.mean(axis=2), sep="\n")
    if csv_out_dir != "/dev/stdout":
        os.makedirs(csv_out_dir, exist_ok=True)
    for experiment, results in zip(experiment_order, cavg):
        write_experiment_results(experiment, results, csv_out_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=str)
    parser.add_argument("csv_out_dir", type=str)
    args = parser.parse_args()
    main(args.experiment_dir, args.csv_out_dir)
