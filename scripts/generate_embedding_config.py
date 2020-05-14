import argparse
import datetime
import os
import yaml


def generate_config_file(model_config, dataset_config, batch_size, dataset_key):
    required_keys = ("datasets", "features")
    output_config = {k: dataset_config[k] for k in required_keys}
    output_config["datasets"] = [d for d in output_config["datasets"] if d["key"] == dataset_key]
    # If pre_process defines 'cache', then these are redundant and we can use the existing cache,
    # otherwise we need to specify these to do signal pre-processing for creating embedding extractor input
    signal_processing_keys = ("pre_initialize", "post_initialize", "pre_process")
    for k in signal_processing_keys:
        if k in dataset_config:
            output_config[k] = dataset_config[k]
            if k == "pre_process" and "cache" in output_config[k]:
                output_config[k]["cache"]["consume"] = False
    # Some models use very short input
    if "post_process" in model_config and "chunks" in model_config["post_process"]:
        output_config["post_process"] = {"chunks": model_config["post_process"]["chunks"]}
    exp_conf = model_config["experiment"]
    output_config["embeddings"] = {
        "extractors": [{
            "experiment_name": exp_conf["name"],
            "best_checkpoint": {"mode": "min", "monitor": "val_loss"},
            "model": exp_conf["model"],
            "cache_directory": exp_conf["cache_directory"],
            "input_shape": exp_conf["input_shape"],
            "output_shape": exp_conf["output_shape"],
        }],
        "batch_size": batch_size,
        "no_unbatch": True}
    output_config["sklearn_experiment"] = {
        "cache_directory": dataset_config["experiment"]["cache_directory"],
        "name": '-'.join((
            exp_conf["model"]["key"],
            exp_conf["name"],
            "embed",
            dataset_key)),
        "model": {"key": "naive_bayes"},
        "data": {k: {split_k: exp_conf["data"][k][split_k] for split_k in ("split", "evaluate_metrics") if split_k in exp_conf["data"][k]} for k in ("train", "validation", "test")}}
    return output_config

def main(args):
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    output_config = generate_config_file(model_config, dataset_config, args.batch_size, args.dataset_key)
    os.makedirs(os.path.dirname(args.output_config), exist_ok=True)
    with open(args.output_config, mode="w", encoding="utf-8") as f:
        print("# auto-generated config file at {}".format(datetime.datetime.now()), file=f)
        print(yaml.dump(output_config), file=f, end='')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_config", type=str)
    parser.add_argument("dataset_config", type=str)
    parser.add_argument("output_config", type=str)
    parser.add_argument("dataset_key", type=str)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--skip-signals-cache", action="store_true")
    main(parser.parse_args())
