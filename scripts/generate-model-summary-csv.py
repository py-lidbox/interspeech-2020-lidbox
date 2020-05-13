import argparse
import importlib
import os
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
import lidbox.api
import lidbox.models.keras_utils


parser = argparse.ArgumentParser()
parser.add_argument("csv_out_path", type=str)
parser.add_argument("experiment_dir", type=str)
args = parser.parse_args()

experiment2modelnum = {key: str(i) for i, key in enumerate((
    "ap19olr-baseline-combined3",
    "mgb3-baseline-combined3",
    "sbs-baseline-combined3",
    "embed512-combined3",
    "channeldropout50-combined3",
    "xvec-ext-combined3",
    "xvec-2d-combined3",
    ), start=1)}
config_prefix = os.path.join(args.experiment_dir, "models", "combined3")
config_files = [
    "config.ap19olr-baseline.yaml",
    "config.mgb3-baseline.yaml",
    "config.sbs-baseline.yaml",
    "config.spherespeaker.yaml",
    "config.xvec-channeldropout.yaml",
    "config.xvec-extended.yaml",
    "config.xvec-2d.yaml",
]

model2params = []
for config_file in config_files:
    print(config_file)
    config_path = os.path.join(config_prefix, config_file)
    config = lidbox.api.load_yaml(config_path)
    model_num = experiment2modelnum[config["experiment"]["name"]]
    keras_wrapper = lidbox.api.KerasWrapper.from_config(config)
    million_params = 1e-6 * keras_wrapper.count_params()
    model_module = importlib.import_module(
            lidbox.models.keras_utils.MODELS_IMPORT_PATH + config["experiment"]["model"]["key"])
    embedding_extractor = getattr(model_module, "as_embedding_extractor")(keras_wrapper.keras_model)
    embedding_dim = embedding_extractor.outputs[0].shape[1]
    model2params.append((model_num, million_params, embedding_dim))

with open(args.csv_out_path, "w") as f:
    print("Model", "$10^6$ params", "$K$", sep=',', file=f)
    for model_num, million_params, embedding_dim in model2params:
        print(model_num, format(million_params, ".1f"), embedding_dim, sep=',', file=f)
print("wrote '{}'".format(args.csv_out_path))
