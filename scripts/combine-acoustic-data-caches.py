"""
In order to ensure the open task is exactly the union of all closed task inputs, this script combines all acoustic data caches into a single cache.
"""
import argparse
import os
import tensorflow as tf
import logging

import lidbox.api
import lidbox.dataset.steps

logger = logging.getLogger("combine-script")

parser = argparse.ArgumentParser()
parser.add_argument("experiment_dir", type=str)
args = parser.parse_args()
expdir = args.experiment_dir

# Use baseline configs for dataset definitions
baseline_configs = [
    os.path.join(expdir, "models", dataset_key, "config.baseline.yaml")
    for dataset_key in ("ap19-olr", "mgb3", "sbs")]
combined_config = os.path.join(expdir, "models/combined3/config.ap19olr-baseline.yaml")

datasets = []
seen_labels = []
for conf_path in baseline_configs:
    split2meta, labels, config = lidbox.api.load_splits_from_config_file(conf_path)
    # Drop 'unknown' OOS mix from ap19-olr
    labels = [l for l in labels if l != "unknown"]
    seen_labels.extend(labels)
    # Dataset up to the step where acoustic data is cached
    del config["post_process"], config["features"], config["experiment"]
    config["pre_process"]["cache"]["consume"] = False
    datasets.append(lidbox.api.create_datasets(split2meta, labels, config))

logger.info("all datasets loaded")

_, all_labels, config = lidbox.api.load_splits_from_config_file(combined_config)
assert set(seen_labels) == set(all_labels), sorted(set(seen_labels) ^ set(all_labels))
label2int, _ = lidbox.api.make_label2onehot(all_labels)

def fix_targets(x):
    return dict(x, target=label2int.lookup(x["label"]))
def is_not_unknown(x):
    return x["label"] != tf.constant("unknown", tf.string)
def cache_ds(ds, split):
    c = config["pre_process"]["cache"]
    cache_kwargs = {
            "directory": os.path.join(c["directory"], "dataset", split),
            "batch_size": c["batch_size"],
            "cache_key": c["key"]}
    return lidbox.dataset.steps.cache(ds, **cache_kwargs)

logger.info("computing dataset sample ratios for weighting random sampling during merge")

# Create weights for each dataset depending on how many samples they have
# Assuming the source datasets are cached, we can iterate over them
dataset_weights = [{k: ds[k].reduce(0, lambda c, x: c + 1).numpy() for k in ("train", "test")} for ds in datasets]
num_total_elements = {k: sum(ds[k] for ds in dataset_weights) for k in ("train", "test")}
dataset_weights = [{k: ds[k] / num_total_elements[k] for k in ("train", "test")} for ds in dataset_weights]

logger.info("merging datasets")

# Create combined dataset iterators by sampling randomly from all 3 dataset iterators
# Use sample ratios as weights to avoid having a tail of samples from only one dataset
# Repeat separately for the training and test set
split2ds = {
    k: (tf.data.experimental.sample_from_datasets(
            [d[k] for d in datasets],
            weights=[d[k] for d in dataset_weights] if k == "train" else None)
        .filter(is_not_unknown)
        # Shuffling is a bit overkill here, it can be removed if too much RAM is being used
        .shuffle(int(2e5))
        .map(fix_targets)
        .apply(lambda ds: cache_ds(ds, k)))
    for k in ("train", "test")}

print_interval = 20000

# Evaluate the combined dataset iterators, causing data to be written into the cache
# We collect the unique paths just to see how many unique files ended up in the combined datasets
for k, ds in split2ds.items():
    unique_paths = set()
    for i, x in enumerate(split2ds[k].as_numpy_iterator(), start=1):
        unique_paths.add(x["path"].decode("utf-8"))
        if i % print_interval == 0:
            logger.info("%d done", i)
    logger.info("%s unique_paths: %d", k, len(unique_paths))
