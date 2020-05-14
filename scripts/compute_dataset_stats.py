"""
Custom lidbox pipeline script for computing stats on the dataset.
"""
def modify_steps(steps, split, labels, init_data, config):
    from lidbox.dataset.steps import Step
    # Find the step where VAD decisions are added
    i = [i for i, s in enumerate(steps) if s.key == "compute_rms_vad"][-1]
    # Compute frequency of dropped and kept frames by the VAD decisions
    steps.insert(i + 1, Step("reduce_stats", {"statistic": "vad_ratio"}))
    return steps[:i + 2]
