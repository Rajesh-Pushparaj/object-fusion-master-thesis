import numpy as np


def compute_cross_entropy_class_weights(cfg, max_num_output_objs, num_samples):
    class_counts = cfg["class_counts"]
    num_objs = (
        class_counts["truck"]
        + class_counts["car"]
        + class_counts["motorcycle"]
        + class_counts["bicycle"]
        + class_counts["pedestrian"]
    )
    class_counts = np.array(
        [
            class_counts["truck"],
            class_counts["car"],
            class_counts["motorcycle"],
            class_counts["bicycle"],
            class_counts["pedestrian"],
            num_samples * max_num_output_objs - num_objs,
        ]
    )

    if (
        "normalization" not in cfg
        or cfg["normalization"] is None
        or cfg["normalization"] == "none"
    ):
        return np.ones_like(class_counts)
    elif cfg["normalization"] == "INS":  # Inverse of Number of Samples
        weights_ins = 1 / class_counts
        weights_ins = weights_ins / np.sum(weights_ins)
        return weights_ins
    elif cfg["normalization"] == "ISNS":  # Inverse Square Root of Number of Samples
        weights_isns = 1 / np.sqrt(class_counts)
        weights_isns = weights_isns / np.sum(weights_isns)
        return weights_isns
    elif cfg["normalization"] == "ENS":  # Effective Number of Samples
        beta = cfg["beta"]
        weights_ens = (1 - beta) / (1 - np.power(beta, class_counts))
        weights_ens = weights_ens / np.sum(weights_ens)
        return weights_ens
    else:
        raise ValueError(
            f"Normalization method: {cfg['normalization']} not supported for class weights computation."
        )
