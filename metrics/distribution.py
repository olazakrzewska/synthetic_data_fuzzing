import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


def wasserstein_per_feature(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numerical_cols: list[str]
) -> dict[str, float]:
    distances = {}

    for col in numerical_cols:
        distances[col] = wasserstein_distance(
            real[col].values,
            synthetic[col].values
        )

    return distances


def distributional_stability_index(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numerical_cols: list[str]
) -> float:
    dists = wasserstein_per_feature(real, synthetic, numerical_cols)
    return float(np.mean(list(dists.values())))
