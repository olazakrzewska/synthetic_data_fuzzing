import numpy as np
import pandas as pd


def feature_level_stability_index(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numerical_cols: list[str],
    eps: float = 1e-8
) -> float:
    scores = []

    for col in numerical_cols:
        mu_r = real[col].mean()
        mu_s = synthetic[col].mean()

        std_r = real[col].std()
        std_s = synthetic[col].std()

        mean_diff = abs(mu_r - mu_s) / (abs(mu_r) + eps)
        std_diff = abs(std_r - std_s) / (std_r + eps)

        scores.append(mean_diff + std_diff)

    return float(np.mean(scores))
