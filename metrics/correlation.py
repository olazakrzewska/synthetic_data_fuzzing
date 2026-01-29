import numpy as np
import pandas as pd


def correlation_matrix(
    df: pd.DataFrame,
    numerical_cols: list[str]
) -> pd.DataFrame:
    return df[numerical_cols].corr(method="pearson")


def correlation_preservation_robustness(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    numerical_cols: list[str]
) -> float:
    R_real = correlation_matrix(real, numerical_cols)
    R_syn = correlation_matrix(synthetic, numerical_cols)

    diff = R_real.values - R_syn.values
    return float(np.linalg.norm(diff, ord="fro"))
