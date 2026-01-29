import pandas as pd


def feasibility_violation_rate(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
) -> float:
    if len(real) == 0:
        return 0.0

    return 1.0 - (len(synthetic) / len(real))
