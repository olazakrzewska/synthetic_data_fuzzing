import pandas as pd
import numpy as np


INPUT_CSV = "../results/full_mutation_results.csv"
OUTPUT_CSV = "../results/aggregated_results.csv"

def sanity_checks(df: pd.DataFrame):
    errors = []

    nan_map = df.isna()

    mask_param = nan_map["Param"] & (df["Mutation"] != "Baseline")
    if mask_param.any():
        errors.append("NaN Param values detected outside Baseline")

    if nan_map["CPR"].any():
        print("[WARN] NaN values detected in CPR (likely zero-variance features)")

    for col in ["DSI", "FSI", "FVR"]:
        if nan_map[col].any():
            errors.append(f"NaN values detected in {col}")

    baseline = df[df["Mutation"] == "Baseline"]
    for metric in ["DSI", "CPR", "FSI", "FVR"]:
        if not np.allclose(baseline[metric], 0.0):
            errors.append(f"Baseline {metric} is not zero")

    monotonic_mutations = ["MeanShift", "VarianceScale", "EpochMutation"]

    for gen in df["Generator"].unique():
        gen_df = df[df["Generator"] == gen]

        for m in monotonic_mutations:
            sub = gen_df[gen_df["Mutation"].str.contains(m)]
            if len(sub) < 2:
                continue

            sub = sub.sort_values("Param")

            dsi_vals = sub["DSI"].values

            if not (np.all(np.diff(dsi_vals) >= -1e-6) or
                    np.all(np.diff(dsi_vals) <= 1e-6)):
                print(
                    f"[WARN] Non-monotonic DSI for {gen} / {m}"
                )

    if errors:
        raise ValueError("Sanity check failed:\n" + "\n".join(errors))



def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        df[df["Mutation"] != "Baseline"]
        .groupby(["Generator", "Mutation"])
        .agg(
            DSI_mean=("DSI", "mean"),
            DSI_median=("DSI", "median"),
            DSI_max=("DSI", "max"),
            CPR_mean=("CPR", "mean"),
            FSI_mean=("FSI", "mean"),
            FVR_mean=("FVR", "mean"),
        )
        .reset_index()
    )

    return agg



def main():
    df = pd.read_csv(INPUT_CSV)

    print("Running sanity checks...")
    sanity_checks(df)
    print("Sanity checks passed.")

    print("Aggregating results...")
    aggregated = aggregate_results(df)

    aggregated.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved aggregated results to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
