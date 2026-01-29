import pandas as pd
import numpy as np

BINS = 10
EPS = 0.1

METRICS = ["DSI", "CPR", "FSI"]

FULL_MUT_FILE = "../results/full_mutation_results.csv"
FUZZ_FILE = "../results/fuzzing_results.csv"

OUT_FILE = "../results/coverage_analysis.csv"


def region_coverage(values, bins=BINS):
    hist, _ = np.histogram(values, bins=bins)
    return np.count_nonzero(hist) / bins


def significant_coverage(df, eps=EPS):
    mask = (
        (df["DSI"] > eps) |
        (df["CPR"] > eps) |
        (df["FSI"] > eps)
    )
    return mask.mean()


def compute_coverage(single_df, fuzz_df):
    rows = []

    for m in METRICS:
        rows.append({
            "Metric": m,
            "Campaign": "SingleMutations",
            "RegionCoverage": region_coverage(single_df[m]),
            "SignificantCoverage": significant_coverage(single_df),
        })

        rows.append({
            "Metric": m,
            "Campaign": "Fuzzing",
            "RegionCoverage": region_coverage(fuzz_df[m]),
            "SignificantCoverage": significant_coverage(fuzz_df),
        })

    return pd.DataFrame(rows)


def main():
    single = pd.read_csv(FULL_MUT_FILE)
    fuzz = pd.read_csv(FUZZ_FILE)

    single = single[single["Generator"] == "CTGAN"]

    coverage_df = compute_coverage(single, fuzz)
    coverage_df.to_csv(OUT_FILE, index=False)

    print("\nCoverage analysis:")
    print(coverage_df)
    print(f"\nSaved to {OUT_FILE}")


if __name__ == "__main__":
    main()
