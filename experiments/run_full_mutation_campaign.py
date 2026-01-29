import pandas as pd

from data.constants import numerical
from generators.copula_generator import CopulaGenerator
from generators.ctgan_generator import CTGANGenerator
from generators.tvae_generator import TVAEGenerator

from metrics.distribution import distributional_stability_index
from metrics.correlation import correlation_preservation_robustness
from metrics.feature import feature_level_stability_index
from metrics.feasibility import feasibility_violation_rate

from mutations.data_level_mutation import (
    MeanMutation,
    VarianceScalingMutation,
    CorrelationBreakingMutation
)
from mutations.generator_level_mutation import EpochMutation


GENERATORS = {
    "Copula": lambda df: CopulaGenerator(df),
    "CTGAN": lambda df: CTGANGenerator(df, seed=42, epochs=20),
    "TVAE": lambda df: TVAEGenerator(df, seed=42, epochs=20),
}


def run_full_mutation_campaign(
    df: pd.DataFrame,
    n_samples: int,
    output_csv: str = "../results/full_mutation_results.csv"
):
    rows = []

    for gen_name, gen_factory in GENERATORS.items():
        print(f"\n=== Generator: {gen_name} ===")

        base_gen = gen_factory(df)
        base_gen.fit()

        base_raw = base_gen.sample(n_samples)
        base_filtered = base_gen.get_data(n_samples)

        rows.append({
            "Generator": gen_name,
            "Mutation": "Baseline",
            "Param": None,
            "DSI": 0.0,
            "CPR": 0.0,
            "FSI": 0.0,
            "FVR": feasibility_violation_rate(base_raw, base_filtered)
        })

        for delta in [1, 2, 5]:
            mutation = MeanMutation("age", delta, f"MeanShift(age,{delta})")
            mutant = mutation.apply(base_gen)
            mutant.fit()

            raw = mutant.sample(n_samples)
            filtered = mutant.get_data(n_samples)

            rows.append({
                "Generator": gen_name,
                "Mutation": "MeanShift",
                "Param": delta,
                "DSI": distributional_stability_index(base_filtered, filtered, numerical),
                "CPR": correlation_preservation_robustness(base_filtered, filtered, numerical),
                "FSI": feature_level_stability_index(base_filtered, filtered, numerical),
                "FVR": feasibility_violation_rate(raw, filtered)
            })

        for scale in [0.5, 1.5, 2.0]:
            mutation = VarianceScalingMutation("age", scale, f"VarianceScale(age,{scale})")
            mutant = mutation.apply(base_gen)
            mutant.fit()

            raw = mutant.sample(n_samples)
            filtered = mutant.get_data(n_samples)

            rows.append({
                "Generator": gen_name,
                "Mutation": "VarianceScale",
                "Param": scale,
                "DSI": distributional_stability_index(base_filtered, filtered, numerical),
                "CPR": correlation_preservation_robustness(base_filtered, filtered, numerical),
                "FSI": feature_level_stability_index(base_filtered, filtered, numerical),
                "FVR": feasibility_violation_rate(raw, filtered)
            })

        mutation = CorrelationBreakingMutation("age", 1, "CorrelationBreak(age)")
        mutant = mutation.apply(base_gen)
        mutant.fit()

        raw = mutant.sample(n_samples)
        filtered = mutant.get_data(n_samples)

        rows.append({
            "Generator": gen_name,
            "Mutation": "CorrelationBreak",
            "Param": 1,
            "DSI": distributional_stability_index(base_filtered, filtered, numerical),
            "CPR": correlation_preservation_robustness(base_filtered, filtered, numerical),
            "FSI": feature_level_stability_index(base_filtered, filtered, numerical),
            "FVR": feasibility_violation_rate(raw, filtered)
        })

        for ep in [5, 50]:
            mutation = EpochMutation(ep, f"EpochMutation({ep})")
            mutant = mutation.apply(base_gen)
            mutant.fit()

            raw = mutant.sample(n_samples)
            filtered = mutant.get_data(n_samples)

            rows.append({
                "Generator": gen_name,
                "Mutation": "EpochMutation",
                "Param": ep,
                "DSI": distributional_stability_index(base_filtered, filtered, numerical),
                "CPR": correlation_preservation_robustness(base_filtered, filtered, numerical),
                "FSI": feature_level_stability_index(base_filtered, filtered, numerical),
                "FVR": feasibility_violation_rate(raw, filtered)
            })

    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved results to {output_csv}")

    return results_df


if __name__ == "__main__":
    df = pd.read_csv("../data/adult_clean.csv")
    run_full_mutation_campaign(df, n_samples=len(df))
