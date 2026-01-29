import random
import pandas as pd

from data.constants import numerical
from generators.ctgan_generator import CTGANGenerator
from metrics.distribution import distributional_stability_index
from metrics.correlation import correlation_preservation_robustness
from metrics.feature import feature_level_stability_index
from metrics.feasibility import feasibility_violation_rate

from mutations.data_level_mutation import (
    MeanMutation,
    VarianceScalingMutation,
    CorrelationBreakingMutation,
)
from mutations.generator_level_mutation import EpochMutation


N_ITERATIONS = 150
MAX_SEQ_LEN = 4
N_SAMPLES = None

OUTPUT_CSV = "../results/fuzzing_results.csv"



def build_mutation_pool():
    return [
        lambda: MeanMutation("age", 1.0, "MeanShift(+1)"),
        lambda: MeanMutation("age", 2.0, "MeanShift(+2)"),
        lambda: VarianceScalingMutation("age", 1.5, "VarianceScale(1.5)"),
        lambda: CorrelationBreakingMutation("age", 1, "CorrelationBreak"),
        lambda: EpochMutation(50, "Epoch(+50)"),
    ]


def run_fuzzing(df):
    global N_SAMPLES
    N_SAMPLES = len(df)

    rows = []

    base_gen = CTGANGenerator(df, seed=42, epochs=20)
    base_gen.fit()

    base_filtered = base_gen.get_data(N_SAMPLES)

    mutation_pool = build_mutation_pool()

    for i in range(N_ITERATIONS):
        seq_len = random.randint(1, MAX_SEQ_LEN)
        mutation_fns = random.choices(mutation_pool, k=seq_len)

        gen = base_gen
        seq_names = []

        for fn in mutation_fns:
            mutation = fn()
            seq_names.append(mutation.name)
            gen = mutation.apply(gen)

        gen.fit()

        raw = gen.sample(N_SAMPLES)
        filtered = gen.get_data(N_SAMPLES)

        rows.append({
            "iteration": i,
            "generator": "CTGAN",
            "mutation_sequence": " -> ".join(seq_names),
            "DSI": distributional_stability_index(base_filtered, filtered, numerical),
            "CPR": correlation_preservation_robustness(base_filtered, filtered, numerical),
            "FSI": feature_level_stability_index(base_filtered, filtered, numerical),
            "FVR": feasibility_violation_rate(raw, filtered),
        })

        if (i + 1) % 10 == 0:
            print(f"Completed {i+1}/{N_ITERATIONS} fuzzing iterations")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved fuzzing results to {OUTPUT_CSV}")

    return df_out


if __name__ == "__main__":
    df = pd.read_csv("../data/adult_clean.csv")
    run_fuzzing(df)

