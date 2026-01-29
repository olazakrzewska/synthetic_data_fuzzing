from typing import List, Callable, Dict
import pandas as pd
from data.constants import numerical

from metrics.correlation import correlation_preservation_robustness
from metrics.distribution import distributional_stability_index
from metrics.feasibility import feasibility_violation_rate
from metrics.feature import feature_level_stability_index
from mutations.equivalance import is_equivalent


def is_structurally_changed(base_gen, mutant_gen):
    return (
            base_gen.seed != mutant_gen.seed
            or base_gen.epochs != mutant_gen.epochs
            or not base_gen.data.equals(mutant_gen.data)
            or base_gen.constraints != mutant_gen.constraints
    )


def is_equivalent(base_metrics, mutant_metrics, eps=1e-3):
    for k in base_metrics:
        if abs(base_metrics[k] - mutant_metrics[k]) > eps:
            return False
    return True
class FuzzingEngine:
    def __init__(
        self,
        base_generator,
        mutations: List,
        metrics: Dict[str, Callable]
    ):
        self.base_generator = base_generator
        self.mutations = mutations
        self.metrics = metrics



    def run(self, n_samples: int) -> pd.DataFrame:
        results = []

        self.base_generator.fit()
        base_raw = self.base_generator.sample(n_samples)
        base_filtered = self.base_generator.get_data(n_samples)

        base_metrics = {
            "DSI": 0.0,
            "CPR": 0.0,
            "FSI": 0.0,
            "FVR": feasibility_violation_rate(base_raw, base_filtered)
        }

        for mutation in self.mutations:
            mutant = mutation.apply(self.base_generator)

            if not is_structurally_changed(self.base_generator, mutant):
                results.append({
                    "mutation": mutation.name,
                    "status": "no_effect"
                })
                continue

            mutant.fit()
            raw = mutant.sample(n_samples)

            if raw.empty:
                results.append({
                    "mutation": mutation.name,
                    "status": "invalid_empty"
                })
                continue

            filtered = mutant.get_data(n_samples)

            metrics = {
                "DSI": distributional_stability_index(base_filtered, filtered, numerical),
                "CPR": correlation_preservation_robustness(base_filtered, filtered, numerical),
                "FSI": feature_level_stability_index(base_filtered, filtered, numerical),
                "FVR": feasibility_violation_rate(raw, filtered)
            }

            equivalent = is_equivalent(base_metrics, metrics)

            row = {
                "mutation": mutation.name,
                **metrics,
                "equivalent": equivalent,
                "status": "ok"
            }

            results.append(row)

        return pd.DataFrame(results)


