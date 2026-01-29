This repository provides a Python framework for robustness evaluation of synthetic data generators using mutation testing and fuzzing. The framework supports multiple generators (Copula, CTGAN, TVAE), data- and parameter-level mutations, and robustness metricsincluding distributional stability and correlation preservation.

## Installation

pip install -r requirements.txt

## Quick Start

from generators.ctgan_generator import CTGANGenerator
from mutations.data_level_mutation import MeanMutation
import pandas as pd

df = pd.read_csv("data/adult_clean.csv")
gen = CTGANGenerator(df, seed=42, epochs=20)
gen.fit()

mutation = MeanMutation("age", 2.0, "mean_shift")
mutant = mutation.apply(gen)
mutant.fit()

synthetic = mutant.sample(1000)

## Reproducing Experiments

To reproduce the full mutation campaigns reported in the paper:

python experiments/run_full_mutation_campaign.py

To run fuzzing experiments:

python experiments/run_fuzzing_campaign.py

## Repository Structure

generators/        synthetic data generators
mutations/         mutation operators
metrics/           robustness metrics
fuzzing/           fuzzing engine
experiments/       scripts used to generate results in the paper



