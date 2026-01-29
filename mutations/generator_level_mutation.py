from abc import abstractmethod

import pandas as pd

from mutations.mutation import Mutation


class GeneratorLevelMutation(Mutation):
    def __init__(self, name):
        self.name=name

    def apply(self, generator):
        return self.mutate_generator(generator)

    @abstractmethod
    def mutate_generator(self):
        pass


class SeedMutation(GeneratorLevelMutation):
    def __init__(self, new_val,name):
        self.new_val = new_val
        self.name=name

    def mutate_generator(self,generator) -> pd.DataFrame:
        return generator.__class__(
            data=generator.data,
            seed=self.new_val,
            epochs=generator.epochs,
            constraints=generator.constraints
        )

class EpochMutation(GeneratorLevelMutation):
    def __init__(self, new_val,name):
        self.new_val = new_val
        self.name=name

    def mutate_generator(self,generator) -> pd.DataFrame:
        return generator.__class__(
            data=generator.data,
            seed=generator.seed,
            epochs=self.new_val,
            constraints=generator.constraints
        )
