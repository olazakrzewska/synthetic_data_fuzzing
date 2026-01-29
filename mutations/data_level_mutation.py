from abc import abstractmethod

import pandas as pd

from mutations.mutation import Mutation


class DataLevelMutation(Mutation):
    def __init__(self, column,name):
        self.column = column
        self.name=name

    def apply(self, generator):
        mutated_data = self.mutate_data(generator.data.copy())

        return generator.__class__(
            data=mutated_data,
            seed=generator.seed,
            epochs=generator.epochs,
            constraints=generator.constraints
        )

    @abstractmethod
    def mutate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        pass


class MeanMutation(DataLevelMutation):
    def __init__(self, column: str, delta: float,name:str):
        super().__init__(column,name)
        self.delta = delta

    def mutate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column in data.columns and pd.api.types.is_numeric_dtype(data[self.column]):
            data[self.column] = data[self.column] + self.delta
        return data

class VarianceScalingMutation(DataLevelMutation):
    def __init__(self, column: str, scale: float,name:str):
        super().__init__(column,name)
        self.scale = scale

    def mutate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column in data.columns and pd.api.types.is_numeric_dtype(data[self.column]):
            mean = data[self.column].mean()
            data[self.column] = mean + self.scale * (data[self.column] - mean)
        return data

class CorrelationBreakingMutation(DataLevelMutation):
    def __init__(self, column: str,on:int,name:str):
        super().__init__(column,name)
        self.on = on

    def mutate_data(self, data: pd.DataFrame) -> pd.DataFrame:
        if self.column in data.columns and self.on==1:
            data[self.column] = data[self.column].sample(frac=1).values
        return data

class RareValueRemovalMutation(DataLevelMutation):
    def __init__(self, column: str, threshold: float, name:str):
        super().__init__(column,name)
        self.threshold = threshold

    def mutate_data(self, data: pd.DataFrame):
        if self.column not in data.columns:
            return data

        value_freq = data[self.column].value_counts(normalize=True)
        rare_values = value_freq[value_freq < self.threshold].index

        filtered_data = data[~data[self.column].isin(rare_values)]

        if filtered_data.empty:
            return data

        return filtered_data.reset_index(drop=True)

