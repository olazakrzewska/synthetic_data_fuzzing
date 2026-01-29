from abc import ABC, abstractmethod
from typing import Callable, List, Optional
import pandas as pd
from sdv.metadata import SingleTableMetadata


class Generator(ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        seed: Optional[int] = None,
        epochs: Optional[int] = None,
        constraints: Optional[List[Callable[[pd.Series], bool]]] = None
    ):
        self.data = data
        self.seed = seed
        self.epochs = epochs
        self.constraints = constraints or []

        self.metadata = SingleTableMetadata()
        self.metadata.detect_from_dataframe(data)
        self.metadata.validate()

        self.synthesizer = self.build_synthesizer()

    def fit(self) -> None:
        self.synthesizer.fit(self.data)

    def sample(self, n: int) -> pd.DataFrame:
        return self.synthesizer.sample(num_rows=n)

    def get_data(self, n: int) -> pd.DataFrame:
        synthetic = self.sample(n)

        if not self.constraints:
            return synthetic
        mask = synthetic.apply(
            lambda row: all(c(row) for c in self.constraints),
            axis=1
        )
        return synthetic[mask].reset_index(drop=True)

    @abstractmethod
    def build_synthesizer(self):
        pass
