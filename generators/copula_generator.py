import numpy as np
import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer

from generators.generator import Generator


class CopulaGenerator(Generator):

    def build_synthesizer(self):
        return GaussianCopulaSynthesizer(self.metadata)

    def sample(self, n: int) -> pd.DataFrame:
        if self.seed is not None:
            np.random.seed(self.seed)
        return self.synthesizer.sample(n)
