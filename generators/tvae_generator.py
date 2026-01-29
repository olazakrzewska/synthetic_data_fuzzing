from sdv.single_table import TVAESynthesizer

from generators.generator import Generator


class TVAEGenerator(Generator):

    def build_synthesizer(self):
        return TVAESynthesizer(
            self.metadata,
            epochs=self.epochs if self.epochs is not None else 5,
            batch_size=500
        )

    def fit(self):
        if self.seed is not None:
            import torch
            import numpy as np
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        super().fit()

