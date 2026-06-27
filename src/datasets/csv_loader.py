import pandas as pd

from src.datasets.base import BaseDataset


class CSVLoader(BaseDataset):

    def load(self):

        return pd.read_csv(self.config.path)