import pandas as pd

from src.datasets.base import BaseDataset


class ParquetLoader(BaseDataset):

    def load(self):

        return pd.read_parquet(self.config.path)