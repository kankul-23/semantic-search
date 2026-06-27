from src.datasets.csv_loader import CSVLoader
from src.datasets.hf_loader import HFLoader
from src.datasets.parquet_loader import ParquetLoader


class DatasetFactory:

    LOADERS = {
        "hf": HFLoader,
        "huggingface": HFLoader,
        "csv": CSVLoader,
        "parquet": ParquetLoader,
    }

    @classmethod
    def create(cls, config):

        source = config.source.lower()

        if source not in cls.LOADERS:
            raise ValueError(
                f"Unknown dataset source '{source}'. "
                f"Supported: {', '.join(cls.LOADERS)}"
            )

        return cls.LOADERS[source](config)