from pathlib import Path

from datasets import load_dataset

from src.datasets.base import BaseDataset


class HFLoader(BaseDataset):

    def load(self):

        kwargs = {
            "path": self.config.name,
            "split": self.config.split,
        }

        if self.config.config_name:
            kwargs["name"] = self.config.config_name

        if self.config.cache:
            cache_dir = Path(self.config.cache_dir)
            cache_dir.mkdir(parents=True, exist_ok=True)

            kwargs["cache_dir"] = str(cache_dir)
            kwargs["download_mode"] = "reuse_cache_if_exists"

        return load_dataset(**kwargs)