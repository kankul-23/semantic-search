import sys
import os

# Добавляем корневую директорию проекта в PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pathlib import Path

from src.config.config import Config
from src.factories.dataset_factory import DatasetFactory


def main():

    root = Path(__file__).parent

    config = Config(root)

    loader = DatasetFactory.create(config.dataset)

    dataset = loader.load()

    print(dataset.features)

    print(dataset.features["passages"])

    row = dataset[0]

    print(len(row["passages"]["passage_text"]))
    print(row["passages"]["is_selected"])

if __name__ == "__main__":
    main()