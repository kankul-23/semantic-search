from abc import ABC, abstractmethod


class BaseDataset(ABC):

    def __init__(self, config):
        self.config = config

    @abstractmethod
    def load(self):
        pass