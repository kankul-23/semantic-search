from abc import ABC, abstractmethod
from typing import List, Tuple
from src.core.document import Document


class BaseRetriever(ABC):
    """
    Абстрактный базовый класс для всех поисковых систем (TF-IDF, Dense, Hybrid).
    """

    @abstractmethod
    def search(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Выполняет поиск по текстовому запросу.

        Возвращает:
            List[Tuple[int, float]]: Список кортежей (document_id, score),
                                     отсортированный по убыванию релевантности.
        """
        pass