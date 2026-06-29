import os
import faiss
import numpy as np
from typing import Tuple, List


class FaissIndex:
    def __init__(self, dimension: int, index_type: str = "flat_ip"):
        """
        Инициализация FAISS индекса.
        :param dimension: Размерность эмбеддингов (например, 768 или 1024)
        :param index_type: Тип индекса ('flat_ip' для Inner Product / Cosine, 'flat_l2' для евклидова расстояния)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self._init_index()

    def _init_index(self):
        """Фабрика для создания конкретного типа индекса FAISS"""
        if self.index_type == "flat_ip":
            # Используем Inner Product. Если эмбеддинги нормированы, это эквивалентно Cosine Similarity
            self.index = faiss.IndexFlatIP(self.dimension)
        elif self.index_type == "flat_l2":
            self.index = faiss.IndexFlatL2(self.dimension)
        else:
            raise ValueError(f"Неподдерживаемый тип индекса: {self.index_type}")

    def build(self, embeddings: np.ndarray):
        """
        Загрузка эмбеддингов в индекс.
        :param embeddings: Массив numpy формы (num_documents, dimension)
        """
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Если используем Inner Product, хорошей практикой является нормализация векторов
        if self.index_type == "flat_ip":
            faiss.normalize_L2(embeddings)

        self.index.add(embeddings)

    def search(self, query_embeddings: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Поиск по индексу.
        :param query_embeddings: Эмбеддинги запросов формы (num_queries, dimension)
        :param top_k: Количество ближайших соседей
        :return: Кортеж из (scores, indices)
        """
        if query_embeddings.dtype != np.float32:
            query_embeddings = query_embeddings.astype(np.float32)

        if self.index_type == "flat_ip":
            faiss.normalize_L2(query_embeddings)

        # FAISS возвращает расстояния/скоры и ID документов
        scores, indices = self.index.search(query_embeddings, top_k)
        return scores, indices

    def save(self, path: str):
        """Сохранение индекса на диск"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        faiss.write_index(self.index, path)
        print(f" FAISS индекс успешно сохранен в: {path}")

    def load(self, path: str):
        """Загрузка индекса с диска"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл индекса не найден: {path}")
        self.index = faiss.read_index(path)
        print(f" FAISS индекс успешно загружен из: {path}")