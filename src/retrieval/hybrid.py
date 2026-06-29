from collections import defaultdict
from typing import List, Tuple
from src.config.config import Config
from src.retrieval.tfidf import TFIDFRetriever
from src.indexing.faiss_index import FaissIndex
from src.embeddings.sentence_transformer import JointSentenceTransformer
from src.retrieval.base import BaseRetriever
import numpy as np


class HybridRetriever:
    """
    Гибридный поисковый ретривер, объединяющий лексический поиск (TF-IDF)
    и семантический поиск (Dense FAISS) с использованием алгоритма Reciprocal Rank Fusion (RRF).
    """

    def __init__(self, config: Config):
        super().__init__()  # Инициализируем родительский класс, если это необходимо
        self.config = config

        # Оставляем без изменений весь остальной код __init__...
        self.tfidf_retriever = TFIDFRetriever(config)
        self.dense_model = JointSentenceTransformer(config=config.model)
        self.faiss_index = FaissIndex(dimension=1024, index_type=config.model.faiss_index_type)
        self.faiss_index.load(config.model.faiss_index_path)
        self.rrf_k = getattr(config.model, "rrf_k", 60)
        self.depth_k = getattr(config.model, "hybrid_depth_k", 100)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """
        Выполняет гибридный поиск по запросу.

        Возвращает:
            Список кортежей [(doc_id, rrf_score), ...] отсортированный по убыванию.
        """
        # --- БЛОК 1: Лексический поиск (TF-IDF) ---
        # Предполагаем, что твой TFIDFRetriever возвращает списки кортежей [(doc_id, score), ...]
        tfidf_results = self.tfidf_retriever.search(query, top_k=self.depth_k)

        # --- БЛОК 2: Семантический поиск (Dense FAISS) ---
        query_vector = self.dense_model.encode([query]).detach().cpu().numpy()
        if len(query_vector.shape) == 1:
            query_vector = np.expand_dims(query_vector, axis=0)

        dense_scores, dense_indices = self.faiss_index.search(query_vector, top_k=self.depth_k)

        # Переводим результаты FAISS в удобный формат [(doc_id, score), ...]
        # Внимание: убедись, что ID документов в FAISS соответствуют реальным ID из MS MARCO
        dense_results = [
            (int(dense_indices[0][i]), float(dense_scores[0][i]))
            for i in range(len(dense_indices[0]))
            if dense_indices[0][i] != -1  # Фильтруем пустые результаты, если они есть
        ]

        # --- БЛОК 3: Reciprocal Rank Fusion (RRF) Слияние ---
        rrf_scores = defaultdict(float)

        # Считаем вклад TF-IDF (ранг начинается с 1)
        for rank, (doc_id, _) in enumerate(tfidf_results, start=1):
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)

        # Считаем вклад FAISS
        for rank, (doc_id, _) in enumerate(dense_results, start=1):
            rrf_scores[doc_id] += 1.0 / (self.rrf_k + rank)

        # Сортируем итоговый словарь по RRF скору в обратном порядке
        sorted_hybrid_results = sorted(
            rrf_scores.items(),
            key=lambda item: item[1],
            reverse=True
        )

        return sorted_hybrid_results[:top_k]