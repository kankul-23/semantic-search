import json
import numpy as np
from src.config.config import Config
from src.embeddings.sentence_transformer import JointSentenceTransformer
from src.retrieval.base import BaseRetriever


class DenseRetriever(BaseRetriever):
    """
    Компонент семантического поиска на основе плотных эмбеддингов (Dense Retrieval).
    """

    def __init__(self, config: Config):
        self.config = config

        # 1. Инициализируем наш энкодер (модель SentenceTransformer)
        self.encoder = JointSentenceTransformer(config.model)

        # 2. Загружаем матрицу эмбеддингов с диска
        matrix_path = config.root / config.model.embeddings_path / "dense_embeddings.npy"
        print(f"Loading Dense embeddings from {matrix_path}...")
        self.embeddings = np.load(matrix_path)

        # 3. Загружаем реальные ID документов для маппинга результатов
        self.doc_ids = self._load_document_ids()

        print(f"DenseRetriever initialized successfully with {self.embeddings.shape[0]} embeddings.")

    def _load_document_ids(self) -> list[int]:
        docs_file = self.config.root / "data" / "processed" / self.config.dataset.name / "documents.jsonl"
        doc_ids = []
        with open(docs_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                doc_ids.append(data["id"])
        return doc_ids

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Семантический поиск по плотным векторам.
        """
        # Превращаем строку-запрос в вектор на GPU/CPU и переводим в numpy
        query_vector = self.encoder.encode([query]).cpu().numpy().flatten()

        # Считаем скалярное произведение (dot product) между вектором запроса (384,)
        # и матрицей документов (603766, 384). Результат — массив скоров (603766,)
        scores = np.dot(self.embeddings, query_vector)

        # Сортируем индексы по убыванию скора
        top_indices = scores.argsort()[:-top_k - 1:-1]

        # Формируем результат
        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
        ]
        return results