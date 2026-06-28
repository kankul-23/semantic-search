import pickle
import json
from pathlib import Path
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
from src.config.config import Config
from src.retrieval.base import BaseRetriever


class TFIDFRetriever(BaseRetriever):
    """
    Компонент поиска на основе классической модели TF-IDF (Sparse Retrieval).
    """

    def __init__(self, config: Config):
        self.config = config

        # Загружаем пути из конфигурации
        model_dir = config.root / config.model.sparse_model_path
        index_dir = config.root / config.model.sparse_index_path

        # 1. Загружаем обученный векторизатор (словарь и веса IDF)
        vectorizer_path = model_dir / "vectorizer.pkl"
        with open(vectorizer_path, "wb" if not vectorizer_path.exists() else "rb") as f:
            self.vectorizer = pickle.load(f)

        # 2. Загружаем сохраненную разреженную матрицу документов
        matrix_path = index_dir / "tfidf_matrix.npz"
        self.tfidf_matrix = sp.load_npz(matrix_path)

        # 3. Нам также нужен маппинг индексов матрицы на реальные ID документов из MS MARCO.
        # Для этого мы прочитаем наш documents.jsonl, чтобы сопоставить порядковый номер строки с id.
        self.doc_ids = self._load_document_ids()

        print(f"TFIDFRetriever initialized successfully with {len(self.doc_ids)} documents.")

    def _load_document_ids(self) -> list[int]:
        """Вспомогательный метод для чтения реальных ID документов из сохраненного корпуса."""
        docs_file = self.config.root / "data" / "processed" / self.config.dataset.name / "documents.jsonl"
        doc_ids = []
        with open(docs_file, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                doc_ids.append(data["id"])
        return doc_ids

    def search(self, query: str, top_k: int = 10) -> list[tuple[int, float]]:
        """
        Поиск релевантных документов по TF-IDF.
        """
        # Превращаем входящий текстовый запрос в разреженный вектор (1, 50000)
        query_vector = self.vectorizer.transform([query])

        # Считаем косинусное сходство между вектором запроса и всеми 603 766 документами в матрице
        # Результат — массив скоров размера (1, 603766)
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()

        # Находим индексы топ-K элементов с самыми высокими значениями скоров
        # argsort сортирует по возрастанию, поэтому берем с конца [:-top_k-1:-1]
        top_indices = similarities.argsort()[:-top_k - 1:-1]

        # Формируем финальный список результатов: (реальный_id_документа, скор_смерти)
        results = [
            (self.doc_ids[idx], float(similarities[idx]))
            for idx in top_indices
        ]

        return results