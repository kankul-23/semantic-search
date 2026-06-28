import torch
from sentence_transformers import SentenceTransformer
from src.config.config import ModelConfig


class JointSentenceTransformer:
    """
    Компонент для генерации плотных эмбеддингов (Dense Embeddings) с использованием Hugging Face / Sentence Transformers.
    """

    def __init__(self, config: ModelConfig):
        self.config = config

        # Определяем девайс (cuda или cpu) автоматически, если в конфиге cuda, но её нет
        self.device = "cuda" if config.device == "cuda" and torch.cuda.is_available() else "cpu"
        print(f"Loading SentenceTransformer model '{config.model_name}' on {self.device}...")

        # Загружаем модель
        self.model = SentenceTransformer(config.model_name, device=self.device)

    def encode(self, texts: list[str]) -> torch.Tensor:
        """
        Превращает список текстов в плотные векторы (эмбеддинги).
        """
        # Используем встроенный метод библиотеки sentence-transformers
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            show_progress_bar=self.config.show_progress_bar,
            normalize_embeddings=self.config.normalize_embeddings,
            convert_to_tensor=True
        )
        return embeddings