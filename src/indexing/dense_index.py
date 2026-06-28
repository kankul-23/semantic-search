import numpy as np
from pathlib import Path
import torch
from src.config.config import ModelConfig
from src.core.document import Document
from src.embeddings.sentence_transformer import JointSentenceTransformer


class DenseIndexer:
    """
    Класс для генерации и сохранения плотного (Dense) индекса эмбеддингов.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.encoder = JointSentenceTransformer(config)

    def build(self, documents: list[Document]) -> np.ndarray:
        print(f"Extracting texts from {len(documents)} documents for Dense Indexing...")
        texts = [doc.text for doc in documents]

        print("Generating embeddings via SentenceTransformer (this may take time, grab a coffee!)...")
        # Переводим в CPU numpy array для удобства сохранения на диск
        with torch.no_grad():
            embeddings = self.encoder.encode(texts).cpu().numpy()

        print(f"Dense Index created with shape: {embeddings.shape}")
        return embeddings

    def save(self, embeddings: np.ndarray, index_dir: Path):
        index_dir.mkdir(parents=True, exist_ok=True)

        matrix_path = index_dir / "dense_embeddings.npy"
        np.save(matrix_path, embeddings)
        print(f"Saved Dense embeddings to {matrix_path}")