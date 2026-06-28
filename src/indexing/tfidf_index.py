import pickle
from pathlib import Path
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from src.config.config import ModelConfig
from src.core.document import Document


class TFIDFIndexer:
    """
    Класс для создания, обучения и сохранения разреженного (Sparse) TF-IDF индекса.
    """

    def __init__(self, config: ModelConfig):
        self.vectorizer = TfidfVectorizer(
            lowercase=config.sparse_lowercase,
            max_features=config.sparse_max_features,
            ngram_range=config.sparse_ngram_range,
            min_df=config.sparse_min_df
        )

    def build(self, documents: list[Document]) -> sp.csr_matrix:
        print(f"Extracting texts from {len(documents)} documents...")
        texts = [doc.text for doc in documents]

        print("Fitting TF-IDF Vectorizer and transforming documents...")
        tfidf_matrix = self.vectorizer.fit_transform(texts)
        print(f"TF-IDF Matrix created with shape: {tfidf_matrix.shape}")
        return tfidf_matrix

    def save(self, tfidf_matrix: sp.csr_matrix, model_dir: Path, index_dir: Path):
        model_dir.mkdir(parents=True, exist_ok=True)
        index_dir.mkdir(parents=True, exist_ok=True)

        vectorizer_path = model_dir / "vectorizer.pkl"
        with open(vectorizer_path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"Saved vectorizer to {vectorizer_path}")

        matrix_path = index_dir / "tfidf_matrix.npz"
        sp.save_npz(matrix_path, tfidf_matrix)
        print(f"Saved TF-IDF matrix to {matrix_path}")