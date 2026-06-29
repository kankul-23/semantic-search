from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------- Dataset ----------------

@dataclass
class DatasetConfig:
    source: str

    name: str = ""
    path: Optional[str] = None

    config_name: Optional[str] = None
    split: str = "train"

    text_column: str = "text"
    query_column: Optional[str] = None
    id_column: Optional[str] = None

    sample_size: Optional[int] = None

    cache: bool = True
    cache_dir: str = "data/cache"


# ---------------- Model ----------------

@dataclass
class ModelConfig:
    # поля для Dense
    model_name: str
    device: str
    batch_size: int
    normalize_embeddings: bool
    show_progress_bar: bool
    save_embeddings: bool
    embeddings_path: str

    # поля для TF-IDF
    sparse_lowercase: bool = True
    sparse_max_features: int = 50000
    sparse_min_df: int = 2
    sparse_ngram_range: list = r"[(1, 1)]"
    sparse_model_path: str = "models/sparse"
    sparse_index_path: str = "indexes/tfidf"

    # поля для FAISS
    faiss_index_type: str = "flat_ip"
    faiss_index_path: str = "indexes/faiss/faiss_index.bin"

    # поля для Гибридного поиска (RRF) ─── NEW ───
    rrf_k: int = 60
    hybrid_depth_k: int = 100

    # преобразование списка из yaml в кортеж для sklearn
    def __post_init__(self):
        if isinstance(self.sparse_ngram_range, list):
            self.sparse_ngram_range = tuple(self.sparse_ngram_range)


# ---------------- Evaluation ----------------

@dataclass
class EvaluationConfig:
    top_k: list
    metrics: list
    relevance_threshold: int
    normalize_scores: bool
    save_results: bool
    results_path: str


# ---------------- API ----------------

@dataclass
class APIConfig:
    host: str
    port: int
    reload: bool
    log_level: str
    default_top_k: int
    max_query_length: int
    timeout_seconds: int


# ---------------- Logging ----------------

@dataclass
class LoggingConfig:
    level: str
    format: str
    save_logs: bool
    log_file: str
    rotation: str
    retention: str
    compression: str


# ---------------- Main config ----------------

class Config:

    def __init__(self, root):

        self.root = Path(root)

        self.dataset = DatasetConfig(
            **load_yaml(self.root / "configs/dataset.yaml")
        )

        self.model = ModelConfig(
            **load_yaml(self.root / "configs/model.yaml")
        )

        self.evaluation = EvaluationConfig(
            **load_yaml(self.root / "configs/evaluation.yaml")
        )

        self.api = APIConfig(
            **load_yaml(self.root / "configs/api.yaml")
        )

        self.logging = LoggingConfig(
            **load_yaml(self.root / "configs/logging.yaml")
        )