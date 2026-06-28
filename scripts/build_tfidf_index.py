import json
from pathlib import Path
from src.config.config import Config
from src.core.document import Document
from src.indexing.tfidf_index import TFIDFIndexer


def main():
    root = Path(__file__).resolve().parent.parent
    config = Config(root)

    # 1. Читаем обработанные документы
    docs_file = config.root / "data" / "processed" / config.dataset.name / "documents.jsonl"

    if not docs_file.exists():
        raise FileNotFoundError(f"Processed documents not found at {docs_file}. Run prepare_dataset.py first.")

    documents = []
    print(f"Loading documents from {docs_file}...")
    with open(docs_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            documents.append(Document(id=data["id"], text=data["text"]))

    # 2. Инициализируем индексатор конфигурацией модели
    indexer = TFIDFIndexer(config.model)

    # 3. Строим матрицу
    tfidf_matrix = indexer.build(documents)

    # 4. Определяем пути сохранения и записываем артефакты
    model_dir = config.root / config.model.sparse_model_path
    index_dir = config.root / config.model.sparse_index_path

    indexer.save(tfidf_matrix, model_dir, index_dir)
    print("TF-IDF Indexing step successfully completed!")


if __name__ == "__main__":
    main()