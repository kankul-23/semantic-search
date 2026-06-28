import json
from pathlib import Path
from src.config.config import Config
from src.core.document import Document
from src.indexing.dense_index import DenseIndexer


def main():
    root = Path(__file__).resolve().parent.parent
    config = Config(root)

    # 1. Читаем документы
    docs_file = config.root / "data" / "processed" / config.dataset.name / "documents.jsonl"
    if not docs_file.exists():
        raise FileNotFoundError(f"Processed documents not found at {docs_file}. Run prepare_dataset.py first.")

    documents = []
    print(f"Loading documents from {docs_file}...")
    with open(docs_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            documents.append(Document(id=data["id"], text=data["text"]))

    # 2. Инициализируем плотный индексатор
    indexer = DenseIndexer(config.model)

    # 3. Генерируем векторы (эмбеддинги)
    embeddings = indexer.build(documents)

    # 4. Сохраняем результат
    index_dir = config.root / config.model.embeddings_path
    indexer.save(embeddings, index_dir)
    print("Dense Indexing step successfully completed!")


if __name__ == "__main__":
    main()