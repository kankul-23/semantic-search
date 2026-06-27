import json
from pathlib import Path

from src.config.config import Config
from src.preprocessing.adapters.adapter import MSMarcoAdapter
from src.factories.dataset_factory import DatasetFactory
from src.preprocessing.pipeline import PreprocessingPipeline
from src.core.document import Document

root = Path(__file__).resolve().parent.parent
config = Config(root)

# 1. Загрузка данных
loader = DatasetFactory.create(config.dataset)
dataset = loader.load()

# 2. Адаптация и Препроцессинг
adapter = MSMarcoAdapter()
samples = adapter.convert(dataset)

pipeline = PreprocessingPipeline()
processed_samples = pipeline.process(samples)  # Здесь чистые объекты Sample

# Создаем директорию для результатов
base_processed_dir = config.root / "data" / "processed"

output_dir = base_processed_dir / config.dataset.name
output_dir.mkdir(parents=True, exist_ok=True)

samples_file = output_dir / "samples.jsonl"
documents_file = output_dir / "documents.jsonl"


# =====================================================================
# ШАГ 1: Сохраняем samples.jsonl (всё для обучения и оценки)
# =====================================================================
samples_file = output_dir / "samples.jsonl"

with open(samples_file, "w", encoding="utf-8") as f:
    for sample in processed_samples:
        json.dump(
            {
                "query_id": sample.query_id,
                "query": sample.query,
                "answers": sample.answers,
                "passages": sample.passages,
                "selected": sample.selected,
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

print(f"Saved {len(processed_samples)} processed samples to {samples_file}")

# =====================================================================
# ШАГ 2: Выделяем уникальные документы для FAISS и сохраняем documents.jsonl
# =====================================================================
documents_file = output_dir / "documents.jsonl"

seen_texts = set()
unique_documents = []
doc_idx = 0

for sample in processed_samples:
    for passage in sample.passages:
        if passage in seen_texts:
            continue

        seen_texts.add(passage)
        # Создаем экземпляр вашего класса Document
        unique_documents.append(Document(id=doc_idx, text=passage))
        doc_idx += 1

with open(documents_file, "w", encoding="utf-8") as f:
    for doc in unique_documents:
        json.dump(
            {
                "id": doc.id,
                "text": doc.text,
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

print(f"Saved {len(unique_documents)} unique documents to {documents_file}")