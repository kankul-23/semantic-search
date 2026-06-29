import json
import sys
import os
import time
from pathlib import Path

# Добавляем src в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config
from src.retrieval.hybrid import HybridRetriever


def main():
    root = Path(__file__).resolve().parent.parent
    config = Config(root)

    print("Инициализация гибридного движка поиска (TF-IDF + FAISS)...")
    start_init = time.time()
    hybrid_retriever = HybridRetriever(config)
    print(f"Движок готов! Время инициализации: {time.time() - start_init:.2f} сек")

    # Загружаем оригинальные тексты для вывода в консоль
    docs_file = config.root / "data" / "processed" / config.dataset.name / "documents.jsonl"
    doc_mapping = {}
    print("Загрузка маппинга документов...")
    with open(docs_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc_mapping[int(data["id"])] = data["text"]

    print("\n--- Гибридный поиск готов! Введи 'exit' для выхода. ---")
    while True:
        query = input("\nВведите поисковый запрос: ")
        if query.strip().lower() == "exit":
            break

        if not query.strip():
            continue

        start_search = time.time()
        results = hybrid_retriever.search(query, top_k=3)
        search_time = (time.time() - start_search) * 1000

        print(f"\n⏱️ Время гибридного поиска + RRF: {search_time:.2f} мс")
        print(f"Top 3 гибридных результата для: '{query}':")

        for i, (doc_id, rrf_score) in enumerate(results, 1):
            print(f"\n[{i}] ID Документа: {doc_id} | Итоговый RRF Скор: {rrf_score:.5f}")
            print(f"Текст: {doc_mapping.get(doc_id, 'Текст не найден в базе jsonl')}")
            print("-" * 60)


if __name__ == "__main__":
    main()