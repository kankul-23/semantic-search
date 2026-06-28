import json
from pathlib import Path
from src.config.config import Config
from src.retrieval.tfidf import TFIDFRetriever


def main():
    root = Path(__file__).resolve().parent.parent
    config = Config(root)

    print("Loading search engine (this may take a few seconds)...")
    retriever = TFIDFRetriever(config)

    # Загрузим оригинальные тексты документов, чтобы выводить их в консоль по ID
    docs_file = config.root / "data" / "processed" / config.dataset.name / "documents.jsonl"
    doc_mapping = {}
    with open(docs_file, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            doc_mapping[data["id"]] = data["text"]

    print("\n--- Search Engine Ready! Type 'exit' to quit. ---")
    while True:
        query = input("\nEnter your search query: ")
        if query.strip().lower() == "exit":
            break

        if not query.strip():
            continue

        results = retriever.search(query, top_k=3)

        print(f"\nTop 3 results for: '{query}':")
        for i, (doc_id, score) in enumerate(results, 1):
            print(f"\n[{i}] Document ID: {doc_id} | TF-IDF Score: {score:.4f}")
            print(f"Text: {doc_mapping.get(doc_id, 'Text not found')}")
            print("-" * 50)


if __name__ == "__main__":
    main()