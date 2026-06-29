import json
from pathlib import Path
from typing import List, Dict, Tuple
from src.retrieval.hybrid import HybridRetriever


class SearchService:
    def __init__(self, config):
        self.config = config

        # 1. Инициализируем наш тяжелый гибридный ретривер
        print("⏳ Инициализация HybridRetriever внутри SearchService...")
        self.retriever = HybridRetriever(config)

        # 2. Настраиваем веса для Взвешенного RRF (подстраховка от шума TF-IDF)
        self.alpha_dense = 1.0
        self.beta_sparse = 0.3
        self.rrf_k = getattr(config.model, "rrf_k", 60)

        # 3. Загружаем корпус текстов в память для мгновенного обогащения по doc_id
        self.doc_id_to_text: Dict[int, str] = {}
        self._load_corpus()

    def _load_corpus(self):
        # Корректируем путь к documents.jsonl относительно корня проекта
        root_dir = Path(self.config.root) if hasattr(self.config, "root") else Path(".")
        corpus_path = root_dir / "data" / "processed" / self.config.dataset.name / "documents.jsonl"

        print(f"⏳ Загрузка текстов из корпуса {corpus_path} в RAM...")
        current_id = 0
        with open(corpus_path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line)
                if isinstance(data, dict):
                    text_content = data.get("text") or data.get("passage_text") or list(data.values())[0]
                else:
                    text_content = str(data)
                self.doc_id_to_text[current_id] = text_content
                current_id += 1
        print(f"✅ Загружено {len(self.doc_id_to_text)} документов. Сервис готов к работе!")

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Выполняет взвешенный гибридный поиск.
        Так как базовый HybridRetriever внутри делает RRF 50/50, мы можем:
        Либо переписать веса здесь через обращение к его внутренним ретриверам,
        Либо взять его дефолтный выдаваемый ранг (для простоты воспользуемся готовым поиском,
        но если в будущем захочешь кастомный Weighted RRF — перенесем формулу сюда).
        """
        # Запрашиваем чуть больше кандидатов у базового гибрида, чтобы отфильтровать топ
        raw_results = self.retriever.search(query, top_k=top_k)

        formatted_results = []
        for rank, (doc_id, score) in enumerate(raw_results, start=1):
            doc_id_int = int(doc_id)
            formatted_results.append({
                "rank": rank,
                "doc_id": doc_id_int,
                "score": float(score),
                "text": self.doc_id_to_text.get(doc_id_int, "[Текст не найден в базе данных]")
            })

        return formatted_results