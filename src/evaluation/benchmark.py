from typing import Any, Dict
import numpy as np  # <-- Обязательно проверяем наличие этого импорта!
from tqdm import tqdm
from src.retrieval.base import BaseRetriever
from src.evaluation.metrics import recall_at_k, reciprocal_rank


class Benchmark:
    """
    Класс для автоматизированного замера качества работы поисковых систем.
    """

    def __init__(self, evaluation_config: Any):
        # Передаем параметры из evaluation.yaml
        self.top_k_list = evaluation_config.top_k

    def evaluate(self, retriever: BaseRetriever, test_samples: list[dict]) -> Dict[str, float]:
        """
        Прогоняет тестовые сэмплы через ретривер и считает агрегированные метрики.
        """
        metrics_results = {f"Recall@{k}": [] for k in self.top_k_list}
        metrics_results["MRR"] = []

        print(f"Running evaluation for {retriever.__class__.__name__} on {len(test_samples)} queries...")

        for sample in tqdm(test_samples, desc="Evaluating"):
            query = sample["query"]
            relevant_ids = sample["relevant_ids"]

            # Запрашиваем максимум из списка top_k
            max_k = max(self.top_k_list)
            search_results = retriever.search(query, top_k=max_k)

            # Вытаскиваем только ID документов
            retrieved_ids = [doc_id for doc_id, _ in search_results]

            # Считаем Recall для каждого K
            for k in self.top_k_list:
                rec = recall_at_k(retrieved_ids, relevant_ids, k)
                metrics_results[f"Recall@{k}"].append(rec)

            # Считаем MRR
            rr = reciprocal_rank(retrieved_ids, relevant_ids)
            metrics_results["MRR"].append(rr)

        # Агрегируем результаты (вычисляем среднее по всем запросам)
        final_report = {}
        for metric_name, values in metrics_results.items():
            final_report[metric_name] = float(np.mean(values))

        return final_report