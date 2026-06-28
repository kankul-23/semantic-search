import numpy as np


def recall_at_k(retrieved_ids: list[int], relevant_ids: list[int], k: int) -> float:
    """
    Вычисляет Recall@K.
    Возвращает 1.0, если хотя бы один релевантный документ попал в топ-K, иначе 0.0.
    (Для MS MARCO у каждого запроса обычно 1 строго релевантный документ в валидации).
    """
    # Берём только первые K результатов поисковика
    top_k_retrieved = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)

    # Находим пересечение (попали ли мы в цель)
    intersection = top_k_retrieved.intersection(relevant_set)

    return 1.0 if len(intersection) > 0 else 0.0


def reciprocal_rank(retrieved_ids: list[int], relevant_ids: list[int]) -> float:
    """
    Вычисляет Reciprocal Rank (RR) для одного запроса.
    Возвращает 1/позицию первого релевантного документа. Если релевантных документов нет в выдаче — 0.0.
    """
    relevant_set = set(relevant_ids)

    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank

    return 0.0