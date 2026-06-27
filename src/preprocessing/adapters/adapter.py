from datasets import Dataset

from src.preprocessing.models.sample import Sample


class MSMarcoAdapter:
    """
    Преобразует датасет MS MARCO
    в список обучающих примеров.
    """

    def convert(self, dataset: Dataset) -> list[Sample]:

        samples = []

        for row in dataset:

            passages = row["passages"]["passage_text"]
            selected = row["passages"]["is_selected"]

            samples.append(
                Sample(
                    query_id=row["query_id"],
                    query=row["query"],
                    answers=row["answers"],
                    passages=passages,
                    selected=selected,
                )
            )

        return samples