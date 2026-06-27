from src.preprocessing.models.sample import Sample
from src.preprocessing.cleaner import TextCleaner


class PreprocessingPipeline:

    def __init__(self):

        self.cleaner = TextCleaner()

    def process(self, samples: list[Sample]) -> list[Sample]:

        processed = []

        for sample in samples:

            # удаляем записи без ответов
            if not sample.answers:
                continue

            query = self.cleaner.clean(sample.query)

            passages = []
            selected = []

            seen = set()

            for passage, flag in zip(sample.passages, sample.selected):

                passage = self.cleaner.clean(passage)

                if not passage:
                    continue

                if passage in seen:
                    continue

                seen.add(passage)

                passages.append(passage)
                selected.append(flag)

            if not passages:
                continue

            sample.query = query
            sample.passages = passages
            sample.selected = selected

            processed.append(sample)

        return processed