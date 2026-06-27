import re


class TextCleaner:

    _spaces = re.compile(r"\s+")

    def clean(self, text: str) -> str:

        if text is None:
            return ""

        text = text.lower()

        text = self._spaces.sub(" ", text)

        return text.strip()