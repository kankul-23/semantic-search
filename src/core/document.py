from dataclasses import dataclass


@dataclass(slots=True)
class Document:
    id: int
    text: str