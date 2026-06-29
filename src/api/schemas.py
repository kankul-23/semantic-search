from pydantic import BaseModel, Field
from typing import List, Optional

class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Поисковый запрос пользователя")
    top_k: int = Field(default=5, ge=1, le=100, description="Количество возвращаемых результатов")

class SearchResultItem(BaseModel):
    rank: int = Field(..., description="Порядковый номер в выдаче")
    doc_id: int = Field(..., description="Глобальный ID документа в корпусе")
    score: float = Field(..., description="Финальный комбинированный скор (RRF)")
    text: str = Field(..., description="Текст найденного пассажа/документа")

class SearchResponse(BaseModel):
    query: str = Field(..., description="Исходный поисковый запрос")
    results: List[SearchResultItem] = Field(..., description="Список найденных документов")
    total_found: int = Field(..., description="Сколько всего документов вернулось")