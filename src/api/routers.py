from fastapi import APIRouter, Depends, HTTPException, Query
from src.api.schemas import SearchRequest, SearchResponse, SearchResultItem
from src.services.search_service import SearchService

router = APIRouter(prefix="/api/v1", tags=["Search"])

# Эту функцию мы определим в main.py для раздачи синглтона сервиса
def get_search_service():
    from src.api.main import app
    return app.state.search_service

@router.get("/search", response_model=SearchResponse)
async def search_get(
    query: str = Query(..., min_length=1, description="Поисковый запрос"),
    top_k: int = Query(default=5, ge=1, le=100, description="Количество результатов"),
    service: SearchService = Depends(get_search_service)
):
    """
    Семантический гибридный поиск по корпусу документов (GET-запрос)
    """
    try:
        results = service.search(query=query, top_k=top_k)
        return SearchResponse(
            query=query,
            results=results,
            total_found=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения поиска: {str(e)}")

@router.post("/search", response_model=SearchResponse)
async def search_post(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service)
):
    """
    Семантический гибридный поиск по корпусу документов (POST-запрос с JSON-телом)
    """
    try:
        results = service.search(query=request.query, top_k=request.top_k)
        return SearchResponse(
            query=request.query,
            results=results,
            total_found=len(results)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка выполнения поиска: {str(e)}")