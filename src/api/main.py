from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config.config import Config  # Твой лоадер конфигурации
from src.services.search_service import SearchService
from src.api.routers import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- STARTUP ---
    print("🚀 Запуск FastAPI Semantic Search Service...")

    # Загружаем конфигурацию (передаем root=".", так как запускать будем из корня проекта)
    config = Config(root=".")

    # Хак путей: проверяем, строки ли там, если нет — кастим (как в ноутбуке)
    config.model.faiss_index_path = str(Path(config.root) / config.model.faiss_index_path)
    if config.model.embeddings_path.endswith("dense_embeddings.npy"):
        config.model.embeddings_path = "indexes/embeddings"

    # Инициализируем синглтон сервиса поиска
    app.state.search_service = SearchService(config)

    yield
    # --- SHUTDOWN ---
    print("🛑 Остановка сервиса поиска. Очистка ресурсов...")
    del app.state.search_service


app = FastAPI(
    title="Hybrid Semantic Search API",
    description="Продакшн API для гибридного поиска (TF-IDF + FAISS Dense v1.5) на датасете MS MARCO",
    version="0.8.0",
    lifespan=lifespan
)

# Настройка CORS для возможности подключения фронтенда
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключаем роуты
app.include_router(router)


@app.get("/health", tags=["System"])
async def health_check():
    return {"status": "healthy", "version": "0.8.0"}


if __name__ == "__main__":
    import uvicorn

    # Запускаем uvicorn локально на порту 8000
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)