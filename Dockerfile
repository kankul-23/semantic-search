# 1. Используем официальный легкий образ Python
FROM python:3.11-slim

# Устанавливаем системные зависимости (git нужен для работы с Hugging Face)
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Устанавливаем рабочую директорию внутри контейнера
WORKDIR /app

# Обновляем pip
RUN pip install --no-cache-dir --upgrade pip

# Сначала ставим облегченную CPU-версию PyTorch (весит ~150 МБ вместо 2 ГБ)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Копируем только pyproject.toml для установки остальных зависимостей
COPY pyproject.toml ./

# Устанавливаем все зависимости проекта из pyproject.toml.
# Флаг --no-deps гарантирует, что pip не попытается перекачать CUDA-версию torch поверх уже установленной.
RUN pip install --no-cache-dir --no-deps .

# Доставляем остальные библиотеки из секции dependencies твоего pyproject.toml
RUN pip install --no-cache-dir transformers sentence-transformers datasets evaluate huggingface-hub faiss-cpu numpy pandas pyarrow fsspec scikit-learn scipy loguru pydantic pydantic-settings tqdm PyYAML uvicorn fastapi

# Копируем всю структуру проекта в контейнер
COPY src/ ./src
COPY configs/ ./configs
COPY indexes/ ./indexes
COPY models/ ./models
COPY data/processed/ ./data/processed

# Открываем порт для FastAPI наружу
EXPOSE 8000

# Информируем код через переменную окружения, что работаем на CPU
ENV DEVICE=cpu

# Команда для запуска нашего FastAPI при старте контейнера
CMD ["python", "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]