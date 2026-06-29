import os
import sys
import numpy as np

# Добавляем src в путь, чтобы импорты работали корректно
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config
from src.indexing.faiss_index import FaissIndex


def main():
    # 1. Инициализируем твой конфиг-менеджер (передаем текущую корневую директорию)
    config = Config(root=".")

    # Забираем пути и параметры из твоего ModelConfig
    embeddings_path = config.model.embeddings_path
    faiss_out_path = config.model.faiss_index_path
    faiss_type = config.model.faiss_index_type

    print(" Загрузка готовых плотных эмбеддингов...")
    if not os.path.exists(embeddings_path):
        print(f"❌ Ошибка: Файл {embeddings_path} не найден. Сначала сгенерируй эмбеддинги!")
        return

    embeddings = np.load(embeddings_path)
    num_docs, dimension = embeddings.shape
    print(f"Успешно загружено документов: {num_docs}, размерность: {dimension}")

    # 2. Инициализируем и строим FAISS индекс
    print(f" Сборка FAISS индекса (тип: {faiss_type})...")
    faiss_index = FaissIndex(dimension=dimension, index_type=faiss_type)
    faiss_index.build(embeddings)

    # 3. Сохраняем результат
    faiss_index.save(faiss_out_path)
    print(" Сборка FAISS индекса полностью завершена!")


if __name__ == "__main__":
    main()