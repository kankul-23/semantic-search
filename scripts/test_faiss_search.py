import os
import sys
import time
import numpy as np

# Добавляем src в путь
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config.config import Config
from src.embeddings.sentence_transformer import JointSentenceTransformer
from src.indexing.faiss_index import FaissIndex


def main():
    print(" Инициализация конфигурации и загрузка моделей...")
    config = Config(root=".")

    # 1. Импортируем твой правильный класс
    from src.embeddings.sentence_transformer import JointSentenceTransformer
    model = JointSentenceTransformer(config=config.model)

    # 2. Инициализируем и загружаем собранный FAISS индекс
    faiss_index = FaissIndex(dimension=1024, index_type=config.model.faiss_index_type)
    faiss_index.load(config.model.faiss_index_path)

    # 3. Тестовый запрос
    query = "What is the capital of France?"
    print(f"\n Наш тестовый запрос: '{query}'")

    # Генерируем эмбеддинг для запроса и переводим в numpy для FAISS
    start_embed = time.time()
    query_vector = model.encode([query]).detach().cpu().numpy()

    if len(query_vector.shape) == 1:
        query_vector = np.expand_dims(query_vector, axis=0)
    print(f"⏱️ Время кодирования запроса: {(time.time() - start_embed) * 1000:.2f} мс")

    # 4. Ищем через FAISS
    top_k = 5
    start_search = time.time()
    scores, indices = faiss_index.search(query_vector, top_k=top_k)
    search_time = (time.time() - start_search) * 1000

    print(f"\n⏱️ Время поиска по 600k+ документам в FAISS: {search_time:.2f} мс")
    print("\n Результаты поиска (Топ-5 индексов документов и их скоры):")
    for i in range(top_k):
        print(f"Ранг {i + 1}: ID документа в базе = {indices[0][i]}, Скорость/Сходство = {scores[0][i]:.4f}")


if __name__ == "__main__":
    main()