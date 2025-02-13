import logging
import os
from typing import Dict, List

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from langchain.vectorstores import FAISS

from data.dataset import Dataset


class EmbeddingsManager:
    def __init__(
        self,
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        persist_directory: str = "db",
    ):
        """
        Ініціалізація менеджера ембедінгів

        Args:
            model_name (str): Назва моделі для створення ембедінгів
            persist_directory (str): Директорія для збереження FAISS індексу
        """
        # Налаштування логування
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Ініціалізація моделі ембедінгів
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={
                "device": "cuda" if os.environ.get("USE_CUDA") == "1" else "cpu"
            },
        )

        self.persist_directory = persist_directory

        # Створюємо директорію, якщо вона не існує
        os.makedirs(persist_directory, exist_ok=True)

    def prepare_documents(self, dataset: List[Dict]) -> List[Document]:
        """
        Підготовка документів з датасету для створення ембедінгів
        """
        documents = []
        for item in dataset:
            # Створюємо метадані для документа, перевіряючи наявність ключів
            metadata = {}

            # Список можливих метаданих
            metadata_fields = [
                "source_file",
                "length",
                "articles",
                "points",
                "page",
                "total_pages",
                "document_path",
            ]

            # Додаємо тільки ті метадані, які є в датасеті
            for field in metadata_fields:
                if field in item:
                    metadata[field] = str(item[field])

            # Перевіряємо наявність тексту
            if "text" not in item:
                self.logger.warning(f"Пропускаємо документ без тексту: {item}")
                continue

            # Створюємо Document з тексту та метаданих
            doc = Document(page_content=item["text"], metadata=metadata)
            documents.append(doc)

        return documents

    def create_vectorstore(self, dataset_path: str, file_type: str = "json") -> FAISS:
        """
        Створення векторної бази даних з датасету
        """
        # Завантаження датасету
        self.logger.info(f"Завантаження датасету з {dataset_path}")
        dataset = Dataset.load_dataset(dataset_path, file_type)

        # Підготовка документів
        self.logger.info("Підготовка документів")
        documents = self.prepare_documents(dataset)

        # Створення векторної бази даних
        self.logger.info("Створення векторного сховища")
        vectorstore = FAISS.from_documents(documents, self.embeddings)

        # Збереження бази
        index_path = os.path.join(self.persist_directory, "index.faiss")
        store_path = os.path.join(self.persist_directory, "store.pkl")
        vectorstore.save_local(self.persist_directory)

        self.logger.info(
            f"Векторне сховище створено та збережено в {self.persist_directory}"
        )

        return vectorstore

    def load_vectorstore(self) -> FAISS:
        """
        Завантаження існуючої векторної бази даних
        """
        index_path = os.path.join(self.persist_directory, "index.faiss")
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Векторне сховище не знайдено в {self.persist_directory}"
            )

        return FAISS.load_local(self.persist_directory, self.embeddings)

    def add_to_vectorstore(self, dataset_path: str, file_type: str = "json") -> None:
        """
        Додавання нових документів до існуючої векторної бази даних

        Args:
            dataset_path (str): Шлях до нового датасету
            file_type (str): Тип файлу датасету ('json', 'csv', 'parquet')
        """
        try:
            # Завантаження існуючої бази даних
            vectorstore = self.load_vectorstore()

            # Завантаження нового датасету
            self.logger.info(f"Завантаження нового датасету з {dataset_path}")
            new_dataset = Dataset.load_dataset(dataset_path, file_type)

            # Підготовка нових документів
            self.logger.info("Підготовка нових документів")
            new_documents = self.prepare_documents(new_dataset)

            # Додавання нових документів до бази
            self.logger.info("Додавання нових документів до векторного сховища")
            vectorstore.add_documents(new_documents)

            # Збереження оновленої бази
            vectorstore.save_local(self.persist_directory)
            self.logger.info(
                f"Векторне сховище оновлено та збережено в {self.persist_directory}"
            )

        except Exception as e:
            self.logger.error(f"Помилка при додаванні нових документів: {e}")
            raise
