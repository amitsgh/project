import logging

from langchain_community.embeddings import HuggingFaceEmbeddings

from backend.config import settings

logger = logging.getLogger(__name__)


class EmbeddingService:
    _instance = None
    _embeddings = None

    model_name = None
    cache_folder = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
            cls._instance.model_name = settings.embedding_model
            cls._instance.cache_folder = settings.cache_folder

        return cls._instance

    @property
    def embeddings(self):
        if self._embeddings is None:
            logger.info("Initializing shared embedding model...")
            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.model_name,
                cache_folder=self.cache_folder,
            )
            logger.info("Embedding model Initialized")

        return self._embeddings

    def get_model_info(self):
        return {
            "model_name": self.model_name,
            "cache_folder": self.cache_folder,
            "is_initialized": self._embeddings is not None,
        }
