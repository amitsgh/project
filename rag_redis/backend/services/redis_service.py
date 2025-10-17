import logging
from typing import List

from langchain.schema import Document
from langchain_redis import RedisVectorStore

from backend.services.embedding_service import EmbeddingService
from backend.config import settings

import redis

logger = logging.getLogger(__name__)

OUTPUT_DIR = "models"


class RedisService:
    def __init__(self):
        self.redis_url = settings.redis_url
        self.index_name = settings.index_name
        self.indexing = settings.indexing

        self.redis_client = redis.from_url(self.redis_url)

        embedding_service = EmbeddingService()
        self.embeddings = embedding_service.embeddings

        self.vector_store = RedisVectorStore(
            redis_url=self.redis_url,
            index_name=self.index_name,
            indexing_algorithm=self.indexing,
            embeddings=self.embeddings,
        )

    def add_documents(self, documents: List[Document]):
        try:
            self.vector_store.add_documents(documents)
            logger.info(f"Added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Error adding documents: {str(e)}")

    def similarity_search(self, query: str, top_k: int = 10) -> List[Document]:
        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            logger.info(f"Found {len(results)} similar documents")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []

    def similarity_search_with_score(self, query: str, top_k: int = 10) -> List[tuple]:
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            logger.info(f"Found {len(results)} similar documents with score")
            return results
        except Exception as e:
            logger.error(f"Error in similarity search with score: {str(e)}")
            return []

    def is_connected(self) -> bool:
        try:
            self.redis_client.ping()
            return True
        except:
            return False
