import logging

from langchain_redis.cache import RedisSemanticCache
from langchain.globals import set_llm_cache

from backend.services.embedding_service import EmbeddingService
from backend.config import settings

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self):
        self.redis_url = settings.redis_url
        self.ttl = settings.redis_cache_ttl
        self.distance_threshold = settings.redis_distance_threshold

        embedding_service = EmbeddingService()
        self.embeddings = embedding_service.embeddings

        self.semantic_cache = RedisSemanticCache(
            redis_url=self.redis_url,
            embeddings=self.embeddings,
            distance_threshold=self.distance_threshold,
            ttl=self.ttl
        )

        set_llm_cache(self.semantic_cache)
        logger.info("Semantic cache initialized")

    def clear_cache(self):
        try:
            self.semantic_cache.clear()
            logger.info("Semantic cache cleared")
            return True

        except Exception as e:
            logger.error(f"Error clearing cache: {str(e)}")
            return False
