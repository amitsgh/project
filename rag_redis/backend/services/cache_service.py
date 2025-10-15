import logging

from langchain_redis.cache import RedisSemanticCache
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.globals import set_llm_cache

logger = logging.getLogger(__name__)


class CacheService:
    def __init__(self, redis_url: str, ttl: int = 3600):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.semantic_cache = RedisSemanticCache(
            redis_url=redis_url,
            embeddings=self.embeddings,
            distance_threshold=0.2,
            ttl=ttl
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
