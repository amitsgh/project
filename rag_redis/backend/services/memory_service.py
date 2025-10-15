import logging
from typing import List

from langchain_redis import RedisChatMessageHistory
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import BaseMessage
from langchain.memory import ConversationSummaryBufferMemory

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"


class MemoryService:
    def __init__(
        self, redis_url: str, session_id: str, memory_key: str = "chat_history"
    ) -> None:
        self.redis_url = redis_url
        self.session_id = session_id
        self.memory_key = memory_key

        self._chat_history = RedisChatMessageHistory(
            redis_url=redis_url, session_id=session_id, ttl=1800  # 30 minutes
        )

        self._llm_service = None
        self._memory = None

    @property
    def llm_service(self):
        if self._llm_service is None:
            from backend.services.llm_service import LLMService
            from backend.config import settings

            base_url = settings.ollama_base_url
            model = settings.ollama_model

            self._llm_service = LLMService(base_url=base_url, model=model)
            logger.info(f"Initialized LLM service with model: {model}")

        return self._llm_service

    @property
    def memory(self):
        """Lazy initialization of memory"""
        if self._memory is None:
            self._memory = ConversationSummaryBufferMemory(
                llm=self.llm_service.llm,
                chat_memory=self._chat_history,
                max_token_limit=2000,
                return_messages=True,
                memory_key=self.memory_key,
            )
            logger.info(f"Initialized memory for session: {self.session_id}")

        return self._memory

    def add_message(self, message: BaseMessage) -> None:
        try:
            self._chat_history.add_message(message)
            logger.info(f"Added message for session {self.session_id}")
        except Exception as e:
            logger.error(
                f"Error adding message for session {self.session_id}: {str(e)}"
            )

    def get_message(self) -> List[BaseMessage]:
        try:
            return self._chat_history.messages
        except Exception as e:
            logger.error(
                f"Error getting message for session {self.session_id}: {str(e)}"
            )
            return []

    def clear_memory(self) -> None:
        try:
            self._chat_history.clear()
            logger.info(f"Cleared chat history")
        except Exception as e:
            logger.error(f"Error clearnig chat history: {str(e)}")

    def get_memory_variables(self) -> dict:
        try:
            return self.memory.load_memory_variables({})
        except Exception as e:
            logger.error(f"Error getting memory variables: {str(e)}")
            return {self.memory_key: ""}
