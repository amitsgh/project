import logging
from typing import List

from langchain_redis import RedisChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.messages.utils import count_tokens_approximately
from langmem.short_term import SummarizationNode

from backend.services.llm_service import LLMService
from backend.config import settings

logger = logging.getLogger(__name__)

OUTPUT_DIR = "output"

class MemoryService:
    def __init__(
        self,
        session_id: str,
    ) -> None:
        self.session_id = session_id

        self.redis_url = settings.redis_url
        self.ttl = settings.memory_ttl
        self.memory_key = settings.memory_key

        llm_service = LLMService()
        self.llm = llm_service.llm

        self._chat_history = RedisChatMessageHistory(
            redis_url=self.redis_url,
            session_id=self.session_id,
            ttl=self.ttl,
        )

        # self._memory = ConversationSummaryBufferMemory(
        #     llm=self.llm,
        #     chat_memory=self._chat_history,
        #     memory_key=self.memory_key,
        #     return_messages=True,
        # )

        # summarization_model = self.llm.bind(max_tokens=128)

        # self._memory = SummarizationNode(
        #     token_counter=count_tokens_approximately,
        #     model=summarization_model,
        #     max_tokens=256,
        #     max_tokens_before_summary=256,
        #     max_summary_tokens=128,
        # ) 

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
        # try:
        #     return self._memory.load_memory_variables({})
        # except Exception as e:
        #     logger.error(f"Error getting memory variables: {str(e)}")
        #     return {self.memory_key: ""}
    
        try:
            messages = self._chat_history.messages
            chat_history = "\n".join([f"{msg.type}: {msg.content}" for msg in messages])
            return {self.memory_key: chat_history}

        except Exception as e:
            logger.error(f"Error getting memory variables: {str(e)}")
            return {self.memory_key: ""}
