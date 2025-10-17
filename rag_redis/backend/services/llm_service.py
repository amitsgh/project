import logging
from typing import List, Dict, Any, Optional

from langchain_ollama import OllamaLLM
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

from backend.config import settings

logger = logging.getLogger(__name__)


class LLMService:
    _instance: Optional["LLMService"] = None
    _llm: Optional[OllamaLLM] = None

    base_url = None
    model_name = None
    temperature = None
    num_predict = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMService, cls).__new__(cls)
            cls._instance.base_url = settings.ollama_base_url
            cls._instance.model_name = settings.ollama_model
            cls._instance.temperature = settings.temperature
            cls._instance.num_predict = settings.max_tokens
            logger.info(f"LLMService configured with model: {cls._instance.model_name}")

        return cls._instance

    @property
    def llm(self) -> OllamaLLM:
        if self._llm is None:
            logger.info(f"Initializing Ollama LLM: {self.model_name}")
            self._llm = OllamaLLM(
                base_url=self.base_url,
                model=self.model_name,
                temperature=self.temperature,
                num_predict=self.num_predict,
            )
            logger.info(f"Ollama LLM '{self.model_name}' initialized successfully")

        return self._llm

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        formatted = []
        for message in messages:
            if isinstance(message, SystemMessage):
                formatted.append(f"System: {message.content}")
            elif isinstance(message, HumanMessage):
                formatted.append(f"Human: {message.content}")
            else:
                formatted.append(f"Assistant: {message.content}")

        return "\n".join(formatted)

    def generate(self, messages: List[BaseMessage]) -> Dict[str, Any]:
        try:
            prompt = self._format_messages(messages)
            response = self.llm.invoke(prompt)

            return {"response": response, "success": True}
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")

            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "success": False,
            }
