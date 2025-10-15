import logging
from typing import List, Dict, Any

from langchain_ollama import OllamaLLM
from langchain.schema import BaseMessage, HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class LLMService:
    def __init__(
        self, base_url: str, model: str, temperatur: int = 0.7, num_predict: int = 1000
    ) -> None:
        self.llm = OllamaLLM(
            base_url=base_url,
            model=model,
            temperature=temperatur,
            num_predict=num_predict,
        )
        self.model_name = model

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
