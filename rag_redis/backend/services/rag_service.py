import logging
from typing import Dict, Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, Runnable
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class RAGService:
    def __init__(self, llm_service, redis_service, memory_service):
        self.llm_service = llm_service
        self.redis_service = redis_service
        self.memory_service = memory_service

        self.rag_chain = self._create_qa_chain()

    def _create_qa_chain(self) -> Runnable:
        prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """You are a helpful AI assistant. Use the following context and conversation history to answer the user's question.
                Context: {context}
                Conversation History: {chat_history}
                Please provide a helpful, accurate response based on the context and conversation history. If the context doesn't contain relevant information, say so politely.""",
                ),
                ("human", "{question}"),
            ]
        )

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        rag_chain = (
            {
                "context": self.redis_service.vector_store.as_retriever(
                    search_kwargs={"k": 10}
                )
                | format_docs,
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self.memory_service.get_memory_variables().get(
                    "chat_history", ""
                ),
            }
            | prompt_template
            | self.llm_service.llm
            | StrOutputParser()
        )

        return rag_chain

    def generate(
        self, question: str, use_memory: bool = True, use_vector_search: bool = True
    ) -> Dict[str, Any]:
        try:
            response = self.rag_chain.invoke(question)

            if use_memory:
                self.memory_service.add_message(HumanMessage(content=question))
                self.memory_service.add_message(SystemMessage(content=response))

            return {
                "response": response,
                "context_used": use_vector_search,
                "memory_used": use_memory,
                "model_used": self.llm_service.model_name,
                "success": True,
            }

        except Exception as e:
            logger.error(f"Error in RAG response: {str(e)}")

            return {
                "response": f"I apologize, but I encountered an error: {str(e)}",
                "context_used": False,
                "memory_used": False,
                "model_used": self.llm_service.model_name,
                "success": False,
            }
