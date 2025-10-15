from typing import List, Dict, Any
import logging

from huggingface_hub import try_to_load_from_cache
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentService:
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len
        )

    def create_documents_from_texts(
        self, texts: List[str], metadatas: List[Dict[str, Any]] = None
    ) -> List[Document]:
        try:
            if metadatas is None:
                metadatas = [{}] * len(texts)

            documents = []
            for i, (text, metadata) in enumerate(zip(texts, metadatas)):
                chunks = self.text_splitter.split_text(text)

                for j, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            **metadata,
                            "chunk_id": f"{i}_{j}",
                            "source": metadata.get("source", f"text+{i}"),
                        },
                    )
                    documents.append(doc)

            logger.info(f"Created {len(documents)} documents from {len(texts)} texts")
            return documents
        except Exception as e:
            logger.error(f"Error creating documents: {str(e)}")
            return []

    def get_sample_documents(self) -> List[Document]:
        """Get sample documents for testing"""
        sample_texts = [
            "Python is a high-level programming language known for its simplicity and readability. It's widely used in web development, data science, artificial intelligence, and automation. Python's syntax is clean and easy to learn, making it popular among beginners and experienced developers alike.",
            "Redis is an in-memory data structure store that can be used as a database, cache, and message broker. It supports various data structures like strings, hashes, lists, sets, and sorted sets. Redis is known for its high performance and is commonly used for caching, session storage, and real-time analytics.",
            "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed. It involves algorithms that can identify patterns in data and make predictions or classifications. Common applications include image recognition, natural language processing, and recommendation systems.",
            "FastAPI is a modern, fast web framework for building APIs with Python 3.7+. It's based on standard Python type hints and provides automatic API documentation, data validation, and serialization. FastAPI is known for its high performance, comparable to Node.js and Go, and is widely used for building microservices and web APIs.",
            "Vector databases are specialized databases designed to store and retrieve high-dimensional vectors efficiently. They're essential for semantic search, recommendation systems, and AI applications. Vector databases use similarity search algorithms to find the most relevant vectors based on distance metrics like cosine similarity or Euclidean distance.",
        ]

        sample_metadatas = [
            {"source": "python_guide", "category": "programming", "topic": "python"},
            {"source": "redis_docs", "category": "database", "topic": "redis"},
            {"source": "ml_tutorial", "category": "ai", "topic": "machine_learning"},
            {"source": "fastapi_docs", "category": "programming", "topic": "fastapi"},
            {
                "source": "vector_db_guide",
                "category": "database",
                "topic": "vector_databases",
            },
        ]

        return self.create_documents_from_texts(sample_texts, sample_metadatas)
