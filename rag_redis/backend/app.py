import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from backend.services.redis_service import RedisService
from backend.services.memory_service import MemoryService
from backend.services.llm_service import LLMService
from backend.services.rag_service import RAGService
from backend.services.cache_service import CacheService
from backend.services.document_service import DocumentService

from backend.utils.utils import timer

from backend.config import settings

from backend.logging_config import setup_logging

setup_logging()

logger = logging.getLogger(__name__)

# Global Service
redis_service = None
llm_service = None
cache_service = None
document_service = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_service, llm_service, cache_service, document_service

    try:
        redis_service = RedisService(
            settings.redis_url, settings.index_name, settings.indexing
        )
        llm_service = LLMService(settings.ollama_base_url, settings.ollama_model)
        cache_service = CacheService(settings.redis_url, settings.redis_cache_ttl)
        document_service = DocumentService()

        if redis_service.is_connected():
            existing_docs = redis_service.similarity_search("test", top_k=1)
            if not existing_docs:
                sample_docs = document_service.get_sample_documents()
                redis_service.add_documents(sample_docs)
                logger.info("Documents Initialized")
            else:
                logger.info("Documents already exist, skipping initialization")

        logger.info("All services initialized successfully")

    except Exception as e:
        logger.error(f"Error initializing services: {str(e)}")

    yield
    logger.info("Shutting down services")


app = FastAPI(
    title="LangChain + Redis AI System",
    description="Production-ready AI system with LangChain, Ollama, and Redis",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_memory_service(session_id: str):
    return MemoryService(settings.redis_url, session_id)


def get_rag_service(session_id: str):
    memory_service = get_memory_service(session_id)
    return RAGService(llm_service, redis_service, memory_service)


@app.get("/")
async def root():
    return {
        "message": "LangChain + Redis AI System is running!",
        "status": "healthy",
        "services": {
            "redis_service": redis_service.is_connected() if redis_service else False,
            "llm_service": llm_service is not None,
            "cache_service": cache_service is not None,
        },
    }


@app.post("/chat")
@timer
async def chat(
    question: str,
    session_id: str = "default",
    use_cache: bool = True,
    use_memory: bool = True,
    use_vector_search: bool = True,
):
    try:
        rag_service = get_rag_service(session_id)

        response = rag_service.generate(
            question=question,
            use_memory=use_memory,
            use_vector_search=use_vector_search,
        )

        return {**response}

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/clear-memory/{session_id}")
async def clear_memory(session_id: str):
    try:
        memory_service = get_memory_service(session_id)
        memory_service.clear_memory()
        return {"message": f"Memory cleared for session: {session_id}"}

    except Exception as e:
        logger.error(f"Error clearing memory: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear-cache")
async def clear_cache():
    try:
        cache_service.clear_cache()
        return {"message": "Cache cleared successfully"}

    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/search")
async def search_documents(query: str, k: int = 10):
    try:
        if not redis_service:
            raise HTTPException(status_code=503, detail="Redis service not available")

        results = redis_service.similarity_search_with_score(query, top_k=k)

        return {
            "query": query,
            "results": [
                {"content": doc.page_content, "metadata": doc.metadata, "score": score}
                for doc, score in results
            ],
        }
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
