from fastapi import APIRouter
from app.core.config import get_settings
from app.services.vector_store import QdrantVectorStore

router = APIRouter(tags=["Health"])


@router.get("/health", summary="System Health & Configuration Check")
async def health_check():
    """
    Returns application status, vector database connectivity,
    and active model providers.
    """
    settings = get_settings()
    try:
        store = QdrantVectorStore()
        store_stats = store.get_stats()
        qdrant_status = "healthy"
    except Exception as e:
        store_stats = {"error": str(e)}
        qdrant_status = "degraded"

    storage_mode = "remote" if (settings.qdrant_url and settings.qdrant_url.strip()) else "local_embedded"

    return {
        "status": "ok" if qdrant_status == "healthy" else "degraded",
        "environment": settings.environment,
        "vector_store": {
            "status": qdrant_status,
            "storage_mode": storage_mode,
            "collection_name": settings.qdrant_collection_name,
            "stats": store_stats
        },
        "llm": {
            "provider": "OpenAI",
            "model": settings.openai_model,
            "configured": settings.has_openai_key
        },
        "embeddings": {
            "provider": "Jina AI",
            "model": settings.jina_embedding_model,
            "dimension": settings.jina_embedding_dim,
            "configured": settings.has_jina_key
        }
    }
