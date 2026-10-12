from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.api.routes_health import router as health_router
from app.api.routes_ingest import router as ingest_router
from app.api.routes_query import router as query_router
from app.services.vector_store import QdrantVectorStore

setup_logging()
logger = get_logger("app.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager: runs on startup and shutdown."""
    settings = get_settings()
    logger.info(f"Starting Reliable RAG API (Environment: {settings.environment})")
    try:
        # Pre-initialize Qdrant collection
        store = QdrantVectorStore()
        stats = store.get_stats()
        logger.info(f"Qdrant collection ready: {stats}")
    except Exception as e:
        logger.warning(f"Could not connect to Qdrant at startup: {e}")
    yield
    logger.info("Shutting down Reliable RAG API")


def create_app() -> FastAPI:
    """Factory function for FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Reliable RAG Knowledge Assistant API",
        description=(
            "Production-grade Retrieval-Augmented Generation API built with FastAPI, Qdrant, "
            "Jina AI embeddings, and OpenAI. Features strict evidence grounding, inline citations, "
            "honest refusal for unanswerable queries, and multi-format document ingestion."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan
    )

    # Enable CORS for Streamlit and external clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Exception Handler
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred.", "error": str(exc)}
        )

    # Root health redirect
    @app.get("/", tags=["Root"])
    async def root():
        return {
            "name": "Reliable RAG API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/api/v1/health"
        }

    # Mount API routers under /api/v1
    app.include_router(health_router, prefix="/api/v1")
    app.include_router(ingest_router, prefix="/api/v1")
    app.include_router(query_router, prefix="/api/v1")

    # Also mount health at /health for docker/k8s probes
    app.include_router(health_router, prefix="")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run("app.main:app", host=settings.api_host, port=settings.api_port, reload=True)
