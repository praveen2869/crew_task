import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.services.vector_store import QdrantVectorStore
from app.services.chunker import RecursiveChunker
from app.services.embedding_service import EmbeddingService
from app.services.llm_service import LLMService
from app.services.rag_service import RAGService


@pytest.fixture
def test_client():
    """FastAPI TestClient fixture."""
    return TestClient(app)


@pytest.fixture
def in_memory_store():
    """In-memory QdrantVectorStore fixture isolated per test."""
    return QdrantVectorStore(in_memory=True, collection_name="test_collection", vector_dim=128)


@pytest.fixture
def chunker():
    """Recursive chunker fixture with standard test parameters."""
    return RecursiveChunker(chunk_size=300, chunk_overlap=50)


@pytest.fixture
def mock_embedding_service():
    """Deterministic embedding service fixture."""
    return EmbeddingService(api_key="", model="test-mock", dim=128)


@pytest.fixture
def test_rag_service(in_memory_store, mock_embedding_service, chunker):
    """RAGService wired with isolated in-memory store."""
    llm = LLMService(api_key="", model="mock-test-model")
    return RAGService(
        vector_store=in_memory_store,
        embedding_service=mock_embedding_service,
        llm_service=llm,
        chunker=chunker
    )
