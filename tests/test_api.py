import io
from fastapi.testclient import TestClient


def test_api_health_endpoint(test_client: TestClient):
    response = test_client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "vector_store" in data
    assert "llm" in data
    assert "embeddings" in data


def test_api_ingest_text(test_client: TestClient):
    payload = {
        "document_name": "api_test_doc.txt",
        "content": "Acme security policy requires MFA hardware tokens for all personnel.",
        "file_type": "txt",
        "metadata": {"author": "Security team"}
    }
    response = test_client.post("/api/v1/documents/text", json=payload)
    assert response.status_code == 201
    data = response.json()
    assert data["status"] == "success"
    assert data["document_name"] == "api_test_doc.txt"
    assert data["chunks_created"] >= 1


def test_api_upload_file(test_client: TestClient):
    file_content = b"# Architecture\nMicroservices communicate via gRPC on port 50051."
    files = {"file": ("architecture.md", io.BytesIO(file_content), "text/markdown")}
    response = test_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 201
    data = response.json()
    assert data["document_name"] == "architecture.md"


def test_api_list_documents(test_client: TestClient):
    response = test_client.get("/api/v1/documents")
    assert response.status_code == 200
    data = response.json()
    assert "total_documents" in data
    assert "documents" in data


def test_api_retrieve_endpoint(test_client: TestClient):
    payload = {
        "query": "MFA hardware tokens security policy",
        "top_k": 3,
        "score_threshold": 0.0
    }
    response = test_client.post("/api/v1/retrieve", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "query" in data
    assert "chunks" in data


def test_api_query_endpoint(test_client: TestClient):
    payload = {
        "query": "What type of MFA tokens are required by the security policy?",
        "top_k": 3,
        "score_threshold": 0.0
    }
    response = test_client.post("/api/v1/query", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "citations" in data
    assert "grounded" in data
