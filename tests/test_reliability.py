import io
import pytest
from fastapi.testclient import TestClient
from app.services.rag_service import RAGService
from app.services.llm_service import STANDARD_REFUSAL


@pytest.mark.asyncio
async def test_unanswerable_query_refusal(test_rag_service: RAGService):
    # Ingest known topic
    await test_rag_service.ingest_raw_text(
        document_name="network_config.txt",
        content="The DNS servers are configured to 8.8.8.8 and 1.1.1.1."
    )

    # Ask unanswerable question about unrelated topic
    response = await test_rag_service.answer_query(
        query="What is the CEO's favorite luxury automobile brand?",
        top_k=3,
        score_threshold=0.85  # High threshold triggers refusal
    )

    assert response.is_refusal is True
    assert (
        STANDARD_REFUSAL.lower() in response.answer.lower() or
        "not contain sufficient information" in response.answer.lower()
    )
    assert len(response.citations) == 0


@pytest.mark.asyncio
async def test_citations_contain_excerpts_and_sources(test_rag_service: RAGService):
    await test_rag_service.ingest_raw_text(
        document_name="retention_policy.txt",
        content="All financial records and compliance reports must be retained for 7 years."
    )

    response = await test_rag_service.answer_query(
        query="retention period for financial records",
        score_threshold=0.0
    )

    assert response.is_refusal is False
    assert len(response.citations) > 0
    citation = response.citations[0]
    assert citation.source == "retention_policy.txt"
    assert "7 years" in citation.excerpt
    assert citation.score >= 0.0


def test_invalid_file_extension_rejected(test_client: TestClient):
    file_content = b"Binary data"
    files = {"file": ("unsupported_image.png", io.BytesIO(file_content), "image/png")}
    response = test_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "Unsupported file format" in response.json()["detail"]


def test_empty_file_rejected(test_client: TestClient):
    files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
    response = test_client.post("/api/v1/documents/upload", files=files)
    assert response.status_code == 400
    assert "empty" in response.json()["detail"]


def test_empty_query_rejected(test_client: TestClient):
    response = test_client.post("/api/v1/query", json={"query": ""})
    assert response.status_code == 422  # Pydantic validation failure


def test_clean_answer_text_removes_inline_citations():
    from app.services.llm_service import clean_answer_text

    raw_answer = "Praveen is working at Ascendion in Bangalore, India as a Backend Developer [Doc: Praveen_Resumee.pdf, Page: 1, Chunk: 0]."
    cleaned = clean_answer_text(raw_answer)
    assert cleaned == "Praveen is working at Ascendion in Bangalore, India as a Backend Developer."
    assert "[Doc:" not in cleaned

    multi_citations = "He worked on RAG [Doc: resume.pdf, Page: 1] and cloud [Document: cloud.txt, Chunk: 2]."
    cleaned_multi = clean_answer_text(multi_citations)
    assert "[Doc:" not in cleaned_multi
    assert "[Document:" not in cleaned_multi
    assert cleaned_multi == "He worked on RAG and cloud."
