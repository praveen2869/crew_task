import pytest
from app.models.document import DocumentChunk
from app.services.vector_store import QdrantVectorStore


def test_vector_store_upsert_and_search(in_memory_store: QdrantVectorStore, mock_embedding_service):
    # Create sample chunks
    chunks = [
        DocumentChunk(
            id="doc1_p1_c0",
            document_name="doc1.txt",
            content="Kubernetes cluster contains 3 Qdrant nodes with 16GB memory.",
            page_number=1,
            chunk_index=0,
            total_chunks=2
        ),
        DocumentChunk(
            id="doc2_p1_c0",
            document_name="doc2.txt",
            content="API key rotation cycle is mandatory every 30 days.",
            page_number=1,
            chunk_index=0,
            total_chunks=1
        )
    ]

    # Generate embeddings
    embeddings = [
        mock_embedding_service._generate_mock_embedding(c.content)
        for c in chunks
    ]

    count = in_memory_store.upsert_chunks(chunks, embeddings)
    assert count == 2

    # Query
    query_vec = mock_embedding_service._generate_mock_embedding("Kubernetes Qdrant nodes")
    results = in_memory_store.search(query_vec, top_k=2, score_threshold=0.0)

    assert len(results) > 0
    assert results[0].document_name in ["doc1.txt", "doc2.txt"]


def test_vector_store_metadata_filter(in_memory_store: QdrantVectorStore, mock_embedding_service):
    chunks = [
        DocumentChunk(
            id="alpha_0", document_name="alpha.md", content="Alpha secret data.", page_number=1, chunk_index=0, total_chunks=1
        ),
        DocumentChunk(
            id="beta_0", document_name="beta.md", content="Beta secret data.", page_number=1, chunk_index=0, total_chunks=1
        )
    ]
    embeddings = [mock_embedding_service._generate_mock_embedding(c.content) for c in chunks]
    in_memory_store.upsert_chunks(chunks, embeddings)

    # Search with document filter
    query_vec = mock_embedding_service._generate_mock_embedding("secret data")
    filtered_results = in_memory_store.search(query_vec, top_k=5, score_threshold=0.0, filter_docs=["alpha.md"])

    assert all(r.document_name == "alpha.md" for r in filtered_results)


def test_vector_store_delete_document(in_memory_store: QdrantVectorStore, mock_embedding_service):
    chunk = DocumentChunk(
        id="del_0", document_name="temp.txt", content="Temporary information to remove.", page_number=1, chunk_index=0, total_chunks=1
    )
    embeddings = [mock_embedding_service._generate_mock_embedding(chunk.content)]
    in_memory_store.upsert_chunks([chunk], embeddings)

    in_memory_store.delete_document("temp.txt")
    docs = in_memory_store.list_documents()
    assert not any(d.document_name == "temp.txt" for d in docs)
