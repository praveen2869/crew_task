from fastapi import APIRouter, HTTPException, status
from app.models.query import QueryRequest, QueryResponse, RetrieveResponse, RetrievedChunk
from app.services.rag_service import RAGService
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(tags=["Question Answering & Retrieval"])


def get_rag_service() -> RAGService:
    return RAGService()


@router.post(
    "/query",
    response_model=QueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask a Grounded Question (RAG Pipeline)"
)
async def query_knowledge_base(payload: QueryRequest):
    """
    Submits a query to the knowledge base:
    1. Generates query embedding via Jina AI
    2. Performs similarity search in Qdrant (with optional metadata filters & score threshold)
    3. Prompts LLM with strict grounding instructions
    4. Extracts verifiable citations with source document name, page, and excerpt
    5. Returns honest refusal if context is insufficient
    """
    try:
        service = get_rag_service()
        response = await service.answer_query(
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            document_filter=payload.document_filter
        )
        return response
    except Exception as e:
        logger.error(f"Error executing query '{payload.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Query execution failed: {str(e)}"
        )


@router.post(
    "/retrieve",
    response_model=RetrieveResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve Raw Matching Chunks (Inspection Endpoint)"
)
async def retrieve_matching_chunks(payload: QueryRequest):
    """
    Retrieves and ranks context chunks from Qdrant without executing LLM generation.
    Useful for testing retrieval precision and inspecting vector similarity scores.
    """
    try:
        service = get_rag_service()
        chunks = await service.retrieve(
            query=payload.query,
            top_k=payload.top_k,
            score_threshold=payload.score_threshold,
            document_filter=payload.document_filter
        )
        return RetrieveResponse(
            query=payload.query,
            total_retrieved=len(chunks),
            chunks=chunks
        )
    except Exception as e:
        logger.error(f"Retrieval failed for query '{payload.query}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Retrieval failed: {str(e)}"
        )
