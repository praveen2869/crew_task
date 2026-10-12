from typing import Optional, List
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Payload for RAG question-answering query."""
    query: str = Field(..., min_length=1, description="Question or prompt to answer")
    top_k: Optional[int] = Field(default=4, ge=1, le=20, description="Maximum number of context chunks to retrieve")
    score_threshold: Optional[float] = Field(default=0.35, ge=0.0, le=1.0, description="Minimum cosine similarity score")
    document_filter: Optional[List[str]] = Field(default=None, description="Optional list of document names to restrict search")


class Citation(BaseModel):
    """Evidence excerpt grounding the answer."""
    source: str = Field(description="Name of the source document")
    page_number: Optional[int] = Field(default=None, description="Page number where excerpt appears (if applicable)")
    chunk_id: str = Field(description="Identifier of the retrieved chunk")
    chunk_index: Optional[int] = Field(default=None, description="Sequence index of the chunk")
    excerpt: str = Field(description="Exact or focal passage excerpt supporting the answer")
    score: float = Field(description="Cosine similarity relevance score")


class QueryResponse(BaseModel):
    """Standardized response for question-answering with evidence and grounding."""
    query: str
    answer: str
    citations: List[Citation] = Field(default_factory=list)
    grounded: bool = Field(description="True if the answer is directly supported by retrieved context")
    is_refusal: bool = Field(default=False, description="True if system refused because evidence was missing")
    confidence_score: float = Field(default=0.0, description="Estimated confidence score based on retrieval similarity")
    retrieved_chunks_count: int = 0
    model_used: str


class RetrievedChunk(BaseModel):
    """Individual retrieved chunk with similarity score."""
    id: str
    document_name: str
    page_number: Optional[int] = None
    chunk_index: int
    content: str
    score: float


class RetrieveResponse(BaseModel):
    """Response for pure retrieval inspection without LLM generation."""
    query: str
    total_retrieved: int
    chunks: List[RetrievedChunk]
