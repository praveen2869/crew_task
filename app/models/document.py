from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Represents an individual text chunk extracted from a document with full provenance."""
    id: str = Field(description="Unique deterministic identifier for the chunk")
    document_name: str = Field(description="Originating document filename or title")
    content: str = Field(description="Text content of the chunk")
    page_number: Optional[int] = Field(default=None, description="Page number for multi-page documents (e.g. PDF)")
    chunk_index: int = Field(description="Zero-indexed sequence number of the chunk within document")
    total_chunks: int = Field(default=1, description="Total number of chunks in the document")
    start_char: int = Field(default=0, description="Character start offset in the original document")
    end_char: int = Field(default=0, description="Character end offset in the original document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary user or extracted metadata")


class RawDocument(BaseModel):
    """Represents a parsed document prior to chunking."""
    name: str
    content: str
    file_type: str
    pages: Optional[List[Dict[str, Any]]] = None  # [{"page_number": 1, "text": "..."}]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestTextRequest(BaseModel):
    """Payload for direct text ingestion via API."""
    document_name: str = Field(..., description="Name/ID to associate with the text document")
    content: str = Field(..., min_length=1, description="Text or Markdown content to ingest")
    file_type: str = Field(default="txt", description="Document type: 'txt' or 'md'")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Optional metadata tags")


class IngestResponse(BaseModel):
    """Response returned upon successful document ingestion."""
    status: str = "success"
    document_name: str
    chunks_created: int
    total_characters: int
    message: str


class DocumentSummary(BaseModel):
    """Summary information for an ingested document."""
    document_name: str
    chunk_count: int
    file_type: Optional[str] = None
    page_count: Optional[int] = None


class DocumentListResponse(BaseModel):
    """Response listing all documents currently in the vector store."""
    total_documents: int
    total_chunks: int
    documents: List[DocumentSummary]
