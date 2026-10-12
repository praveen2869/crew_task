from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from app.models.document import (
    IngestResponse,
    IngestTextRequest,
    DocumentListResponse,
)
from app.services.rag_service import RAGService
from app.core.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Document Ingestion"])


def get_rag_service() -> RAGService:
    return RAGService()


@router.post(
    "/upload",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload & Index a Document (PDF, TXT, MD)"
)
async def upload_document(
    file: UploadFile = File(..., description="Document file (.pdf, .txt, or .md)"),
    tags: Optional[str] = Form(None, description="Comma-separated optional tags")
):
    """
    Accepts PDF, TXT, or Markdown documents, extracts text preserving page numbers,
    recursively splits into linguistic chunks, computes Jina embeddings,
    and indexes them in Qdrant.
    """
    filename = file.filename or "uploaded_document.txt"
    ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

    if ext not in {".pdf", ".txt", ".md"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{ext}'. Only .pdf, .txt, and .md files are accepted."
        )

    try:
        content_bytes = await file.read()
        if not content_bytes:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Uploaded file '{filename}' is empty."
            )

        metadata = {}
        if tags:
            metadata["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

        service = get_rag_service()
        response = await service.ingest_file(content_bytes, filename, custom_metadata=metadata)
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ingestion failed for '{filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest document: {str(e)}"
        )


@router.post(
    "/text",
    response_model=IngestResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest Raw Text or Markdown Content"
)
async def ingest_text(payload: IngestTextRequest):
    """
    Directly ingests raw string text or markdown with associated metadata.
    """
    try:
        service = get_rag_service()
        response = await service.ingest_raw_text(
            document_name=payload.document_name,
            content=payload.content,
            file_type=payload.file_type,
            custom_metadata=payload.metadata
        )
        return response
    except Exception as e:
        logger.error(f"Text ingestion failed for '{payload.document_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest raw text: {str(e)}"
        )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List All Ingested Documents"
)
async def list_documents():
    """
    Returns an inventory of all indexed documents with chunk counts and metadata.
    """
    try:
        service = get_rag_service()
        docs = service.list_documents()
        total_chunks = sum(d.chunk_count for d in docs)
        return DocumentListResponse(
            total_documents=len(docs),
            total_chunks=total_chunks,
            documents=docs
        )
    except Exception as e:
        logger.error(f"Failed to list documents: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve document inventory: {str(e)}"
        )


@router.delete(
    "/{document_name}",
    status_code=status.HTTP_200_OK,
    summary="Delete a Document by Name"
)
async def delete_document(document_name: str):
    """
    Deletes all vector points and chunks belonging to a specific document.
    """
    try:
        service = get_rag_service()
        service.delete_document(document_name)
        return {"status": "success", "message": f"Document '{document_name}' deleted."}
    except Exception as e:
        logger.error(f"Failed to delete document '{document_name}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete document: {str(e)}"
        )


@router.delete(
    "",
    status_code=status.HTTP_200_OK,
    summary="Clear and Reset the Entire Knowledge Base"
)
async def clear_all_documents():
    """
    Wipes the Qdrant collection and reinitializes it.
    """
    try:
        service = get_rag_service()
        service.clear_all()
        return {"status": "success", "message": "Knowledge base collection wiped and reset."}
    except Exception as e:
        logger.error(f"Failed to clear knowledge base: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reset collection: {str(e)}"
        )
