import os
import re
from typing import List, Optional, Dict, Any
from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.document import DocumentChunk, RawDocument, IngestResponse, DocumentSummary
from app.models.query import QueryResponse, Citation, RetrievedChunk, RetrieveResponse
from app.services.document_parser import DocumentParser
from app.services.chunker import RecursiveChunker
from app.services.embedding_service import EmbeddingService
from app.services.vector_store import QdrantVectorStore
from app.services.llm_service import LLMService, STANDARD_REFUSAL, clean_answer_text

logger = get_logger(__name__)


class RAGService:
    """
    Orchestrates the complete RAG workflow:
    - Document ingestion (parse -> chunk -> embed -> index)
    - Query pipeline (embed query -> vector search -> LLM grounded generation -> citation mapping)
    - Provenance and confidence tracking
    """

    def __init__(
        self,
        vector_store: Optional[QdrantVectorStore] = None,
        embedding_service: Optional[EmbeddingService] = None,
        llm_service: Optional[LLMService] = None,
        chunker: Optional[RecursiveChunker] = None
    ):
        settings = get_settings()
        self.vector_store = vector_store or QdrantVectorStore()
        self.embedding_service = embedding_service or EmbeddingService()
        self.llm_service = llm_service or LLMService()
        self.chunker = chunker or RecursiveChunker()
        self.default_top_k = settings.retrieval_top_k
        self.default_threshold = settings.score_threshold
        self._last_ingested_doc: Optional[str] = None

    def _is_summary_query(self, query: str) -> bool:
        """Detects whether a query is asking for a general summary or overview."""
        q = query.strip().lower()
        if any(term in q for term in ["summar", "overview", "tldr", "tl;dr", "brief me", "outline"]):
            return True
        patterns = [
            r"what is this .* (about|document|file)",
            r"what does this .* (say|discuss|contain|cover)",
            r"tell me about this",
            r"explain this",
            r"key points",
            r"main points"
        ]
        return any(re.search(pat, q) for pat in patterns)

    def _resolve_target_document_for_summary(self, query: str) -> Optional[str]:
        """Resolves target document name if mentioned in query, or defaults to latest ingested."""
        q = query.lower()
        docs = self.list_documents()
        for d in docs:
            stem = d.document_name.rsplit(".", 1)[0].lower()
            if d.document_name.lower() in q or (len(stem) > 3 and stem in q):
                return d.document_name

        if not self._last_ingested_doc and os.path.exists("qdrant_storage/.last_ingested_doc"):
            try:
                with open("qdrant_storage/.last_ingested_doc", "r", encoding="utf-8") as f:
                    self._last_ingested_doc = f.read().strip()
            except Exception:
                pass

        if self._last_ingested_doc:
            return self._last_ingested_doc
        if len(docs) == 1:
            return docs[0].document_name
        return None

    async def ingest_file(
        self,
        file_bytes: bytes,
        filename: str,
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestResponse:
        """Parses, chunks, embeds, and indexes an uploaded file (PDF, TXT, or MD)."""
        logger.info(f"Starting ingestion for file: '{filename}' ({len(file_bytes)} bytes)")
        self._last_ingested_doc = filename
        try:
            os.makedirs("qdrant_storage", exist_ok=True)
            with open("qdrant_storage/.last_ingested_doc", "w", encoding="utf-8") as f:
                f.write(filename)
        except Exception:
            pass
        raw_doc = DocumentParser.parse_file(file_bytes, filename)
        if custom_metadata:
            raw_doc.metadata.update(custom_metadata)

        chunks = self.chunker.chunk_document(raw_doc)
        if not chunks:
            raise ValueError(f"Document '{filename}' produced 0 chunks. Ensure file is not empty.")

        texts = [c.content for c in chunks]
        embeddings = await self.embedding_service.embed_passages(texts)

        # Store in Qdrant
        self.vector_store.upsert_chunks(chunks, embeddings)

        total_chars = sum(len(c.content) for c in chunks)
        return IngestResponse(
            status="success",
            document_name=filename,
            chunks_created=len(chunks),
            total_characters=total_chars,
            message=f"Successfully indexed '{filename}' ({len(chunks)} chunks, {total_chars} characters)"
        )

    async def ingest_raw_text(
        self,
        document_name: str,
        content: str,
        file_type: str = "txt",
        custom_metadata: Optional[Dict[str, Any]] = None
    ) -> IngestResponse:
        """Ingests raw text or markdown directly into the vector store."""
        self._last_ingested_doc = document_name
        try:
            os.makedirs("qdrant_storage", exist_ok=True)
            with open("qdrant_storage/.last_ingested_doc", "w", encoding="utf-8") as f:
                f.write(document_name)
        except Exception:
            pass
        cleaned = DocumentParser.clean_text(content)
        raw_doc = RawDocument(
            name=document_name,
            content=cleaned,
            file_type=file_type,
            pages=[{"page_number": 1, "text": cleaned}],
            metadata=custom_metadata or {}
        )

        chunks = self.chunker.chunk_document(raw_doc)
        if not chunks:
            raise ValueError(f"Text for '{document_name}' produced 0 chunks.")

        texts = [c.content for c in chunks]
        embeddings = await self.embedding_service.embed_passages(texts)
        self.vector_store.upsert_chunks(chunks, embeddings)

        total_chars = sum(len(c.content) for c in chunks)
        return IngestResponse(
            status="success",
            document_name=document_name,
            chunks_created=len(chunks),
            total_characters=total_chars,
            message=f"Successfully indexed '{document_name}' ({len(chunks)} chunks)"
        )

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        document_filter: Optional[List[str]] = None
    ) -> List[RetrievedChunk]:
        """Retrieves top relevant chunks from Qdrant vector store."""
        k = top_k if top_k is not None else self.default_top_k
        threshold = score_threshold if score_threshold is not None else self.default_threshold

        query_vec = await self.embedding_service.embed_query(query)
        chunks = self.vector_store.search(
            query_vector=query_vec,
            top_k=k,
            score_threshold=threshold,
            filter_docs=document_filter
        )
        return chunks

    def _extract_focal_excerpt(self, chunk_text: str, max_chars: int = 240) -> str:
        """Extracts the most representative excerpt snippet from a chunk."""
        cleaned = " ".join(chunk_text.split())
        if len(cleaned) <= max_chars:
            return cleaned
        # Return first complete sentence or slice cleanly at word boundary
        sentence_end = cleaned[:max_chars].rfind(". ")
        if sentence_end > 80:
            return cleaned[:sentence_end + 1]
        space_end = cleaned[:max_chars].rfind(" ")
        if space_end > 80:
            return cleaned[:space_end] + "..."
        return cleaned[:max_chars] + "..."

    async def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        document_filter: Optional[List[str]] = None
    ) -> QueryResponse:
        """
        Executes end-to-end RAG query:
        1. Embeds query & performs vector search in Qdrant
        2. Validates retrieved passages against threshold
        3. Calls OpenAI with strict grounding prompt
        4. Synthesizes citations with document name, page, chunk id, and excerpts
        """
        retrieved_chunks = await self.retrieve(
            query=query,
            top_k=top_k,
            score_threshold=score_threshold,
            document_filter=document_filter
        )

        # Fallback for summarization / overview requests where semantic query distance is naturally low
        if not retrieved_chunks and self._is_summary_query(query):
            target_doc = self._resolve_target_document_for_summary(query)
            logger.info(f"Summarization intent detected. Fetching representative chunks for: {target_doc or 'all'}")
            retrieved_chunks = self.vector_store.get_document_chunks(
                document_name=target_doc,
                limit=top_k or self.default_top_k
            )

        # Case 1: No chunks retrieved above threshold -> Honest refusal
        if not retrieved_chunks:
            logger.info(f"No chunks retrieved above threshold for query: '{query}'")
            return QueryResponse(
                query=query,
                answer=STANDARD_REFUSAL,
                citations=[],
                grounded=False,
                is_refusal=True,
                confidence_score=0.0,
                retrieved_chunks_count=0,
                model_used=self.llm_service.model
            )

        # Case 2: Pass context to LLM
        answer, is_refusal, model_name = self.llm_service.generate_answer(query, retrieved_chunks)
        if not is_refusal:
            answer = clean_answer_text(answer)

        # Build structured citations
        citations: List[Citation] = []
        if not is_refusal:
            for chunk in retrieved_chunks:
                excerpt = self._extract_focal_excerpt(chunk.content)
                citations.append(Citation(
                    source=chunk.document_name,
                    page_number=chunk.page_number,
                    chunk_id=chunk.id,
                    chunk_index=chunk.chunk_index,
                    excerpt=excerpt,
                    score=chunk.score
                ))

        # Calculate confidence score based on top retrieval scores
        top_score = retrieved_chunks[0].score if retrieved_chunks else 0.0
        avg_score = sum(c.score for c in retrieved_chunks) / len(retrieved_chunks)
        confidence = round(float((top_score * 0.7) + (avg_score * 0.3)), 3)

        return QueryResponse(
            query=query,
            answer=answer,
            citations=citations,
            grounded=not is_refusal,
            is_refusal=is_refusal,
            confidence_score=0.0 if is_refusal else confidence,
            retrieved_chunks_count=len(retrieved_chunks),
            model_used=model_name
        )

    def list_documents(self) -> List[DocumentSummary]:
        """Lists all indexed documents."""
        return self.vector_store.list_documents()

    def delete_document(self, doc_name: str) -> bool:
        """Deletes all chunks of a document."""
        self.vector_store.delete_document(doc_name)
        return True

    def clear_all(self) -> bool:
        """Clears all vectors in Qdrant collection."""
        return self.vector_store.clear_collection()
