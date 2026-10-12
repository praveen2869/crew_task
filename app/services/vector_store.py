import os
import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.document import DocumentChunk, DocumentSummary
from app.models.query import RetrievedChunk

logger = get_logger(__name__)


class QdrantVectorStore:
    """
    Manages vector storage and semantic search operations using Qdrant.
    Supports local embedded disk storage, in-memory mode (for testing), or remote Qdrant servers.
    """

    def __init__(self, in_memory: bool = False, collection_name: Optional[str] = None, vector_dim: Optional[int] = None):
        settings = get_settings()
        self.collection_name = collection_name or settings.qdrant_collection_name
        self.vector_dim = vector_dim or settings.jina_embedding_dim

        if in_memory:
            logger.info("Initializing Qdrant in IN-MEMORY mode")
            self.client = QdrantClient(":memory:")
        elif settings.qdrant_url and settings.qdrant_url.strip():
            logger.info(f"Connecting to remote Qdrant server at: {settings.qdrant_url}")
            self.client = QdrantClient(
                url=settings.qdrant_url.strip(),
                api_key=settings.qdrant_api_key.strip() if settings.qdrant_api_key else None
            )
        else:
            storage_path = os.path.abspath(settings.qdrant_storage_path)
            os.makedirs(storage_path, exist_ok=True)
            logger.info(f"Initializing Qdrant in LOCAL EMBEDDED mode at: {storage_path}")
            self.client = QdrantClient(path=storage_path)

        self._ensure_collection()

    def _ensure_collection(self):
        """Creates the Qdrant collection and payload indexes if they do not exist."""
        try:
            collections = self.client.get_collections().collections
            exists = any(c.name == self.collection_name for c in collections)

            if not exists:
                logger.info(f"Creating Qdrant collection '{self.collection_name}' (dim={self.vector_dim}, distance=COSINE)")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.vector_dim,
                        distance=qmodels.Distance.COSINE
                    )
                )

                settings = get_settings()
                if settings.qdrant_url and settings.qdrant_url.strip():
                    self.client.create_payload_index(
                        collection_name=self.collection_name,
                        field_name="document_name",
                        field_schema=qmodels.PayloadSchemaType.KEYWORD
                    )
                logger.info(f"Collection '{self.collection_name}' created successfully")
        except Exception as e:
            logger.error(f"Error ensuring Qdrant collection: {e}")
            raise

    def upsert_chunks(self, chunks: List[DocumentChunk], embeddings: List[List[float]]) -> int:
        """Stores chunks and their corresponding embedding vectors in Qdrant."""
        if len(chunks) != len(embeddings):
            raise ValueError(f"Chunks count ({len(chunks)}) != embeddings count ({len(embeddings)})")

        if not chunks:
            return 0

        # Verify dimension against collection configuration
        actual_dim = len(embeddings[0])
        try:
            info = self.client.get_collection(self.collection_name)
            current_dim = getattr(info.config.params.vectors, "size", self.vector_dim)
        except Exception:
            current_dim = self.vector_dim

        if actual_dim != current_dim:
            logger.info(f"Recreating collection to match actual embedding dimension ({actual_dim} != {current_dim})")
            self.vector_dim = actual_dim
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qmodels.VectorParams(size=self.vector_dim, distance=qmodels.Distance.COSINE)
            )

        points = []
        for chunk, vector in zip(chunks, embeddings):
            # Generate UUID v5 from chunk ID for stable Qdrant point IDs
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.id))
            payload = {
                "chunk_id": chunk.id,
                "document_name": chunk.document_name,
                "content": chunk.content,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata
            }
            points.append(qmodels.PointStruct(
                id=point_id,
                vector=vector,
                payload=payload
            ))

        # Batch upsert
        batch_size = 64
        for i in range(0, len(points), batch_size):
            batch = points[i:i + batch_size]
            self.client.upsert(
                collection_name=self.collection_name,
                points=batch
            )

        logger.info(f"Successfully upserted {len(chunks)} chunks into '{self.collection_name}'")
        return len(chunks)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: float = 0.35,
        filter_docs: Optional[List[str]] = None
    ) -> List[RetrievedChunk]:
        """
        Executes semantic search with optional metadata filtering and score thresholding.
        """
        query_filter = None
        if filter_docs and len(filter_docs) > 0:
            query_filter = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key="document_name",
                        match=qmodels.MatchAny(any=filter_docs)
                    )
                ]
            )

        try:
            # Query Qdrant
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=top_k,
                score_threshold=score_threshold,
                query_filter=query_filter,
                with_payload=True
            )
            scored_points = results.points
        except Exception as e:
            # Fallback to search method if query_points has older API variation
            try:
                scored_points = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_vector,
                    limit=top_k,
                    score_threshold=score_threshold,
                    query_filter=query_filter,
                    with_payload=True
                )
            except Exception as e2:
                logger.error(f"Search failed in Qdrant: {e2}")
                return []

        retrieved: List[RetrievedChunk] = []
        for point in scored_points:
            payload = point.payload or {}
            retrieved.append(RetrievedChunk(
                id=payload.get("chunk_id", str(point.id)),
                document_name=payload.get("document_name", "unknown"),
                page_number=payload.get("page_number"),
                chunk_index=payload.get("chunk_index", 0),
                content=payload.get("content", ""),
                score=round(float(point.score), 4)
            ))

        logger.info(f"Retrieved {len(retrieved)} chunks for query vector (threshold={score_threshold})")
        return retrieved

    def get_document_chunks(
        self,
        document_name: Optional[str] = None,
        limit: int = 5
    ) -> List[RetrievedChunk]:
        """Fetches leading representative chunks for document summarization/overview."""
        try:
            scroll_filter = None
            if document_name:
                scroll_filter = qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="document_name",
                            match=qmodels.MatchValue(value=document_name)
                        )
                    ]
                )
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit * 3,
                with_payload=True,
                with_vectors=False
            )
            if not points:
                return []

            # Sort by chunk_index to capture the document's introduction and flow
            sorted_points = sorted(points, key=lambda p: (p.payload or {}).get("chunk_index", 0))[:limit]
            chunks = []
            for p in sorted_points:
                payload = p.payload or {}
                chunks.append(RetrievedChunk(
                    id=payload.get("chunk_id", str(p.id)),
                    document_name=payload.get("document_name", "unknown"),
                    content=payload.get("content", ""),
                    score=0.85,
                    page_number=payload.get("page_number"),
                    chunk_index=payload.get("chunk_index", 0),
                    metadata=payload
                ))
            return chunks
        except Exception as e:
            logger.error(f"Error fetching representative chunks: {e}")
            return []

    def list_documents(self) -> List[DocumentSummary]:
        """Scans the collection and aggregates statistics per document."""
        doc_stats: Dict[str, Dict[str, Any]] = {}
        offset = None

        try:
            while True:
                records, next_offset = self.client.scroll(
                    collection_name=self.collection_name,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False
                )

                for r in records:
                    payload = r.payload or {}
                    doc_name = payload.get("document_name", "unknown")
                    file_type = payload.get("file_type")
                    page_num = payload.get("page_number")

                    if doc_name not in doc_stats:
                        doc_stats[doc_name] = {
                            "chunk_count": 0,
                            "file_type": file_type,
                            "max_page": page_num or 1
                        }

                    doc_stats[doc_name]["chunk_count"] += 1
                    if page_num and page_num > doc_stats[doc_name]["max_page"]:
                        doc_stats[doc_name]["max_page"] = page_num

                if next_offset is None:
                    break
                offset = next_offset

        except Exception as e:
            logger.error(f"Error scrolling documents: {e}")

        summaries = [
            DocumentSummary(
                document_name=doc,
                chunk_count=data["chunk_count"],
                file_type=data["file_type"],
                page_count=data["max_page"] if data["file_type"] == "pdf" else None
            )
            for doc, data in doc_stats.items()
        ]
        return sorted(summaries, key=lambda s: s.document_name)

    def delete_document(self, document_name: str) -> int:
        """Deletes all chunks belonging to the specified document."""
        try:
            res = self.client.delete(
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[
                            qmodels.FieldCondition(
                                key="document_name",
                                match=qmodels.MatchValue(value=document_name)
                            )
                        ]
                    )
                )
            )
            logger.info(f"Deleted points for document '{document_name}'")
            return 1
        except Exception as e:
            logger.error(f"Failed to delete document '{document_name}': {e}")
            raise

    def clear_collection(self) -> bool:
        """Clears all points by recreating the collection."""
        try:
            self.client.delete_collection(self.collection_name)
            self._ensure_collection()
            logger.info(f"Collection '{self.collection_name}' wiped and reset")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            raise

    def get_stats(self) -> Dict[str, Any]:
        """Returns collection stats."""
        try:
            info = self.client.get_collection(self.collection_name)
            return {
                "collection_name": self.collection_name,
                "points_count": getattr(info, "points_count", getattr(info, "vectors_count", 0)),
                "status": getattr(info, "status", "green"),
                "vector_dim": self.vector_dim
            }
        except Exception as e:
            return {"error": str(e), "collection_name": self.collection_name}
