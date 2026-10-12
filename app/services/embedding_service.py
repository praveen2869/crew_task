import math
import hashlib
from typing import List, Optional
import httpx
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Manages embedding generation using Jina AI Embeddings API.
    Supports task-specific embeddings: 'retrieval.passage' for documents and 'retrieval.query' for queries.
    Provides offline deterministic fallback when JINA_API_KEY is not configured.
    """

    API_URL = "https://api.jina.ai/v1/embeddings"

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, dim: Optional[int] = None):
        settings = get_settings()
        self.api_key = api_key if api_key is not None else settings.jina_api_key
        self.model = model or settings.jina_embedding_model
        self.dimension = dim or settings.jina_embedding_dim
        self._is_live = bool(self.api_key and self.api_key.strip() and not self.api_key.startswith("your_"))

        if self._is_live:
            logger.info(f"EmbeddingService initialized using Jina AI API (model='{self.model}')")
        else:
            logger.warning(
                "EmbeddingService running in DETERMINISTIC MOCK MODE (no JINA_API_KEY set). "
                "Embeddings will be generated locally via semantic pseudo-hashing for offline testing."
            )

    @property
    def is_live(self) -> bool:
        return self._is_live

    STOPWORDS = {"what", "is", "the", "and", "in", "of", "for", "to", "a", "an", "defined", "according", "by", "are", "with", "from", "on", "at", "as", "or", "which", "how", "under", "between", "does", "this", "relate"}

    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generates a deterministic, normalized vector based on character and n-gram hash buckets.
        Semantically overlapping texts share non-zero coordinate projections.
        """
        dim = self.dimension
        vec = [0.0] * dim

        words = [w.strip(".,;:?!'\"()[]{}").lower() for w in text.split()]
        significant_words = [w for w in words if w and w not in self.STOPWORDS and len(w) > 2]

        if not significant_words:
            significant_words = words or ["text"]

        for word in significant_words:
            h = int(hashlib.sha256(word.encode()).hexdigest(), 16)
            idx = h % dim
            sign = 1.0 if ((h >> 8) % 2 == 0) else -1.0
            vec[idx] += sign * 2.0

            # Sub-tokens
            for i in range(len(word) - 2):
                tri = word[i:i+3]
                h_tri = int(hashlib.md5(tri.encode()).hexdigest(), 16)
                idx_tri = h_tri % dim
                sign_tri = 0.5 if ((h_tri >> 4) % 2 == 0) else -0.5
                vec[idx_tri] += sign_tri

        # Normalize to unit length (L2 norm)
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        else:
            vec[0] = 1.0
        return vec

    async def embed_texts(self, texts: List[str], task: str = "retrieval.passage") -> List[List[float]]:
        """
        Embeds a list of strings with the specified Jina retrieval task.
        task: 'retrieval.passage' for documents/chunks, 'retrieval.query' for queries.
        """
        if not texts:
            return []

        if not self._is_live:
            return [self._generate_mock_embedding(t) for t in texts]

        # Call live Jina Embeddings API in batches of 32
        batch_size = 32
        all_embeddings: List[List[float]] = []

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key.strip()}"
        }

        async with httpx.AsyncClient(timeout=45.0) as client:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                payload = {
                    "model": self.model,
                    "task": task,
                    "normalized": True,
                    "input": [{"text": t} for t in batch]
                }

                try:
                    response = await client.post(self.API_URL, headers=headers, json=payload)
                    response.raise_for_status()
                    data = response.json()

                    # Extract embeddings ordered by index
                    items = data.get("data", [])
                    sorted_items = sorted(items, key=lambda x: x.get("index", 0))
                    embeddings = [item["embedding"] for item in sorted_items]

                    if embeddings and len(embeddings[0]) != self.dimension:
                        # Auto-update dimension to match model output
                        self.dimension = len(embeddings[0])

                    all_embeddings.extend(embeddings)

                except httpx.HTTPStatusError as e:
                    logger.error(f"Jina API returned error {e.response.status_code}: {e.response.text}")
                    raise RuntimeError(f"Jina AI API error ({e.response.status_code}): {e.response.text}") from e
                except Exception as e:
                    logger.error(f"Failed to connect to Jina API: {e}")
                    raise RuntimeError(f"Embedding request failed: {e}") from e

        return all_embeddings

    async def embed_query(self, query: str) -> List[float]:
        """Embeds a single search query using task='retrieval.query'."""
        embeddings = await self.embed_texts([query], task="retrieval.query")
        return embeddings[0]

    async def embed_passages(self, passages: List[str]) -> List[List[float]]:
        """Embeds multiple document chunks using task='retrieval.passage'."""
        return await self.embed_texts(passages, task="retrieval.passage")
