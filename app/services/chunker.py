import hashlib
import re
from typing import List, Optional, Tuple
from app.models.document import DocumentChunk, RawDocument
from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RecursiveChunker:
    """
    Recursively splits text into overlapping chunks using natural linguistic boundaries:
    paragraphs, sentences, and words. Preserves exact page numbers and provenance.
    """

    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ", ""]

    def __init__(self, chunk_size: Optional[int] = None, chunk_overlap: Optional[int] = None):
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(f"chunk_overlap ({self.chunk_overlap}) must be strictly less than chunk_size ({self.chunk_size})")

    def _generate_chunk_id(self, doc_name: str, page_number: Optional[int], chunk_index: int) -> str:
        """Creates a stable, reproducible chunk ID."""
        base = f"{doc_name}:{page_number or 1}:{chunk_index}"
        short_hash = hashlib.md5(base.encode()).hexdigest()[:8]
        clean_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", doc_name)[:20]
        return f"{clean_name}_p{page_number or 1}_c{chunk_index}_{short_hash}"

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively splits text using hierarchical separators."""
        text = text.strip()
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        # Find the highest priority separator present in the text
        separator = separators[-1]
        new_separators = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                break
            if sep in text:
                separator = sep
                new_separators = separators[i + 1:]
                break

        # Split by separator
        if separator:
            splits = text.split(separator)
        else:
            # Character by character fallback
            splits = list(text)

        # Merge splits back together up to chunk_size with overlap
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for s in splits:
            s_len = len(s) + (len(separator) if separator and current_chunk else 0)

            if current_len + s_len <= self.chunk_size:
                current_chunk.append(s)
                current_len += s_len
            else:
                if current_chunk:
                    merged = separator.join(current_chunk).strip()
                    if merged:
                        chunks.append(merged)

                    # Build overlap window from the tail of current_chunk
                    overlap_parts: List[str] = []
                    overlap_len = 0
                    for part in reversed(current_chunk):
                        part_len = len(part) + (len(separator) if separator and overlap_parts else 0)
                        if overlap_len + part_len <= self.chunk_overlap:
                            overlap_parts.insert(0, part)
                            overlap_len += part_len
                        else:
                            break
                    current_chunk = overlap_parts
                    current_len = overlap_len

                # Handle pieces that are individually larger than chunk_size
                if len(s) > self.chunk_size:
                    if new_separators:
                        sub_chunks = self._split_text(s, new_separators)
                        chunks.extend(sub_chunks)
                    else:
                        # Hard character slice if no more separators exist
                        for start in range(0, len(s), self.chunk_size - self.chunk_overlap):
                            chunks.append(s[start:start + self.chunk_size])
                else:
                    current_chunk.append(s)
                    current_len += len(s) + (len(separator) if separator and len(current_chunk) > 1 else 0)

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged and (not chunks or merged != chunks[-1]):
                chunks.append(merged)

        return chunks

    def chunk_document(self, raw_doc: RawDocument) -> List[DocumentChunk]:
        """
        Splits a RawDocument into a list of DocumentChunk instances.
        If pages are present, processes page-by-page to guarantee page citation accuracy.
        """
        all_chunks: List[DocumentChunk] = []
        global_chunk_idx = 0
        char_offset = 0

        if raw_doc.pages and len(raw_doc.pages) > 0:
            # Process page by page
            for page in raw_doc.pages:
                page_num = page.get("page_number", 1)
                page_text = page.get("text", "")
                if not page_text.strip():
                    continue

                page_pieces = self._split_text(page_text, self.SEPARATORS)
                for piece in page_pieces:
                    if not piece.strip():
                        continue
                    start_char = char_offset + page_text.find(piece)
                    end_char = start_char + len(piece)

                    chunk_id = self._generate_chunk_id(raw_doc.name, page_num, global_chunk_idx)
                    chunk = DocumentChunk(
                        id=chunk_id,
                        document_name=raw_doc.name,
                        content=piece,
                        page_number=page_num,
                        chunk_index=global_chunk_idx,
                        total_chunks=0,  # Updated after collection
                        start_char=max(0, start_char),
                        end_char=max(0, end_char),
                        metadata={
                            "file_type": raw_doc.file_type,
                            "length_chars": len(piece),
                            **raw_doc.metadata
                        }
                    )
                    all_chunks.append(chunk)
                    global_chunk_idx += 1
                char_offset += len(page_text)
        else:
            # Single-pass chunking
            raw_text = raw_doc.content
            pieces = self._split_text(raw_text, self.SEPARATORS)
            for piece in pieces:
                if not piece.strip():
                    continue
                start_char = raw_text.find(piece)
                end_char = start_char + len(piece)

                chunk_id = self._generate_chunk_id(raw_doc.name, 1, global_chunk_idx)
                chunk = DocumentChunk(
                    id=chunk_id,
                    document_name=raw_doc.name,
                    content=piece,
                    page_number=1,
                    chunk_index=global_chunk_idx,
                    total_chunks=0,
                    start_char=max(0, start_char),
                    end_char=max(0, end_char),
                    metadata={
                        "file_type": raw_doc.file_type,
                        "length_chars": len(piece),
                        **raw_doc.metadata
                    }
                )
                all_chunks.append(chunk)
                global_chunk_idx += 1

        # Backfill total_chunks
        total = len(all_chunks)
        for chunk in all_chunks:
            chunk.total_chunks = total

        logger.info(f"Chunked document '{raw_doc.name}' into {total} chunks (avg {sum(len(c.content) for c in all_chunks) // max(1, total)} chars/chunk)")
        return all_chunks
