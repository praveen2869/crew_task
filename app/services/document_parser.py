import io
import re
from typing import Optional, List, Dict, Any
from pypdf import PdfReader
from app.models.document import RawDocument
from app.core.logging import get_logger

logger = get_logger(__name__)


class DocumentParser:
    """Parses PDF, TXT, and Markdown files into normalized RawDocument structures."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    @classmethod
    def clean_text(cls, text: str) -> str:
        """Cleans control characters and normalizes excessive whitespace."""
        if not text:
            return ""
        # Normalize CRLF to LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        # Replace non-printable characters except newlines/tabs
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
        # Collapse excessive newlines (max 2 consecutive)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @classmethod
    def parse_pdf(cls, file_bytes: bytes, filename: str) -> RawDocument:
        """Extracts text from a PDF file preserving per-page breakdown."""
        reader = PdfReader(io.BytesIO(file_bytes))
        pages_data: List[Dict[str, Any]] = []
        full_text_parts: List[str] = []

        total_pages = len(reader.pages)
        logger.info(f"Parsing PDF '{filename}' with {total_pages} pages")

        for idx, page in enumerate(reader.pages, start=1):
            try:
                page_text = page.extract_text() or ""
                cleaned = cls.clean_text(page_text)
                if cleaned:
                    pages_data.append({"page_number": idx, "text": cleaned})
                    full_text_parts.append(f"--- Page {idx} ---\n" + cleaned)
            except Exception as e:
                logger.warning(f"Failed to extract text from page {idx} of '{filename}': {e}")

        full_content = "\n\n".join(full_text_parts)
        return RawDocument(
            name=filename,
            content=full_content,
            file_type="pdf",
            pages=pages_data,
            metadata={"total_pages": total_pages}
        )

    @classmethod
    def parse_txt(cls, file_bytes: bytes, filename: str) -> RawDocument:
        """Decodes plain text files with UTF-8 / fallback encodings."""
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1", errors="replace")

        cleaned = cls.clean_text(content)
        return RawDocument(
            name=filename,
            content=cleaned,
            file_type="txt",
            pages=[{"page_number": 1, "text": cleaned}],
            metadata={"char_count": len(cleaned)}
        )

    @classmethod
    def parse_markdown(cls, file_bytes: bytes, filename: str) -> RawDocument:
        """Parses markdown files, maintaining structure."""
        try:
            content = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            content = file_bytes.decode("latin-1", errors="replace")

        cleaned = cls.clean_text(content)
        return RawDocument(
            name=filename,
            content=cleaned,
            file_type="md",
            pages=[{"page_number": 1, "text": cleaned}],
            metadata={"char_count": len(cleaned)}
        )

    @classmethod
    def parse_file(cls, file_bytes: bytes, filename: str) -> RawDocument:
        """Inspects extension and delegates to appropriate parser."""
        ext = "." + filename.split(".")[-1].lower() if "." in filename else ""

        if ext == ".pdf":
            return cls.parse_pdf(file_bytes, filename)
        elif ext == ".txt":
            return cls.parse_txt(file_bytes, filename)
        elif ext == ".md":
            return cls.parse_markdown(file_bytes, filename)
        else:
            raise ValueError(f"Unsupported file format: '{ext}'. Supported formats: {cls.SUPPORTED_EXTENSIONS}")
