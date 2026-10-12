import pytest
from app.services.document_parser import DocumentParser
from app.services.chunker import RecursiveChunker
from app.models.document import RawDocument


def test_document_parser_txt():
    content = b"Line 1\nLine 2\nLine 3"
    raw = DocumentParser.parse_file(content, "test.txt")
    assert raw.file_type == "txt"
    assert "Line 1" in raw.content
    assert len(raw.pages) == 1
    assert raw.pages[0]["page_number"] == 1


def test_document_parser_markdown():
    content = b"# Header 1\n\nSome text under header.\n\n## Subheader\nMore details."
    raw = DocumentParser.parse_file(content, "test.md")
    assert raw.file_type == "md"
    assert "Header 1" in raw.content


def test_document_parser_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported file format"):
        DocumentParser.parse_file(b"content", "file.xlsx")


def test_recursive_chunker_basic():
    chunker = RecursiveChunker(chunk_size=100, chunk_overlap=20)
    text = "First sentence here. Second sentence follows. Third sentence comes next. Fourth sentence wraps up."
    raw = RawDocument(name="sample.txt", content=text, file_type="txt", pages=[{"page_number": 1, "text": text}])
    chunks = chunker.chunk_document(raw)

    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.document_name == "sample.txt"
        assert chunk.page_number == 1
        assert len(chunk.content) <= 120
        assert chunk.total_chunks == len(chunks)


def test_recursive_chunker_multi_page():
    chunker = RecursiveChunker(chunk_size=150, chunk_overlap=30)
    raw = RawDocument(
        name="guide.pdf",
        content="",
        file_type="pdf",
        pages=[
            {"page_number": 1, "text": "Page one content with information about security policies."},
            {"page_number": 2, "text": "Page two content detailing API key rotation and cryptographic controls."}
        ]
    )
    chunks = chunker.chunk_document(raw)
    pages = [c.page_number for c in chunks]

    assert 1 in pages
    assert 2 in pages
    assert all(c.document_name == "guide.pdf" for c in chunks)


def test_chunker_invalid_overlap():
    with pytest.raises(ValueError, match="chunk_overlap.*must be strictly less than"):
        RecursiveChunker(chunk_size=100, chunk_overlap=100)
