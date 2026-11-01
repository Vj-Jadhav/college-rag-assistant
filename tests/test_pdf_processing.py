"""Unit and Integration Tests for PDF Document Processing and Chunking.

Verifies:
- PDF loading via PyPDF
- Text extraction across readable pages
- Preservation of document filename, page number, and path metadata
- Graceful handling of empty pages, non-existent files, and corrupt PDFs
- Multiple PDF batch and directory loading
- Chunking with default chunk size (1500) and overlap (200)
- Preservation of source metadata for every chunk (source_filename, page_number, chunk_id)
- Multi-chunk splitting on larger documents with boundary-aware splits
"""

import pytest
from pathlib import Path
import io
from document_loader import (
    load_pdf,
    load_multiple_pdfs,
    load_directory,
    PDFNotFoundError,
    PDFCorruptError,
    DocumentLoaderError,
)
from chunker import (
    split_single_document,
    split_text_into_chunks,
    find_natural_split_point,
)


@pytest.fixture
def sample_pdf_dir():
    return Path(__file__).resolve().parent.parent / "data"


class TestDocumentLoader:
    """Test suite for document_loader.py."""

    def test_load_valid_pdf(self, sample_pdf_dir):
        pdf_path = sample_pdf_dir / "admission_guidelines.pdf"
        assert pdf_path.exists(), "Sample PDF must exist"

        pages = load_pdf(pdf_path)
        assert len(pages) == 2

        # Verify page 1 metadata
        page1 = pages[0]
        assert page1["source_filename"] == "admission_guidelines.pdf"
        assert page1["page_number"] == 1
        assert page1["total_pages"] == 2
        assert "ADMISSION GUIDELINES" in page1["text"]
        assert "75 percent" in page1["text"]

        # Verify page 2 metadata
        page2 = pages[1]
        assert page2["source_filename"] == "admission_guidelines.pdf"
        assert page2["page_number"] == 2
        assert "FEE STRUCTURE" in page2["text"]
        assert "INR 65,000" in page2["text"]

    def test_empty_page_handling(self, sample_pdf_dir):
        # hostel_rules.pdf contains page 1 with content, page 2 blank
        pdf_path = sample_pdf_dir / "hostel_rules.pdf"
        assert pdf_path.exists()

        # With skip_empty=True (default), blank page 2 must be omitted
        pages = load_pdf(pdf_path, skip_empty=True)
        assert len(pages) == 1
        assert pages[0]["page_number"] == 1
        assert "CAMPUS HOSTEL RULES" in pages[0]["text"]

        # With skip_empty=False, blank page 2 is returned with empty text
        all_pages = load_pdf(pdf_path, skip_empty=False)
        assert len(all_pages) == 2
        assert all_pages[1]["page_number"] == 2
        assert all_pages[1]["text"] == ""

    def test_nonexistent_pdf_raises_error(self, tmp_path):
        missing = tmp_path / "does_not_exist.pdf"
        with pytest.raises(PDFNotFoundError):
            load_pdf(missing)

    def test_corrupt_pdf_raises_error(self, tmp_path):
        corrupt = tmp_path / "corrupt_document.pdf"
        corrupt.write_text("NOT A VALID PDF HEADER", encoding="utf-8")
        with pytest.raises(PDFCorruptError):
            load_pdf(corrupt)

    def test_load_multiple_pdfs(self, sample_pdf_dir):
        files = [
            sample_pdf_dir / "admission_guidelines.pdf",
            sample_pdf_dir / "academic_regulations.pdf",
        ]
        pages = load_multiple_pdfs(files)
        # admission_guidelines has 2 pages, academic_regulations has 3 pages = 5 pages total
        assert len(pages) == 5

        filenames = {p["source_filename"] for p in pages}
        assert filenames == {"admission_guidelines.pdf", "academic_regulations.pdf"}

    def test_load_directory(self, sample_pdf_dir):
        pages = load_directory(sample_pdf_dir)
        # admission (2) + academic (3) + hostel (1 valid, 1 blank skipped) = 6 pages
        assert len(pages) == 6
        for p in pages:
            assert "source_filename" in p
            assert "page_number" in p
            assert "text" in p
            assert len(p["text"]) > 0


class TestChunker:
    """Test suite for chunker.py."""

    def test_invalid_parameters_raise_error(self):
        doc = {"text": "Valid text", "source_filename": "test.pdf", "page_number": 1}
        with pytest.raises(ValueError):
            split_single_document(doc, chunk_size=0)
        with pytest.raises(ValueError):
            split_single_document(doc, chunk_size=500, chunk_overlap=-1)
        with pytest.raises(ValueError):
            split_single_document(doc, chunk_size=500, chunk_overlap=500)
        with pytest.raises(ValueError):
            split_single_document(doc, chunk_size=500, chunk_overlap=600)

    def test_empty_document_produces_no_chunks(self):
        doc = {"text": "   ", "source_filename": "test.pdf", "page_number": 1}
        chunks = split_single_document(doc)
        assert chunks == []

    def test_single_chunk_when_text_fits(self):
        text = "Short college announcement regarding holiday."
        doc = {"text": text, "source_filename": "circular.pdf", "page_number": 3}
        chunks = split_single_document(doc, chunk_size=1500, chunk_overlap=200)

        assert len(chunks) == 1
        c = chunks[0]
        assert c["chunk_id"] == "circular.pdf_p3_c0"
        assert c["text"] == text
        assert c["source_filename"] == "circular.pdf"
        assert c["page_number"] == 3
        assert c["chunk_index"] == 0

    def test_multi_chunk_splitting_and_overlap(self):
        # Create text of ~1000 characters
        paragraphs = [
            f"Paragraph {i}: The department of computer science offers advanced laboratory equipment for students."
            for i in range(12)
        ]
        full_text = "\n\n".join(paragraphs)
        doc = {"text": full_text, "source_filename": "cse_guide.pdf", "page_number": 2}

        # Chunk with size 300 and overlap 50
        chunks = split_single_document(doc, chunk_size=300, chunk_overlap=50)
        assert len(chunks) > 1

        for idx, chunk in enumerate(chunks):
            assert chunk["source_filename"] == "cse_guide.pdf"
            assert chunk["page_number"] == 2
            assert chunk["chunk_id"] == f"cse_guide.pdf_p2_c{idx}"
            assert chunk["chunk_index"] == idx
            assert len(chunk["text"]) <= 350  # Respects boundary within margin
            assert len(chunk["text"]) > 0

    def test_split_text_into_chunks_default_settings(self, sample_pdf_dir):
        pages = load_directory(sample_pdf_dir)
        # Default settings: chunk_size=1500, chunk_overlap=200
        chunks = split_text_into_chunks(pages, chunk_size=1500, chunk_overlap=200)
        assert len(chunks) >= len(pages)

        # Verify required metadata presence on every chunk
        for chunk in chunks:
            assert "source_filename" in chunk
            assert "page_number" in chunk
            assert "chunk_id" in chunk
            assert "chunk_index" in chunk
            assert "text" in chunk
            assert isinstance(chunk["source_filename"], str)
            assert isinstance(chunk["page_number"], int)
            assert isinstance(chunk["chunk_id"], str)
            assert len(chunk["text"]) > 0

    def test_natural_split_point_priority(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        # Split point should favor sentence boundary [.!?] over slicing mid-sentence
        split_idx = find_natural_split_point(text, target_end=35, min_end=20)
        # "Sentence two. " ends at index 28
        assert text[split_idx - 1] in [".", " "]

