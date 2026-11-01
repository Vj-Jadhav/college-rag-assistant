"""Document Loader Module.

Ingests and extracts text from PDF documents using PyPDF.
Preserves page-level metadata (filename, page number) and
handles empty pages, corrupt files, and batch processing.
"""

from pathlib import Path
from typing import List, Dict, Any, Union, Optional
import logging
from pypdf import PdfReader
from pypdf.errors import PdfReadError

logger = logging.getLogger(__name__)


class DocumentLoaderError(Exception):
    """Base exception for document loading and parsing errors."""
    pass


class PDFNotFoundError(DocumentLoaderError):
    """Raised when the specified PDF file cannot be found."""
    pass


class PDFCorruptError(DocumentLoaderError):
    """Raised when a PDF file is corrupt, encrypted, or unreadable."""
    pass


def load_pdf(
    file_path: Union[Path, str],
    skip_empty: bool = True,
    clean_whitespace: bool = True,
) -> List[Dict[str, Any]]:
    """Extract text content and page metadata from a single PDF document.

    Args:
        file_path: Absolute or relative path to the PDF file.
        skip_empty: Whether to omit pages with no extractable text (default: True).
        clean_whitespace: Normalize inconsistent spaces and excessive blank lines (default: True).

    Returns:
        List of dictionaries with page text and source metadata:
        [
            {
                "text": str,
                "source_filename": str,
                "page_number": int,   # 1-indexed
                "total_pages": int,
                "file_path": str,
            },
            ...
        ]

    Raises:
        PDFNotFoundError: If file_path does not exist.
        PDFCorruptError: If PDF structure is invalid or unreadable.
    """
    path = Path(file_path).resolve()

    if not path.exists():
        raise PDFNotFoundError(f"PDF document not found: {path}")

    if not path.is_file():
        raise PDFNotFoundError(f"Path is not a valid file: {path}")

    if path.suffix.lower() != ".pdf":
        logger.warning(f"File {path.name} does not have a .pdf extension.")

    try:
        reader = PdfReader(str(path))
    except (PdfReadError, Exception) as exc:
        raise PDFCorruptError(f"Failed to open/parse PDF {path.name}: {exc}") from exc

    if reader.is_encrypted:
        try:
            # Attempt empty password decryption
            reader.decrypt("")
        except Exception as exc:
            raise PDFCorruptError(f"PDF {path.name} is password-protected: {exc}") from exc

    total_pages = len(reader.pages)
    if total_pages == 0:
        logger.warning(f"PDF document has 0 pages: {path.name}")
        return []

    documents: List[Dict[str, Any]] = []

    for idx, page in enumerate(reader.pages):
        page_num = idx + 1
        try:
            raw_text = page.extract_text() or ""
        except Exception as exc:
            logger.warning(f"Error extracting text from page {page_num} of {path.name}: {exc}")
            raw_text = ""

        if clean_whitespace and raw_text:
            # Clean up carriage returns and excessive whitespace while preserving paragraph breaks
            lines = [line.strip() for line in raw_text.splitlines()]
            raw_text = "\n".join(lines).strip()

        if skip_empty and not raw_text.strip():
            logger.debug(f"Skipping empty text page {page_num} in {path.name}")
            continue

        documents.append({
            "text": raw_text,
            "source_filename": path.name,
            "page_number": page_num,
            "total_pages": total_pages,
            "file_path": str(path),
        })

    logger.info(f"Loaded {len(documents)} text page(s) from {path.name} (total {total_pages} pages).")
    return documents


def load_multiple_pdfs(
    file_paths: List[Union[Path, str]],
    skip_empty: bool = True,
    clean_whitespace: bool = True,
    ignore_corrupt: bool = True,
) -> List[Dict[str, Any]]:
    """Extract text from multiple PDF files.

    Args:
        file_paths: List of file paths to process.
        skip_empty: Whether to skip empty pages.
        clean_whitespace: Normalize spacing.
        ignore_corrupt: If True, log warnings for corrupt files rather than aborting.

    Returns:
        List of all document page dictionaries across all input PDFs.
    """
    all_pages: List[Dict[str, Any]] = []

    for fp in file_paths:
        try:
            pages = load_pdf(fp, skip_empty=skip_empty, clean_whitespace=clean_whitespace)
            all_pages.extend(pages)
        except (PDFNotFoundError, PDFCorruptError) as exc:
            if ignore_corrupt:
                logger.error(f"Skipping unreadable document {fp}: {exc}")
            else:
                raise

    return all_pages


def load_directory(
    directory_path: Union[Path, str],
    recursive: bool = True,
    skip_empty: bool = True,
    clean_whitespace: bool = True,
    ignore_corrupt: bool = True,
) -> List[Dict[str, Any]]:
    """Scan a directory and load all discoverable PDF documents.

    Args:
        directory_path: Directory containing PDF files.
        recursive: Whether to search subdirectories recursively.
        skip_empty: Skip empty pages.
        clean_whitespace: Clean up whitespace in extracted text.
        ignore_corrupt: Skip corrupt files with a logged error.

    Returns:
        List of document page dictionaries.
    """
    dir_path = Path(directory_path).resolve()
    if not dir_path.exists() or not dir_path.is_dir():
        raise DocumentLoaderError(f"Directory not found or invalid: {dir_path}")

    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(dir_path.glob(pattern))

    logger.info(f"Found {len(pdf_files)} PDF file(s) in {dir_path}")
    return load_multiple_pdfs(
        pdf_files,
        skip_empty=skip_empty,
        clean_whitespace=clean_whitespace,
        ignore_corrupt=ignore_corrupt,
    )
