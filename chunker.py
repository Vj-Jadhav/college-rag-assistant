"""Text Chunking Module.

Splits document text into overlapping passages while preserving
rich metadata (source filename, page number, chunk identifier)
for vector search and retrieval-augmented generation.
"""

from typing import List, Dict, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)


class ChunkerError(Exception):
    """Base exception for chunking errors."""
    pass


def find_natural_split_point(text: str, target_end: int, min_end: int) -> int:
    """Find a natural boundary (paragraph, sentence, or word boundary) near target_end.

    Args:
        text: Full source text string.
        target_end: Ideal end index based on chunk_size.
        min_end: Minimum acceptable end index to enforce meaningful chunk length.

    Returns:
        int: Best split index in [min_end, target_end] or target_end if no break found.
    """
    if target_end >= len(text):
        return len(text)

    search_region = text[min_end:target_end]

    # Priority 1: Double newline (paragraph boundary)
    p_break = search_region.rfind("\n\n")
    if p_break != -1:
        return min_end + p_break + 2

    # Priority 2: Single newline
    n_break = search_region.rfind("\n")
    if n_break != -1:
        return min_end + n_break + 1

    # Priority 3: Sentence ending (. , ! , ? )
    sent_breaks = [m.end() for m in re.finditer(r"[.!?]\s+", search_region)]
    if sent_breaks:
        return min_end + sent_breaks[-1]

    # Priority 4: Word boundary (whitespace)
    w_break = search_region.rfind(" ")
    if w_break != -1:
        return min_end + w_break + 1

    # Fallback: Hard cut at target_end
    return target_end


def split_single_document(
    document: Dict[str, Any],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """Split a single document page into chunks with metadata preservation.

    Args:
        document: Dictionary containing 'text', 'source_filename', and 'page_number'.
        chunk_size: Maximum characters per chunk (default: 1500).
        chunk_overlap: Overlap characters between consecutive chunks (default: 200).

    Returns:
        List of chunk dictionaries with metadata:
        [
            {
                "chunk_id": str,
                "text": str,
                "source_filename": str,
                "page_number": int,
                "chunk_index": int,
                "start_char": int,
                "end_char": int,
            },
            ...
        ]
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}")
    if chunk_overlap >= chunk_size:
        raise ValueError(f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})")

    raw_text = document.get("text", "")
    text = raw_text.strip()
    source_filename = document.get("source_filename", "unknown.pdf")
    page_number = document.get("page_number", 1)

    if not text:
        return []

    text_len = len(text)

    # If text fits within single chunk
    if text_len <= chunk_size:
        chunk_id = f"{source_filename}_p{page_number}_c0"
        return [{
            "chunk_id": chunk_id,
            "text": text,
            "source_filename": source_filename,
            "page_number": page_number,
            "chunk_index": 0,
            "start_char": 0,
            "end_char": text_len,
        }]

    chunks: List[Dict[str, Any]] = []
    start = 0
    chunk_idx = 0

    # Minimum threshold to search for natural boundary (80% of chunk_size)
    min_boundary_offset = int(chunk_size * 0.8)

    while start < text_len:
        target_end = min(start + chunk_size, text_len)

        if target_end == text_len:
            actual_end = text_len
        else:
            min_end = max(start + min_boundary_offset, start + 1)
            actual_end = find_natural_split_point(text, target_end, min_end)

        chunk_content = text[start:actual_end].strip()

        if chunk_content:
            chunk_id = f"{source_filename}_p{page_number}_c{chunk_idx}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_content,
                "source_filename": source_filename,
                "page_number": page_number,
                "chunk_index": chunk_idx,
                "start_char": start,
                "end_char": actual_end,
            })
            chunk_idx += 1

        if actual_end >= text_len:
            break

        # Advance start position with overlap
        next_start = actual_end - chunk_overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start

    return chunks


def split_text_into_chunks(
    documents: List[Dict[str, Any]],
    chunk_size: int = 1500,
    chunk_overlap: int = 200,
) -> List[Dict[str, Any]]:
    """Split a collection of document pages into chunks.

    Args:
        documents: List of document dictionaries (from document_loader).
        chunk_size: Maximum character length per chunk (default: 1500).
        chunk_overlap: Overlapping character count (default: 200).

    Returns:
        Consolidated list of all chunk dictionaries across all documents.
    """
    all_chunks: List[Dict[str, Any]] = []

    for doc in documents:
        page_chunks = split_single_document(
            document=doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        all_chunks.extend(page_chunks)

    logger.info(f"Split {len(documents)} document page(s) into {len(all_chunks)} chunk(s).")
    return all_chunks
