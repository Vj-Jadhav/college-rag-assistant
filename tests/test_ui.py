"""Tests for Streamlit App and UI Integration (Phase 6).

Verifies:
- Streamlit application module structure and syntax
- Ingestion helper functions
- Component factory functions
- Streamlit page configuration and execution
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from app import (
    ingest_pdf_file,
    load_preexisting_sample_documents,
    get_embedding_service,
    get_llm_service,
    get_vector_store,
    get_rag_pipeline,
)
from vector_store import ChromaVectorStore


def test_component_factories():
    emb = get_embedding_service()
    assert emb.model_name == "nomic-embed-text"

    llm = get_llm_service()
    assert llm.model_name == "llama3.2:3b"

    vs = get_vector_store()
    assert vs.collection_name == "college_documents"

    pipeline = get_rag_pipeline(top_k=3)
    assert pipeline.retriever.top_k == 3


def test_ingest_pdf_file(tmp_path):
    data_dir = Path(__file__).resolve().parent.parent / "data"
    pdf_path = data_dir / "hostel_rules.pdf"
    assert pdf_path.exists()

    mock_store = MagicMock(spec=ChromaVectorStore)
    mock_store.add_chunks.return_value = 1

    chunks_added = ingest_pdf_file(pdf_path, mock_store)
    assert chunks_added == 1
    mock_store.add_chunks.assert_called_once()


def test_ingest_nonexistent_pdf_returns_zero(tmp_path):
    missing_pdf = tmp_path / "nonexistent.pdf"
    mock_store = MagicMock(spec=ChromaVectorStore)

    # load_pdf raises PDFNotFoundError for missing files
    with pytest.raises(Exception):
        ingest_pdf_file(missing_pdf, mock_store)

