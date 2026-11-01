"""Unit and Integration Tests for Vector Database and Embeddings Storage.

Verifies:
- ChromaDB persistent storage inside local chroma_db/
- Chunk insertion with rich metadata (filename, page number, chunk_id)
- Multi-document support
- Duplicate ingestion prevention (skip vs. overwrite update)
- Safe document deletion and chunk deletion
- Persistence across application / client restarts
- Query retrieval and cosine similarity ranking
- Reset and statistics reporting
"""

import pytest
from pathlib import Path
from unittest.mock import MagicMock
from vector_store import (
    ChromaVectorStore,
    VectorStoreError,
)
from embeddings import (
    OllamaEmbeddingService,
    OllamaChromaEmbeddingFunction,
)


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Temporary directory for isolated ChromaDB tests."""
    return tmp_path / "test_chroma_db"


@pytest.fixture
def sample_chunks():
    """Sample chunk collection from multiple PDF documents."""
    return [
        {
            "chunk_id": "admission_guidelines.pdf_p1_c0",
            "text": "B.Tech admission eligibility requires minimum 75% aggregate in 10+2 PCM.",
            "source_filename": "admission_guidelines.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "chunk_id": "admission_guidelines.pdf_p2_c0",
            "text": "Annual tuition fee is INR 65,000 per semester. Scholarships available for top 500 ranks.",
            "source_filename": "admission_guidelines.pdf",
            "page_number": 2,
            "chunk_index": 0,
        },
        {
            "chunk_id": "academic_regulations.pdf_p1_c0",
            "text": "Mandatory minimum attendance is 75 percent for end-semester examination eligibility.",
            "source_filename": "academic_regulations.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
        {
            "chunk_id": "hostel_rules.pdf_p1_c0",
            "text": "Hostel curfew timing is 9:30 PM on weekdays. Mess dining timings are strictly followed.",
            "source_filename": "hostel_rules.pdf",
            "page_number": 1,
            "chunk_index": 0,
        },
    ]


@pytest.fixture
def sample_embeddings():
    """Distinct synthetic normalized embedding vectors (dimension 4)."""
    return [
        [1.0, 0.0, 0.0, 0.0],  # admission p1
        [0.9, 0.1, 0.0, 0.0],  # admission p2
        [0.0, 1.0, 0.0, 0.0],  # academic p1
        [0.0, 0.0, 1.0, 0.0],  # hostel p1
    ]


class TestChromaVectorStore:
    """Test suite for ChromaVectorStore."""

    def test_initialization(self, temp_chroma_dir):
        store = ChromaVectorStore(persist_dir=temp_chroma_dir, collection_name="test_col")
        assert store.count() == 0
        assert store.get_indexed_documents() == []
        stats = store.get_collection_stats()
        assert stats["total_chunks"] == 0
        assert stats["total_documents"] == 0

    def test_add_chunks_with_metadata(self, temp_chroma_dir, sample_chunks, sample_embeddings):
        store = ChromaVectorStore(persist_dir=temp_chroma_dir)
        inserted = store.add_chunks(sample_chunks, sample_embeddings)
        assert inserted == 4
        assert store.count() == 4

        # Verify indexed documents list
        docs = store.get_indexed_documents()
        assert len(docs) == 3
        assert docs == ["academic_regulations.pdf", "admission_guidelines.pdf", "hostel_rules.pdf"]

        # Verify has_document
        assert store.has_document("admission_guidelines.pdf") is True
        assert store.has_document("nonexistent.pdf") is False

    def test_persistence_across_restarts(self, temp_chroma_dir, sample_chunks, sample_embeddings):
        # 1. First session: insert vectors
        store1 = ChromaVectorStore(persist_dir=temp_chroma_dir, collection_name="persistent_col")
        store1.add_chunks(sample_chunks, sample_embeddings)
        assert store1.count() == 4
        del store1

        # 2. Second session: simulate application restart by creating new instance at same path
        store2 = ChromaVectorStore(persist_dir=temp_chroma_dir, collection_name="persistent_col")
        assert store2.count() == 4
        assert store2.get_indexed_documents() == [
            "academic_regulations.pdf",
            "admission_guidelines.pdf",
            "hostel_rules.pdf",
        ]

        # 3. Query persisted collection
        matches = store2.query(query_embedding=[1.0, 0.0, 0.0, 0.0], n_results=1)
        assert len(matches) == 1
        assert matches[0]["chunk_id"] == "admission_guidelines.pdf_p1_c0"
        assert matches[0]["source_filename"] == "admission_guidelines.pdf"
        assert matches[0]["page_number"] == 1

    def test_duplicate_prevention_skip_vs_overwrite(self, temp_chroma_dir):
        store = ChromaVectorStore(persist_dir=temp_chroma_dir)

        initial_chunk = [{
            "chunk_id": "syllabus.pdf_p1_c0",
            "text": "Initial syllabus content version 1",
            "source_filename": "syllabus.pdf",
            "page_number": 1,
            "chunk_index": 0,
        }]
        initial_vec = [[0.5, 0.5, 0.5, 0.5]]

        # Ingest first time
        store.add_chunks(initial_chunk, initial_vec)
        assert store.count() == 1

        # Re-upload with overwrite=False -> skipped
        store.add_chunks(initial_chunk, initial_vec, overwrite=False)
        assert store.count() == 1

        # Re-upload with overwrite=True and updated content -> cleanly replaced
        updated_chunks = [
            {
                "chunk_id": "syllabus.pdf_p1_c0",
                "text": "Updated syllabus content version 2",
                "source_filename": "syllabus.pdf",
                "page_number": 1,
                "chunk_index": 0,
            },
            {
                "chunk_id": "syllabus.pdf_p1_c1",
                "text": "Additional syllabus section",
                "source_filename": "syllabus.pdf",
                "page_number": 1,
                "chunk_index": 1,
            },
        ]
        updated_vecs = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        store.add_chunks(updated_chunks, updated_vecs, overwrite=True)

        assert store.count() == 2
        res = store.query(query_embedding=[1.0, 0.0, 0.0, 0.0], n_results=1)
        assert "version 2" in res[0]["text"]

    def test_document_deletion(self, temp_chroma_dir, sample_chunks, sample_embeddings):
        store = ChromaVectorStore(persist_dir=temp_chroma_dir)
        store.add_chunks(sample_chunks, sample_embeddings)
        assert store.count() == 4

        # Delete admission_guidelines.pdf (which has 2 chunks)
        deleted_count = store.delete_document("admission_guidelines.pdf")
        assert deleted_count == 2
        assert store.count() == 2
        assert store.has_document("admission_guidelines.pdf") is False
        assert store.has_document("academic_regulations.pdf") is True

    def test_single_chunk_deletion(self, temp_chroma_dir, sample_chunks, sample_embeddings):
        store = ChromaVectorStore(persist_dir=temp_chroma_dir)
        store.add_chunks(sample_chunks, sample_embeddings)

        success = store.delete_chunk("hostel_rules.pdf_p1_c0")
        assert success is True
        assert store.count() == 3
        assert store.has_document("hostel_rules.pdf") is False

    def test_query_ranking_and_metadata_filter(self, temp_chroma_dir, sample_chunks, sample_embeddings):
        store = ChromaVectorStore(persist_dir=temp_chroma_dir)
        store.add_chunks(sample_chunks, sample_embeddings)

        # Query targeting academic vector [0.0, 1.0, 0.0, 0.0]
        results = store.query(query_embedding=[0.0, 1.0, 0.0, 0.0], n_results=2)
        assert len(results) == 2
        top_match = results[0]
        assert top_match["source_filename"] == "academic_regulations.pdf"
        assert top_match["similarity"] > 0.99

        # Filter query by source_filename
        filtered = store.query(
            query_embedding=[1.0, 0.0, 0.0, 0.0],
            where={"source_filename": "hostel_rules.pdf"},
            n_results=2,
        )
        assert len(filtered) == 1
        assert filtered[0]["source_filename"] == "hostel_rules.pdf"

    def test_reset_collection(self, temp_chroma_dir, sample_chunks, sample_embeddings):
        store = ChromaVectorStore(persist_dir=temp_chroma_dir)
        store.add_chunks(sample_chunks, sample_embeddings)
        assert store.count() == 4

        store.reset()
        assert store.count() == 0
        assert store.get_indexed_documents() == []

    def test_embedding_service_auto_embedding(self, temp_chroma_dir):
        mock_service = MagicMock(spec=OllamaEmbeddingService)
        mock_service.embed_chunks.return_value = [[0.1, 0.2, 0.3, 0.4]]

        store = ChromaVectorStore(persist_dir=temp_chroma_dir, embedding_service=mock_service)
        chunk = [{
            "chunk_id": "test_c1",
            "text": "Autonomous college campus",
            "source_filename": "test.pdf",
            "page_number": 1,
            "chunk_index": 0,
        }]

        # Call add_chunks without passing embeddings
        store.add_chunks(chunk)
        assert store.count() == 1
        mock_service.embed_chunks.assert_called_once_with(chunk)


class TestOllamaEmbeddingAdapter:
    """Test suite for Chroma embedding function adapter in embeddings.py."""

    def test_chroma_embedding_function_adapter(self):
        mock_service = MagicMock(spec=OllamaEmbeddingService)
        mock_service.get_embeddings.return_value = [[0.1, 0.2], [0.3, 0.4]]

        adapter = OllamaChromaEmbeddingFunction(mock_service)
        result = adapter(["Doc 1", "Doc 2"])
        assert result == [[0.1, 0.2], [0.3, 0.4]]
        mock_service.get_embeddings.assert_called_once_with(["Doc 1", "Doc 2"])
