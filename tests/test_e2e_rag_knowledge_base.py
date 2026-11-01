"""End-to-End Integration Test for Document Processing, Chunking, and Vector Store.

Verifies the complete Phase 1-4 pipeline:
1. Loading college documents from data/ via PyPDF
2. Chunking text with preserved metadata
3. Storing in persistent ChromaDB
4. Querying and similarity matching
5. Persistence across store reinstantiation
6. Document deduplication and safe cleanup
"""

import pytest
from pathlib import Path
from document_loader import load_directory
from chunker import split_text_into_chunks
from vector_store import ChromaVectorStore


def test_full_knowledge_base_pipeline(tmp_path):
    # 1. Load documents from project data directory
    data_dir = Path(__file__).resolve().parent.parent / "data"
    assert data_dir.exists()

    pages = load_directory(data_dir)
    assert len(pages) >= 5, "Expected at least 5 readable pages across sample PDFs"

    # 2. Chunk documents
    chunks = split_text_into_chunks(pages, chunk_size=1500, chunk_overlap=200)
    assert len(chunks) >= len(pages)

    # 3. Create synthetic test embeddings with dimension 8
    # (Unit-test isolation without relying on external/Ollama server online)
    synthetic_embeddings = []
    for idx, c in enumerate(chunks):
        # Deterministic distinct vector per document
        vec = [0.0] * 8
        if "admission" in c["source_filename"]:
            vec[0] = 1.0
            vec[1] = 0.1 * (idx % 3)
        elif "academic" in c["source_filename"]:
            vec[2] = 1.0
            vec[3] = 0.1 * (idx % 3)
        else:
            vec[4] = 1.0
            vec[5] = 0.1 * (idx % 3)
        synthetic_embeddings.append(vec)

    # 4. Ingest into persistent ChromaDB
    chroma_dir = tmp_path / "e2e_chroma"
    store = ChromaVectorStore(persist_dir=chroma_dir, collection_name="college_e2e")
    inserted = store.add_chunks(chunks, synthetic_embeddings)
    assert inserted == len(chunks)
    assert store.count() == len(chunks)

    # 5. Verify indexed documents list
    docs = store.get_indexed_documents()
    assert "admission_guidelines.pdf" in docs
    assert "academic_regulations.pdf" in docs
    assert "hostel_rules.pdf" in docs

    # 6. Verify duplicate avoidance
    re_added = store.add_chunks(chunks, synthetic_embeddings, overwrite=False)
    assert re_added == 0
    assert store.count() == len(chunks)

    # 7. Semantic Query Retrieval
    admission_query_vec = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    results = store.query(query_embedding=admission_query_vec, n_results=2)
    assert len(results) == 2
    assert "admission" in results[0]["source_filename"]
    assert results[0]["page_number"] in [1, 2]
    assert results[0]["similarity"] > 0.95

    # 8. Re-open store to verify disk persistence
    del store
    store_reloaded = ChromaVectorStore(persist_dir=chroma_dir, collection_name="college_e2e")
    assert store_reloaded.count() == len(chunks)
    assert store_reloaded.get_indexed_documents() == docs

    # 9. Clean up single document
    del_count = store_reloaded.delete_document("hostel_rules.pdf")
    assert del_count >= 1
    assert "hostel_rules.pdf" not in store_reloaded.get_indexed_documents()

