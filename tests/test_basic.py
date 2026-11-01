"""Basic smoke and import test for College RAG Assistant modules."""

import importlib
import pytest

MODULES_TO_TEST = [
    "document_loader",
    "chunker",
    "embeddings",
    "vector_store",
    "retriever",
    "llm_service",
    "rag_service",
]

@pytest.mark.parametrize("module_name", MODULES_TO_TEST)
def test_module_import(module_name: str):
    """Ensure that all core project modules can be successfully imported."""
    module = importlib.import_module(module_name)
    assert module is not None

