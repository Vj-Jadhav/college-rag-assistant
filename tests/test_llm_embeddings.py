"""Unit and Integration Tests for LLM and Embedding Services.

Verifies:
- Service initialization and default configurations
- Ollama availability and offline connection error handling
- Model not found error handling
- Mocked generation and streaming workflows
- Mocked single and batch embedding workflows
- Strict compliance: 100% local, no external API keys or cloud providers
"""

import pytest
from unittest.mock import MagicMock, patch
from llm_service import (
    OllamaLLMService,
    OllamaServiceError,
    OllamaConnectionError,
    OllamaModelNotFoundError,
)
from embeddings import (
    OllamaEmbeddingService,
    OllamaEmbeddingError,
    OllamaEmbeddingConnectionError,
    OllamaEmbeddingModelNotFoundError,
)


class TestOllamaLLMService:
    """Test suite for OllamaLLMService."""

    def test_initialization_defaults(self):
        service = OllamaLLMService()
        assert service.model_name == "llama3.2:3b"
        assert service.base_url == "http://localhost:11434"

    def test_custom_initialization(self):
        service = OllamaLLMService(model_name="custom-model:latest", base_url="http://127.0.0.1:11434/")
        assert service.model_name == "custom-model:latest"
        assert service.base_url == "http://127.0.0.1:11434"

    def test_service_unreachable_handling(self):
        # Point to a guaranteed non-existent local port
        service = OllamaLLMService(base_url="http://127.0.0.1:59999")
        assert service.is_service_available() is False
        assert service.is_model_available() is False

        health = service.check_health()
        assert health["service_running"] is False
        assert health["model_available"] is False
        assert "Ollama is not running" in health["error"]

    def test_generate_raises_connection_error_when_offline(self):
        service = OllamaLLMService(base_url="http://127.0.0.1:59999")
        with pytest.raises(OllamaConnectionError) as exc_info:
            service.generate("Hello world")
        assert "Could not connect to Ollama" in str(exc_info.value)

    def test_stream_generate_raises_connection_error_when_offline(self):
        service = OllamaLLMService(base_url="http://127.0.0.1:59999")
        with pytest.raises(OllamaConnectionError):
            list(service.stream_generate("Hello world"))

    def test_model_not_found_error_raised(self):
        service = OllamaLLMService(model_name="nonexistent:model")
        with patch.object(service, "is_service_available", return_value=True):
            with patch.object(service, "list_installed_models", return_value=["llama3:8b"]):
                with pytest.raises(OllamaModelNotFoundError) as exc_info:
                    service.generate("Hello")
                assert "nonexistent:model" in str(exc_info.value)

    def test_mocked_generate_success(self):
        service = OllamaLLMService(model_name="llama3.2:3b")
        mock_response = MagicMock()
        mock_response.message.content = "This is a local response from Llama 3.2."

        with patch.object(service, "is_service_available", return_value=True):
            with patch.object(service, "is_model_available", return_value=True):
                with patch.object(service._client, "chat", return_value=mock_response):
                    result = service.generate("What is RAG?")
                    assert result == "This is a local response from Llama 3.2."

    def test_mocked_stream_generate_success(self):
        service = OllamaLLMService(model_name="llama3.2:3b")

        chunk1 = MagicMock()
        chunk1.message.content = "Retrieval "
        chunk2 = MagicMock()
        chunk2.message.content = "Augmented "
        chunk3 = MagicMock()
        chunk3.message.content = "Generation."

        with patch.object(service, "is_service_available", return_value=True):
            with patch.object(service, "is_model_available", return_value=True):
                with patch.object(service._client, "chat", return_value=[chunk1, chunk2, chunk3]):
                    stream = service.stream_generate("What is RAG?")
                    chunks = list(stream)
                    assert chunks == ["Retrieval ", "Augmented ", "Generation."]
                    assert "".join(chunks) == "Retrieval Augmented Generation."


class TestOllamaEmbeddingService:
    """Test suite for OllamaEmbeddingService."""

    def test_initialization_defaults(self):
        service = OllamaEmbeddingService()
        assert service.model_name == "nomic-embed-text"
        assert service.base_url == "http://localhost:11434"

    def test_service_unreachable_handling(self):
        service = OllamaEmbeddingService(base_url="http://127.0.0.1:59999")
        assert service.is_service_available() is False
        assert service.is_model_available() is False

        health = service.check_health()
        assert health["service_running"] is False
        assert health["model_available"] is False
        assert "Ollama is not running" in health["error"]

    def test_empty_text_validation(self):
        service = OllamaEmbeddingService()
        with patch.object(service, "_validate_ready"):
            with pytest.raises(ValueError):
                service.get_embedding("")
            with pytest.raises(ValueError):
                service.get_embedding("   ")

    def test_mocked_get_embedding_single(self):
        service = OllamaEmbeddingService(model_name="nomic-embed-text")
        sample_vec = [0.123, -0.456, 0.789, 0.012]

        with patch.object(service, "is_service_available", return_value=True):
            with patch.object(service, "is_model_available", return_value=True):
                with patch.object(service._client, "embed", return_value=MagicMock(embeddings=[sample_vec])):
                    vector = service.get_embedding("College Document text")
                    assert vector == sample_vec
                    assert len(vector) == 4

    def test_mocked_get_embeddings_batch(self):
        service = OllamaEmbeddingService(model_name="nomic-embed-text")
        vec1 = [0.1, 0.2, 0.3]
        vec2 = [0.4, 0.5, 0.6]

        with patch.object(service, "is_service_available", return_value=True):
            with patch.object(service, "is_model_available", return_value=True):
                with patch.object(service._client, "embed", return_value=MagicMock(embeddings=[vec1, vec2])):
                    vectors = service.get_embeddings(["First chunk", "Second chunk"])
                    assert len(vectors) == 2
                    assert vectors[0] == vec1
                    assert vectors[1] == vec2


class TestLocalOnlyConstraint:
    """Verifies no external or paid API dependencies are referenced."""

    def test_no_paid_api_keys(self):
        import os
        forbidden_keys = ["OPENAI_API_KEY", "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "COHERE_API_KEY"]
        for key in forbidden_keys:
            assert key not in os.environ or os.environ[key] == ""

    def test_services_use_local_default_urls(self):
        llm = OllamaLLMService()
        emb = OllamaEmbeddingService()
        assert "localhost" in llm.base_url or "127.0.0.1" in llm.base_url
        assert "localhost" in emb.base_url or "127.0.0.1" in emb.base_url

