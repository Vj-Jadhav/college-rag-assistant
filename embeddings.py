"""Embeddings Module.

Integrates with local Ollama runtime to produce vector embeddings
using the 'nomic-embed-text' model without external APIs.
Includes ChromaDB compatible embedding function adapter.
"""

from typing import List, Dict, Any, Optional
import logging
import time
import ollama

logger = logging.getLogger(__name__)


class OllamaEmbeddingError(Exception):
    """Base exception for Ollama embedding errors."""
    pass


class OllamaEmbeddingConnectionError(OllamaEmbeddingError):
    """Raised when Ollama server cannot be reached."""
    pass


class OllamaEmbeddingModelNotFoundError(OllamaEmbeddingError):
    """Raised when the specified embedding model is missing."""
    pass


class OllamaEmbeddingService:
    """Wrapper for generating text embeddings locally using Ollama."""

    def __init__(
        self,
        model_name: str = "nomic-embed-text",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize the Ollama Embedding Service.

        Args:
            model_name: Embedding model name (default: 'nomic-embed-text').
            base_url: Ollama server endpoint (default: 'http://localhost:11434').
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._client = ollama.Client(host=self.base_url)
        self._last_warn_time = 0.0

    def is_service_available(self) -> bool:
        """Check if the Ollama server is running and reachable."""
        try:
            self._client.list()
            return True
        except Exception as exc:
            now = time.time()
            if now - self._last_warn_time > 30.0:
                self._last_warn_time = now
                logger.warning(f"Ollama unreachable at {self.base_url}: {exc}")
            return False

    def list_installed_models(self) -> List[str]:
        """List models installed in Ollama."""
        try:
            response = self._client.list()
            models = getattr(response, "models", None)
            if models is None and isinstance(response, dict):
                models = response.get("models", [])
            elif models is None and isinstance(response, list):
                models = response

            names = []
            for m in (models or []):
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if not name and isinstance(m, dict):
                    name = m.get("model") or m.get("name")
                if name:
                    names.append(str(name))
            return names
        except Exception as exc:
            raise OllamaEmbeddingConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Ensure it is running (`ollama serve`)."
            ) from exc

    def is_model_available(self) -> bool:
        """Check if embedding model is pulled locally."""
        try:
            models = self.list_installed_models()
            target = self.model_name.lower()
            return any(
                target == m.lower() or target in m.lower() or m.lower().startswith(target)
                for m in models
            )
        except OllamaEmbeddingConnectionError:
            return False

    def check_health(self) -> Dict[str, Any]:
        """Health check for embedding service."""
        available = self.is_service_available()
        if not available:
            return {
                "service_running": False,
                "model_available": False,
                "model_name": self.model_name,
                "base_url": self.base_url,
                "error": f"Ollama is not running at {self.base_url}. Start it with 'ollama serve'.",
            }
        ready = self.is_model_available()
        return {
            "service_running": True,
            "model_available": ready,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "error": None if ready else f"Embedding model '{self.model_name}' not found. Run 'ollama pull {self.model_name}'.",
        }

    def _validate_ready(self) -> None:
        """Validate connection and model readiness."""
        if not self.is_service_available():
            raise OllamaEmbeddingConnectionError(
                f"Ollama server is unreachable at {self.base_url}. "
                "Please verify Ollama is installed and running (`ollama serve`)."
            )
        if not self.is_model_available():
            raise OllamaEmbeddingModelNotFoundError(
                f"Embedding model '{self.model_name}' is not installed. "
                f"Run 'ollama pull {self.model_name}' to download it."
            )

    def get_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for a single text string.

        Args:
            text: Input string to embed.

        Returns:
            List[float]: Vector embedding representation.
        """
        self._validate_ready()

        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text.")

        try:
            # Modern ollama-python provides .embed(model=..., input=...)
            if hasattr(self._client, "embed"):
                response = self._client.embed(model=self.model_name, input=text)
                embeddings = getattr(response, "embeddings", None)
                if embeddings is None and isinstance(response, dict):
                    embeddings = response.get("embeddings")
                if embeddings and len(embeddings) > 0:
                    return embeddings[0]

            # Fallback for older client signatures
            response = self._client.embeddings(model=self.model_name, prompt=text)
            embedding = getattr(response, "embedding", None)
            if embedding is None and isinstance(response, dict):
                embedding = response.get("embedding")
            if embedding:
                return embedding

            raise OllamaEmbeddingError(f"Unexpected response structure from Ollama: {response}")
        except ollama.ResponseError as exc:
            if exc.status_code == 404 or "not found" in str(exc).lower():
                raise OllamaEmbeddingModelNotFoundError(
                    f"Model '{self.model_name}' not found: {exc.error}"
                ) from exc
            raise OllamaEmbeddingError(f"Embedding failed: {exc.error}") from exc
        except Exception as exc:
            if isinstance(exc, OllamaEmbeddingError):
                raise
            raise OllamaEmbeddingError(f"Failed to generate embedding: {exc}") from exc

    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Generate vector embeddings for a batch of text strings.

        Args:
            texts: List of input strings.

        Returns:
            List[List[float]]: List of vector embeddings.
        """
        self._validate_ready()

        if not texts:
            return []

        # Try batch embedding if supported
        try:
            if hasattr(self._client, "embed"):
                response = self._client.embed(model=self.model_name, input=texts)
                embeddings = getattr(response, "embeddings", None)
                if embeddings is None and isinstance(response, dict):
                    embeddings = response.get("embeddings")
                if embeddings and len(embeddings) == len(texts):
                    return embeddings
        except Exception as exc:
            logger.debug(f"Batch embedding returned: {exc}; falling back to sequential.")

        # Sequential generation fallback
        results = []
        for text in texts:
            results.append(self.get_embedding(text))
        return results

    def embed_chunks(self, chunks: List[Dict[str, Any]]) -> List[List[float]]:
        """Generate embeddings for a list of document chunk dictionaries.

        Args:
            chunks: List of chunk dictionaries, each containing a 'text' key.

        Returns:
            List of embedding vectors aligned with input chunks.
        """
        texts = [c.get("text", "") for c in chunks]
        return self.get_embeddings(texts)


class OllamaChromaEmbeddingFunction:
    """ChromaDB-compatible EmbeddingFunction adapter wrapping OllamaEmbeddingService."""

    def __init__(self, embedding_service: OllamaEmbeddingService):
        self.embedding_service = embedding_service

    def __call__(self, input: List[str]) -> List[List[float]]:
        """Generate embeddings for input documents required by ChromaDB."""
        return self.embedding_service.get_embeddings(input)
