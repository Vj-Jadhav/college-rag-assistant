"""LLM Service Module.

Handles generation requests to local Ollama runtime hosting
the 'llama3.2:3b' language model without external APIs.
"""

from typing import Dict, Any, Generator, Optional, List
import logging
import time
import ollama

logger = logging.getLogger(__name__)


class OllamaServiceError(Exception):
    """Base exception for Ollama service errors."""
    pass


class OllamaConnectionError(OllamaServiceError):
    """Raised when the Ollama service is unreachable."""
    pass


class OllamaModelNotFoundError(OllamaServiceError):
    """Raised when the requested model is not downloaded in Ollama."""
    pass


class OllamaLLMService:
    """Interface for querying the locally hosted Llama 3.2 model via Ollama."""

    def __init__(
        self,
        model_name: str = "llama3.2:3b",
        base_url: str = "http://localhost:11434",
    ):
        """Initialize the Ollama LLM Service.

        Args:
            model_name: Target model name (default: 'llama3.2:3b').
            base_url: Base URL for local Ollama server (default: 'http://localhost:11434').
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")
        self._client = ollama.Client(host=self.base_url)
        self._last_warn_time = 0.0

    def is_service_available(self) -> bool:
        """Check if the Ollama server is running and reachable.

        Returns:
            bool: True if reachable, False otherwise.
        """
        try:
            self._client.list()
            return True
        except Exception as exc:
            now = time.time()
            if now - self._last_warn_time > 30.0:
                self._last_warn_time = now
                logger.warning(f"Ollama service unreachable at {self.base_url}: {exc}")
            return False

    def list_installed_models(self) -> List[str]:
        """List all model names currently pulled and available locally.

        Returns:
            List[str]: List of model name strings.

        Raises:
            OllamaConnectionError: If unable to reach Ollama server.
        """
        try:
            response = self._client.list()
            # ollama-python returns ListResponse object with .models attribute
            models = getattr(response, "models", None)
            if models is None and isinstance(response, dict):
                models = response.get("models", [])
            elif models is None and isinstance(response, list):
                models = response

            model_names = []
            for m in (models or []):
                # m can be a Model object or dict
                name = getattr(m, "model", None) or getattr(m, "name", None)
                if not name and isinstance(m, dict):
                    name = m.get("model") or m.get("name")
                if name:
                    model_names.append(str(name))
            return model_names
        except Exception as exc:
            raise OllamaConnectionError(
                f"Cannot connect to Ollama service at {self.base_url}. "
                "Ensure Ollama is installed and running (`ollama serve`)."
            ) from exc

    def is_model_available(self) -> bool:
        """Check if self.model_name is downloaded in Ollama.

        Returns:
            bool: True if model is pulled locally, False otherwise.
        """
        try:
            installed = self.list_installed_models()
            target = self.model_name.lower()
            # Match either exact name or prefix (e.g. 'llama3.2:3b' or 'llama3.2:3b-instruct-q4_K_M')
            return any(
                target == m.lower() or target in m.lower() or m.lower().startswith(target)
                for m in installed
            )
        except OllamaConnectionError:
            return False

    def check_health(self) -> Dict[str, Any]:
        """Perform a comprehensive health check on the LLM service.

        Returns:
            Dict containing service availability, model presence, and installed models.
        """
        available = self.is_service_available()
        if not available:
            return {
                "service_running": False,
                "model_available": False,
                "model_name": self.model_name,
                "base_url": self.base_url,
                "installed_models": [],
                "error": f"Ollama is not running at {self.base_url}. Start it with 'ollama serve'.",
            }

        installed = self.list_installed_models()
        model_ready = self.is_model_available()
        return {
            "service_running": True,
            "model_available": model_ready,
            "model_name": self.model_name,
            "base_url": self.base_url,
            "installed_models": installed,
            "error": None if model_ready else f"Model '{self.model_name}' not found. Pull it via 'ollama pull {self.model_name}'.",
        }

    def _validate_ready(self) -> None:
        """Validate connection and model availability prior to inference."""
        if not self.is_service_available():
            raise OllamaConnectionError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Please verify Ollama is installed and started (`ollama serve`)."
            )
        if not self.is_model_available():
            raise OllamaModelNotFoundError(
                f"Model '{self.model_name}' is not installed in Ollama. "
                f"Run 'ollama pull {self.model_name}' in your terminal to download it."
            )

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> str:
        """Generate a complete text response from the local LLM.

        Args:
            prompt: User message / query.
            system_prompt: Optional instructions or context for the system.
            temperature: Sampling temperature (lower = more deterministic).

        Returns:
            str: Generated response text.

        Raises:
            OllamaConnectionError: If Ollama service is unreachable.
            OllamaModelNotFoundError: If the model is missing.
            OllamaServiceError: For any other generation failure.
        """
        self._validate_ready()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": temperature},
                stream=False,
            )
            # Response handling for ollama-python ChatResponse / dict
            message = getattr(response, "message", None)
            if message and hasattr(message, "content"):
                return message.content
            if isinstance(response, dict) and "message" in response:
                return response["message"].get("content", "")
            return str(response)
        except ollama.ResponseError as exc:
            if exc.status_code == 404 or "not found" in str(exc).lower():
                raise OllamaModelNotFoundError(
                    f"Model '{self.model_name}' not found: {exc.error}"
                ) from exc
            raise OllamaServiceError(f"Ollama generation error: {exc.error}") from exc
        except Exception as exc:
            raise OllamaServiceError(f"Failed to generate response: {exc}") from exc

    def stream_generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.2,
    ) -> Generator[str, None, None]:
        """Stream response tokens as they are produced by the local LLM.

        Args:
            prompt: User message / query.
            system_prompt: Optional instructions or context for the system.
            temperature: Sampling temperature.

        Yields:
            str: Text chunk increments.
        """
        self._validate_ready()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            stream = self._client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": temperature},
                stream=True,
            )
            for chunk in stream:
                message = getattr(chunk, "message", None)
                if message and hasattr(message, "content"):
                    content = message.content
                elif isinstance(chunk, dict) and "message" in chunk:
                    content = chunk["message"].get("content", "")
                else:
                    content = ""
                if content:
                    yield content
        except ollama.ResponseError as exc:
            if exc.status_code == 404 or "not found" in str(exc).lower():
                raise OllamaModelNotFoundError(
                    f"Model '{self.model_name}' not found: {exc.error}"
                ) from exc
            raise OllamaServiceError(f"Ollama streaming error: {exc.error}") from exc
        except Exception as exc:
            raise OllamaServiceError(f"Failed to stream response: {exc}") from exc
