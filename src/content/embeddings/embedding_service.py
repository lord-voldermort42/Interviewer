"""
Embedding service with multiple backend options:
- OpenAI: Uses OpenAI's embedding API (default)
- vLLM: Uses local vLLM server for offline embeddings
- NoOp: Disabled embeddings (returns zero vectors)

Configure via EMBEDDING_BACKEND environment variable.
"""

from abc import ABC, abstractmethod
from typing import List
import numpy as np
import os
from openai import OpenAI, AzureOpenAI


class EmbeddingBackend(ABC):
    """Abstract base class for embedding backends."""

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """Return the dimension of embeddings produced by this backend."""
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for the given text."""
        pass

    @abstractmethod
    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts (optional optimization)."""
        pass


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI embedding backend using text-embedding-3-small.
    Uses AzureOpenAI if AZURE_OPENAI_ENDPOINT is set, otherwise standard OpenAI.
    """

    def __init__(self, model: str = "text-embedding-3-small"):
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        if azure_endpoint:
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                api_version=os.getenv("OPENAI_API_VERSION", "2024-02-01"),
                azure_endpoint=azure_endpoint,
            )
        else:
            self.client = OpenAI()
        self.model = model
        self._dimension = 1536  # text-embedding-3-small dimension

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts in a single API call."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]


class VLLMEmbeddingBackend(EmbeddingBackend):
    """vLLM embedding backend for offline/local embeddings.

    Requires a vLLM server running with an embedding model.
    Configure via environment variables:
    - VLLM_EMBEDDING_URL: vLLM server URL (default: http://localhost:8000/v1)
    - VLLM_EMBEDDING_MODEL: Model name (default: BAAI/bge-small-en-v1.5)
    - VLLM_EMBEDDING_DIM: Embedding dimension (default: 384)
    """

    def __init__(self):
        self.base_url = os.getenv("VLLM_EMBEDDING_URL", "http://localhost:8000/v1")
        self.model = os.getenv("VLLM_EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B")
        self._dimension = int(os.getenv("VLLM_EMBEDDING_DIM", "384"))

        # Create OpenAI client pointing to vLLM server
        # vLLM provides OpenAI-compatible API
        self.client = OpenAI(
            base_url=self.base_url,
            api_key="EMPTY"  # vLLM doesn't require API key
        )

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding using vLLM server."""
        response = self.client.embeddings.create(
            input=text,
            model=self.model
        )
        return np.array(response.data[0].embedding, dtype=np.float32)

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model
        )
        return [np.array(item.embedding, dtype=np.float32) for item in response.data]


class NoOpEmbeddingBackend(EmbeddingBackend):
    """No-op embedding backend that disables embeddings.

    Returns zero vectors. This makes search operations non-functional
    but allows the system to run without making any embedding API calls.
    """

    def __init__(self, dimension: int = 384):
        """
        Args:
            dimension: Dimension for zero vectors (default: 384)
                      Set via NOOP_EMBEDDING_DIM env var
        """
        self._dimension = int(os.getenv("NOOP_EMBEDDING_DIM", str(dimension)))

    def get_embedding_dimension(self) -> int:
        return self._dimension

    def get_embedding(self, text: str) -> np.ndarray:
        """Return zero vector."""
        return np.zeros(self._dimension, dtype=np.float32)

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Return zero vectors for all texts."""
        return [np.zeros(self._dimension, dtype=np.float32) for _ in texts]


class EmbeddingService:
    """Service for generating embeddings with configurable backends.

    Usage:
        service = EmbeddingService()  # Uses env var EMBEDDING_BACKEND
        embedding = service.get_embedding("some text")

    Environment Variables:
        EMBEDDING_BACKEND: Backend to use (openai|vllm|noop), default: openai
    """

    def __init__(self, backend: str = None):
        """
        Args:
            backend: Override backend selection. If None, uses EMBEDDING_BACKEND env var.
                    Options: 'openai', 'vllm', 'noop'
        """
        if backend is None:
            backend = os.getenv("EMBEDDING_BACKEND", "openai").lower()

        self.backend_name = backend
        self._backend = self._create_backend(backend)

    def _create_backend(self, backend: str) -> EmbeddingBackend:
        """Create the appropriate backend instance."""
        if backend == "openai":
            return OpenAIEmbeddingBackend()
        elif backend == "vllm":
            return VLLMEmbeddingBackend()
        elif backend == "noop":
            return NoOpEmbeddingBackend()
        else:
            raise ValueError(
                f"Unknown embedding backend: {backend}. "
                f"Options: 'openai', 'vllm', 'noop'"
            )

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings from the current backend."""
        return self._backend.get_embedding_dimension()

    def get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for a single text."""
        return self._backend.get_embedding(text)

    def get_embeddings_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Generate embeddings for multiple texts."""
        return self._backend.get_embeddings_batch(texts)

    def is_noop(self) -> bool:
        """Check if embeddings are disabled (noop backend)."""
        return self.backend_name == "noop"
