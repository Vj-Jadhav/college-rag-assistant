"""Retriever Module.

Performs semantic similarity retrieval against ChromaDB vector store
using local Ollama embeddings to fetch relevant college document context.
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class RetrieverError(Exception):
    """Base exception for retrieval errors."""
    pass


class DocumentRetriever:
    """Retrieves top-k relevant document passages based on query embeddings."""

    def __init__(
        self,
        vector_store: Any,
        embedding_service: Any,
        top_k: int = 3,
        min_similarity: float = 0.0,
    ):
        """Initialize DocumentRetriever.

        Args:
            vector_store: ChromaVectorStore instance.
            embedding_service: OllamaEmbeddingService instance.
            top_k: Number of chunks to retrieve (default: 3).
            min_similarity: Minimum cosine similarity threshold (default: 0.0).
        """
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, query: str) -> List[Dict[str, Any]]:
        """Retrieve most relevant context chunks for a query string.

        Args:
            query: User's natural language question.

        Returns:
            List of top-k matching chunk dictionaries with metadata:
            [
                {
                    "chunk_id": str,
                    "text": str,
                    "source_filename": str,
                    "page_number": int,
                    "similarity": float,
                    "distance": float,
                },
                ...
            ]
        """
        if not query or not query.strip():
            logger.warning("Empty or whitespace query passed to retriever.")
            return []

        clean_query = query.strip()

        # Check vector store population
        try:
            total_vectors = self.vector_store.count()
            if total_vectors == 0:
                logger.warning("Vector store is empty. No documents to retrieve.")
                return []
        except Exception as exc:
            raise RetrieverError(f"Failed to inspect vector store: {exc}") from exc

        # Generate query vector
        try:
            query_embedding = self.embedding_service.get_embedding(clean_query)
        except Exception as exc:
            raise RetrieverError(f"Failed to generate embedding for query '{clean_query}': {exc}") from exc

        # Query vector store for top-k nearest chunks
        try:
            raw_results = self.vector_store.query(
                query_embedding=query_embedding,
                n_results=self.top_k,
            )
        except Exception as exc:
            raise RetrieverError(f"Vector store search failed for query '{clean_query}': {exc}") from exc

        # Filter by minimum similarity if configured
        filtered_results: List[Dict[str, Any]] = []
        for r in raw_results:
            sim = r.get("similarity", 0.0)
            if sim >= self.min_similarity:
                filtered_results.append(r)

        logger.info(
            f"Retrieved {len(filtered_results)} chunk(s) for query: '{clean_query[:50]}...'"
        )
        return filtered_results

    def format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a structured text block for the LLM prompt.

        Args:
            chunks: List of retrieved chunk dictionaries.

        Returns:
            str: Structured context string including document name and page number.
        """
        if not chunks:
            return ""

        context_blocks = []
        for idx, chunk in enumerate(chunks, start=1):
            source = chunk.get("source_filename", "Unknown Document")
            page = chunk.get("page_number", "Unknown Page")
            text = chunk.get("text", "").strip()

            block = (
                f"--- [Document: {source} | Page: {page}] ---\n"
                f"{text}"
            )
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def extract_sources(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract unique document and page citation sources from retrieved chunks.

        Args:
            chunks: List of retrieved chunk dictionaries.

        Returns:
            List of unique source dictionaries:
            [{"source_filename": str, "page_number": int, "chunk_id": str}, ...]
        """
        seen = set()
        sources = []

        for chunk in chunks:
            doc_name = chunk.get("source_filename", "")
            page_num = chunk.get("page_number", 1)
            chunk_id = chunk.get("chunk_id", "")

            key = (doc_name, page_num)
            if key not in seen and doc_name:
                seen.add(key)
                sources.append({
                    "source_filename": doc_name,
                    "page_number": page_num,
                    "chunk_id": chunk_id,
                })

        return sources
