"""Vector Store Module.

Manages persistent indexing, storage, deduplication, and semantic retrieval
of document chunk embeddings using local ChromaDB.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import logging
import chromadb
from chromadb.api.models.Collection import Collection

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass


class DocumentNotFoundError(VectorStoreError):
    """Raised when an operation targets a document not present in vector store."""
    pass


class EmptyCollectionError(VectorStoreError):
    """Raised when query is executed on an empty collection."""
    pass


class ChromaVectorStore:
    """Manages document embeddings and collection storage in local ChromaDB."""

    def __init__(
        self,
        persist_dir: Union[Path, str] = "chroma_db",
        collection_name: str = "college_documents",
        embedding_service: Optional[Any] = None,
    ):
        """Initialize ChromaDB vector store with persistent on-disk storage.

        Args:
            persist_dir: Directory path where ChromaDB sqlite and index files persist.
            collection_name: Name of the vector collection (default: 'college_documents').
            embedding_service: Optional OllamaEmbeddingService instance for automatic embedding.
        """
        self.persist_dir = Path(persist_dir).resolve()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self.embedding_service = embedding_service

        try:
            # Initialize persistent client pointing to local directory
            self._client = chromadb.PersistentClient(path=str(self.persist_dir))
            # Use cosine distance for normalized semantic search
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Initialized ChromaVectorStore (Collection: '{self.collection_name}', "
                f"Storage: '{self.persist_dir}', Existing Vectors: {self._collection.count()})"
            )
        except Exception as exc:
            raise VectorStoreError(f"Failed to initialize ChromaDB at {self.persist_dir}: {exc}") from exc

    @property
    def collection(self) -> Collection:
        """Access underlying Chroma collection."""
        return self._collection

    def count(self) -> int:
        """Return total number of indexed vectors in the collection."""
        try:
            return self._collection.count()
        except Exception as exc:
            raise VectorStoreError(f"Failed to count collection items: {exc}") from exc

    def get_indexed_documents(self) -> List[str]:
        """Return a sorted list of unique document filenames present in the vector store."""
        try:
            total = self.count()
            if total == 0:
                return []

            # Retrieve only metadata keys
            results = self._collection.get(include=["metadatas"])
            metadatas = results.get("metadatas") or []

            filenames = set()
            for meta in metadatas:
                if meta and "source_filename" in meta:
                    filenames.add(str(meta["source_filename"]))

            return sorted(list(filenames))
        except Exception as exc:
            raise VectorStoreError(f"Failed to retrieve indexed document names: {exc}") from exc

    def has_document(self, filename: str) -> bool:
        """Check whether a specific document has already been ingested into the store.

        Args:
            filename: Target PDF filename (e.g. 'admission_guidelines.pdf').

        Returns:
            bool: True if at least one chunk for this filename is indexed.
        """
        try:
            results = self._collection.get(
                where={"source_filename": filename},
                limit=1,
            )
            return len(results.get("ids", [])) > 0
        except Exception as exc:
            logger.warning(f"Error checking document existence for '{filename}': {exc}")
            return False

    def delete_document(self, filename: str) -> int:
        """Delete all indexed chunks associated with a specific document filename.

        Args:
            filename: Target filename to delete.

        Returns:
            int: Number of chunks removed.
        """
        try:
            # Query existing chunk IDs for filename
            existing = self._collection.get(
                where={"source_filename": filename},
                include=["metadatas"],
            )
            ids_to_delete = existing.get("ids", [])
            count = len(ids_to_delete)

            if count > 0:
                self._collection.delete(where={"source_filename": filename})
                logger.info(f"Deleted {count} chunk(s) for document '{filename}'")

            return count
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete document '{filename}': {exc}") from exc

    def delete_chunk(self, chunk_id: str) -> bool:
        """Delete a single chunk by its unique chunk_id.

        Args:
            chunk_id: Unique chunk identifier.

        Returns:
            bool: True if deletion succeeded.
        """
        try:
            self._collection.delete(ids=[chunk_id])
            logger.info(f"Deleted chunk '{chunk_id}'")
            return True
        except Exception as exc:
            raise VectorStoreError(f"Failed to delete chunk '{chunk_id}': {exc}") from exc

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: Optional[List[List[float]]] = None,
        overwrite: bool = True,
    ) -> int:
        """Add document chunks and their vector embeddings to ChromaDB.

        Handles duplicate protection:
        - If overwrite=True (default), any existing chunks for the same source_filename
          are purged before new chunks are stored (ensuring safe document update).
        - If overwrite=False, chunks belonging to already-indexed documents are skipped.

        Args:
            chunks: List of chunk dictionaries containing 'chunk_id', 'text',
                    'source_filename', and 'page_number'.
            embeddings: Optional precomputed embeddings list. If omitted,
                        self.embedding_service is used to generate them.
            overwrite: Whether to overwrite existing document records.

        Returns:
            int: Number of chunks newly indexed.
        """
        if not chunks:
            logger.warning("add_chunks called with empty chunks list.")
            return 0

        # Validate embeddings
        if embeddings is None:
            if self.embedding_service is None:
                raise VectorStoreError(
                    "No embeddings provided and no embedding_service configured on ChromaVectorStore."
                )
            logger.info(f"Generating embeddings for {len(chunks)} chunk(s) via OllamaEmbeddingService...")
            embeddings = self.embedding_service.embed_chunks(chunks)

        if len(chunks) != len(embeddings):
            raise VectorStoreError(
                f"Mismatch between number of chunks ({len(chunks)}) and embeddings ({len(embeddings)})."
            )

        # Handle duplicate prevention by source_filename
        filenames_in_batch = {c.get("source_filename") for c in chunks if c.get("source_filename")}

        filtered_chunks: List[Dict[str, Any]] = []
        filtered_embeddings: List[List[float]] = []

        for fname in filenames_in_batch:
            if not fname:
                continue

            doc_exists = self.has_document(fname)

            if doc_exists and overwrite:
                logger.info(f"Document '{fname}' already exists in vector store. Purging old chunks for clean update.")
                self.delete_document(fname)
            elif doc_exists and not overwrite:
                logger.info(f"Document '{fname}' already exists and overwrite=False. Skipping ingestion for this document.")
                continue

            # Collect chunks and embeddings for this document
            for chunk, emb in zip(chunks, embeddings):
                if chunk.get("source_filename") == fname:
                    filtered_chunks.append(chunk)
                    filtered_embeddings.append(emb)

        if not filtered_chunks:
            logger.info("No new chunks to insert after duplicate filtering.")
            return 0

        # Prepare ChromaDB parameters
        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for c in filtered_chunks:
            chunk_id = str(c.get("chunk_id"))
            text = str(c.get("text", "")).strip()
            source_filename = str(c.get("source_filename", "unknown.pdf"))
            page_number = int(c.get("page_number", 1))

            ids.append(chunk_id)
            documents.append(text)
            metadatas.append({
                "source_filename": source_filename,
                "page_number": page_number,
                "chunk_id": chunk_id,
                "chunk_index": int(c.get("chunk_index", 0)),
            })

        try:
            self._collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=filtered_embeddings,
                metadatas=metadatas,
            )
            logger.info(f"Successfully upserted {len(ids)} chunk(s) into collection '{self.collection_name}'.")
            return len(ids)
        except Exception as exc:
            raise VectorStoreError(f"Failed to upsert chunks into ChromaDB: {exc}") from exc

    def query(
        self,
        query_text: Optional[str] = None,
        query_embedding: Optional[List[float]] = None,
        n_results: int = 4,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Perform semantic search against indexed chunks.

        Args:
            query_text: Plain text search query (requires embedding_service).
            query_embedding: Precomputed embedding vector of the query.
            n_results: Number of top results to retrieve (default: 4).
            where: Optional metadata filter dictionary (e.g. {'source_filename': '...'}).

        Returns:
            List of matching dictionaries containing:
            [
                {
                    "chunk_id": str,
                    "text": str,
                    "source_filename": str,
                    "page_number": int,
                    "distance": float,
                    "similarity": float,  # 1.0 - distance
                    "metadata": dict,
                },
                ...
            ]
        """
        if self.count() == 0:
            logger.warning("Query executed against empty vector store.")
            return []

        if query_embedding is None:
            if not query_text or not query_text.strip():
                raise ValueError("Must provide either query_text or query_embedding.")
            if self.embedding_service is None:
                raise VectorStoreError("query_text requires an embedding_service configured on ChromaVectorStore.")
            query_embedding = self.embedding_service.get_embedding(query_text)

        actual_n = min(n_results, self.count())

        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_n,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"ChromaDB query failed: {exc}") from exc

        parsed_results: List[Dict[str, Any]] = []

        ids_list = results.get("ids", [[]])[0]
        docs_list = results.get("documents", [[]])[0]
        metas_list = results.get("metadatas", [[]])[0]
        dists_list = results.get("distances", [[]])[0]

        for chunk_id, text, meta, dist in zip(ids_list, docs_list, metas_list, dists_list):
            metadata = meta or {}
            # Cosine distance to similarity: similarity = 1 - cosine_distance
            sim = max(0.0, 1.0 - float(dist)) if dist is not None else 1.0
            parsed_results.append({
                "chunk_id": chunk_id,
                "text": text,
                "source_filename": metadata.get("source_filename", ""),
                "page_number": metadata.get("page_number", 1),
                "distance": float(dist) if dist is not None else 0.0,
                "similarity": sim,
                "metadata": metadata,
            })

        return parsed_results

    def reset(self) -> None:
        """Clear all indexed data and reinitialize the collection."""
        try:
            self._client.delete_collection(self.collection_name)
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Reset collection '{self.collection_name}' successfully.")
        except Exception as exc:
            raise VectorStoreError(f"Failed to reset collection '{self.collection_name}': {exc}") from exc

    def get_collection_stats(self) -> Dict[str, Any]:
        """Return diagnostic statistics about the current vector database state."""
        docs = self.get_indexed_documents()
        return {
            "collection_name": self.collection_name,
            "persist_dir": str(self.persist_dir),
            "total_chunks": self.count(),
            "total_documents": len(docs),
            "documents": docs,
        }
