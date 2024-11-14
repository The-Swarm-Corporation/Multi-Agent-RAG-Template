import asyncio
from typing import List, Dict, Optional, Union, Any
from pathlib import Path
import hashlib
from datetime import datetime
from functools import wraps

import pinecone
from loguru import logger
import tiktoken
from sentence_transformers import SentenceTransformer
import numpy as np


def sync_wrapper(async_func):
    """Decorator to convert async functions to sync for external interface."""

    @wraps(async_func)
    def sync_func(*args, **kwargs):
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        return loop.run_until_complete(async_func(*args, **kwargs))

    return sync_func


class PineconeManager:
    """
    A production-grade class for managing Pinecone vector database operations.

    This class handles:
    - Initialization of Pinecone connection
    - Text embedding generation
    - Adding single and multiple documents
    - Querying the vector database
    - Bulk loading from folders

    Attributes:
        index_name (str): Name of the Pinecone index
        dimension (int): Dimension of the embedding vectors
        metric (str): Distance metric used for similarity search
        embedder (SentenceTransformer): Model for generating text embeddings
        batch_size (int): Size of batches for bulk operations
        namespace (str): Namespace in Pinecone index
    """

    def __init__(
        self,
        api_key: str,
        index_name: str,
        environment: str,
        embedding_model: str = "all-MiniLM-L6-v2",
        dimension: int = 384,
        metric: str = "cosine",
        batch_size: int = 100,
        namespace: str = "",
    ) -> None:
        """
        Initialize the PineconeManager.

        Args:
            api_key: Pinecone API key
            index_name: Name of the Pinecone index
            environment: Pinecone environment
            embedding_model: Name of the sentence-transformer model to use
            dimension: Dimension of the embedding vectors
            metric: Distance metric for similarity search
            batch_size: Size of batches for bulk operations
            namespace: Namespace in Pinecone index

        Raises:
            ValueError: If invalid parameters are provided
            ConnectionError: If connection to Pinecone fails
        """
        logger.info(f"Initializing PineconeManager with index: {index_name}")

        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.batch_size = batch_size
        self.namespace = namespace

        # Initialize Pinecone
        try:
            pinecone.init(api_key=api_key, environment=environment)
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {str(e)}")
            raise ConnectionError(f"Could not connect to Pinecone: {str(e)}")

        # Create index if it doesn't exist
        if index_name not in pinecone.list_indexes():
            logger.info(f"Creating new index: {index_name}")
            pinecone.create_index(name=index_name, dimension=dimension, metric=metric)

        self.index = pinecone.Index(index_name)

        # Initialize embedding model
        try:
            self.embedder = SentenceTransformer(embedding_model)
            logger.info(f"Initialized embedding model: {embedding_model}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise ValueError(f"Could not load embedding model: {str(e)}")

        # Initialize tokenizer for length validation
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def _generate_embedding(self, text: str) -> np.ndarray:
        """
        Generate embedding for input text.

        Args:
            text: Input text to embed

        Returns:
            numpy.ndarray: Embedding vector

        Raises:
            ValueError: If text is empty or too long
        """
        if not text.strip():
            raise ValueError("Empty text provided")

        tokens = self.tokenizer.encode(text)
        if len(tokens) > 8191:  # OpenAI's token limit
            raise ValueError(f"Text too long: {len(tokens)} tokens (max 8191)")

        return self.embedder.encode(text)

    def _generate_id(self, text: str) -> str:
        """Generate a deterministic ID for a piece of text."""
        return hashlib.md5(text.encode()).hexdigest()

    @sync_wrapper
    async def add(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a single text item to the index.

        Args:
            text: Text to add to the index
            metadata: Optional metadata to store with the vector

        Returns:
            str: ID of the added item

        Raises:
            ValueError: If text is invalid
            ConnectionError: If Pinecone operation fails
        """
        try:
            vector_id = self._generate_id(text)
            embedding = self._generate_embedding(text)

            metadata = metadata or {}
            metadata.update(
                {
                    "text": text,
                    "timestamp": datetime.utcnow().isoformat(),
                    "char_count": len(text),
                }
            )

            await self.index.upsert(
                vectors=[(vector_id, embedding.tolist(), metadata)],
                namespace=self.namespace,
            )

            logger.info(f"Added text to index with ID: {vector_id}")
            return vector_id

        except Exception as e:
            logger.error(f"Failed to add text to index: {str(e)}")
            raise

    async def _async_query(
        self,
        query_text: str,
        top_k: int = 5,
        include_metadata: bool = True,
        score_threshold: Optional[float] = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Internal async query implementation.

        Args:
            query_text: Text to search for
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing matches and scores
        """
        try:
            query_embedding = self._generate_embedding(query_text)

            results = await self.index.query(
                vector=query_embedding.tolist(),
                top_k=top_k,
                include_metadata=include_metadata,
                namespace=self.namespace,
            )

            matches = []
            for match in results.matches:
                if score_threshold and match.score < score_threshold:
                    continue

                matches.append(
                    {"id": match.id, "score": match.score, "metadata": match.metadata}
                )

            logger.info(f"Query returned {len(matches)} results")
            return matches

        except Exception as e:
            logger.error(f"Query failed: {str(e)}")
            raise

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        include_metadata: bool = True,
        score_threshold: Optional[float] = 0.7,
    ) -> List[Dict[str, Any]]:
        """
        Synchronous method to query the index with input text.

        Args:
            query_text: Text to search for
            top_k: Number of results to return
            include_metadata: Whether to include metadata in results
            score_threshold: Minimum similarity score threshold

        Returns:
            List of dictionaries containing matches and scores

        Raises:
            ValueError: If query is invalid
            ConnectionError: If Pinecone operation fails
        """
        try:
            return sync_wrapper(self._async_query)(
                query_text, top_k, include_metadata, score_threshold
            )
        except Exception as e:
            logger.error(f"Synchronous query failed: {str(e)}")
            raise

    @sync_wrapper
    async def add_folder(
        self,
        folder_path: Union[str, Path],
        file_extensions: List[str] = [".txt", ".md"],
        recursive: bool = True,
    ) -> List[str]:
        """
        Add all compatible files from a folder to the index.

        Args:
            folder_path: Path to folder
            file_extensions: List of file extensions to process
            recursive: Whether to process subfolders

        Returns:
            List of added file IDs

        Raises:
            FileNotFoundError: If folder doesn't exist
            IOError: If file reading fails
        """
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")

        added_ids = []
        pattern = "**/*" if recursive else "*"

        try:
            for file_path in folder_path.glob(pattern):
                if file_path.suffix.lower() not in file_extensions:
                    continue

                logger.info(f"Processing file: {file_path}")

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()

                    metadata = {
                        "filename": file_path.name,
                        "filepath": str(file_path),
                        "file_size": file_path.stat().st_size,
                    }

                    vector_id = await self.add(text, metadata)
                    added_ids.append(vector_id)

                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {str(e)}")
                    continue

            logger.info(f"Added {len(added_ids)} files to index")
            return added_ids

        except Exception as e:
            logger.error(f"Folder processing failed: {str(e)}")
            raise

    @sync_wrapper
    async def delete(self, vector_ids: List[str]) -> None:
        """
        Delete vectors from the index.

        Args:
            vector_ids: List of vector IDs to delete

        Raises:
            ConnectionError: If Pinecone operation fails
        """
        try:
            await self.index.delete(ids=vector_ids, namespace=self.namespace)
            logger.info(f"Deleted {len(vector_ids)} vectors from index")
        except Exception as e:
            logger.error(f"Failed to delete vectors: {str(e)}")
            raise

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        try:
            pinecone.deinit()
            logger.info("Cleaned up Pinecone connection")
        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")
