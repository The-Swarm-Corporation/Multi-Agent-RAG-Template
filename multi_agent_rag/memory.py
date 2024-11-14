import os
import pinecone
from typing import Optional, List
from pathlib import Path
from loguru import logger
from llama_index.core import SimpleDirectoryReader
from llama_index.embeddings import get_embedding

class LlamaPineconeDB:
    """Manage document indexing and querying using Pinecone and LlamaIndex for memory."""

    def __init__(self, data_dir: str = "docs", pinecone_api_key: str = "", pinecone_env: str = "us-east1-aws") -> None:
        self.data_dir = data_dir
        self.index = None
        
        # Initialize Pinecone with the specified environment
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        index_name = "llama-memory-index"

        # Check if the index already exists; if not, create it
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=768)  # assuming 768 for embedding vector size
        self.pinecone_index = pinecone.Index(index_name)

        logger.info("Initialized LlamaPineconeDB")
        data_path = Path(self.data_dir)
        if not data_path.exists():
            logger.error(f"Directory not found: {self.data_dir}")
            raise FileNotFoundError(f"Directory {self.data_dir} does not exist")
        
        self.add_documents()

    def add_documents(self) -> None:
        """Read and index documents into Pinecone."""
        try:
            documents = SimpleDirectoryReader(self.data_dir).load_data()
            for doc in documents:
                embedding = get_embedding(doc.text)  # Get embedding for document text
                # Use doc id and embedding in upsert
                self.pinecone_index.upsert([(doc.id, embedding)])
            logger.success(f"Documents indexed successfully from {self.data_dir}")
        except Exception as e:
            logger.error(f"Error indexing documents: {str(e)}")
            raise

    def query(self, query: str, top_k: int = 5) -> List[str]:
        """Retrieve similar documents using Pinecone and provide memory.

        Args:
            query (str): The query string.
            top_k (int): Number of similar documents to retrieve.

        Returns:
            List[str]: List of top-k similar document texts.
        """
        try:
            query_embedding = get_embedding(query)
            results = self.pinecone_index.query(query_embedding, top_k=top_k, include_values=True)

            # Retrieve top documents
            top_docs = [match['text'] for match in results['matches']]
            logger.info(f"Retrieved {len(top_docs)} documents for query: {query}")
            return top_docs
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            raise

    def close(self):
        """Clean up Pinecone resources."""
        pinecone.deinit()

# Initialize the Pinecone DB using API key from environment variables
pinecone_api_key = os.getenv("PINECONE_API_KEY")  # Ensure this is set in your environment

if pinecone_api_key:
    llama_pinecone_db = LlamaPineconeDB(
        data_dir="docs",
        pinecone_api_key=pinecone_api_key,
        pinecone_env="us-east1-aws"  # Free tier environment
    )
    response = llama_pinecone_db.query("What is the medical history of patient 1?")
    print(response)
    llama_pinecone_db.close()
else:
    logger.error("Pinecone API key not found. Please set the PINECONE_API_KEY environment variable.")
