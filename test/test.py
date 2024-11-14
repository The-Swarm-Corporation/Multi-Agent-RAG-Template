import unittest
from unittest.mock import MagicMock

class TestRAGPipeline(unittest.TestCase):

    def setUp(self):
        # Mock the retriever and generator
        self.mock_retriever = MagicMock()
        self.mock_generator = MagicMock()

    def test_retrieve_documents(self):
        # Test if the retriever returns the expected documents
        self.mock_retriever.retrieve.return_value = ["doc1", "doc2"]
        query = "What is RAG?"
        retrieved_docs = self.mock_retriever.retrieve(query)
        
        self.assertEqual(retrieved_docs, ["doc1", "doc2"])
        self.mock_retriever.retrieve.assert_called_with(query)

    def test_generate_response(self):
        # Test if the generator produces the correct response based on documents
        self.mock_generator.generate.return_value = "RAG is a method combining retrieval and generation."
        docs = ["doc1", "doc2"]
        response = self.mock_generator.generate(docs)
        
        self.assertEqual(response, "RAG is a method combining retrieval and generation.")
        self.mock_generator.generate.assert_called_with(docs)

    def test_rag_pipeline(self):
        # Test the full pipeline: retrieval and generation together
        query = "What is RAG?"
        self.mock_retriever.retrieve.return_value = ["doc1", "doc2"]
        self.mock_generator.generate.return_value = "RAG is a method combining retrieval and generation."
        
        # Simulate the RAG pipeline
        docs = self.mock_retriever.retrieve(query)
        response = self.mock_generator.generate(docs)
        
        self.assertEqual(response, "RAG is a method combining retrieval and generation.")

if __name__ == "__main__":
    unittest.main()
