"""
Vector store management for document embeddings and retrieval.
"""
import os
from pathlib import Path
from typing import List, Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document

from config.settings import settings
from src.rag_system.utils.logger import get_logger

logger = get_logger()


class VectorStoreManager:
    """Manages vector database operations including creation, loading, and updates."""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self._initialize_embeddings()
    
    def _initialize_embeddings(self) -> None:
        """Initialize the embedding model."""
        try:
            logger.info(f"Initializing embedding model: {settings.EMBEDDING_MODEL_PATH}")
            logger.info(f"Using device: {settings.EMBEDDING_DEVICE}")
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=settings.EMBEDDING_MODEL_PATH,
                model_kwargs={'device': settings.EMBEDDING_DEVICE.value}
            )
            logger.info("Embedding model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
    def load_or_create_vector_store(self, documents: Optional[List[Document]] = None) -> bool:
        """
        Load existing vector store or create a new one with provided documents.
        
        Args:
            documents: Optional list of documents to create vector store with
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Try to load existing vector store
            if self._load_existing_vector_store():
                logger.info("Loaded existing vector store")
                return True
            
            # Create new vector store if documents are provided
            if documents:
                return self._create_new_vector_store(documents)
            else:
                logger.warning("No existing vector store found and no documents provided")
                return False
                
        except Exception as e:
            logger.error(f"Failed to load or create vector store: {str(e)}")
            return False
    
    def _load_existing_vector_store(self) -> bool:
        """
        Load an existing vector store from disk.
        
        Returns:
            True if successfully loaded, False otherwise
        """
        try:
            persist_dir = settings.PERSIST_DIR
            
            if not persist_dir.exists() or not any(persist_dir.iterdir()):
                logger.info("No existing vector store found")
                return False
            
            logger.info(f"Loading existing vector store from {persist_dir}")
            self.vector_store = Chroma(
                persist_directory=str(persist_dir),
                embedding_function=self.embeddings
            )
            
            # Test the vector store by getting collection info
            collection = self.vector_store._collection
            count = collection.count()
            logger.info(f"Vector store loaded with {count} documents")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load existing vector store: {str(e)}")
            logger.info("This might be due to embedding model changes or corrupted data")
            return False
    
    def _create_new_vector_store(self, documents: List[Document]) -> bool:
        """
        Create a new vector store with provided documents.
        
        Args:
            documents: Documents to add to the vector store
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not documents:
                logger.warning("No documents provided for vector store creation")
                return False
            
            logger.info(f"Creating new vector store with {len(documents)} documents")
            
            # Ensure persist directory exists and is clean
            persist_dir = settings.PERSIST_DIR
            if persist_dir.exists():
                import shutil
                shutil.rmtree(persist_dir)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(persist_dir)
            )
            
            logger.info("Vector store created and persisted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create new vector store: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Document]) -> bool:
        """
        Add new documents to the existing vector store.
        
        Args:
            documents: Documents to add
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.vector_store:
                logger.error("No vector store available to add documents to")
                return False
            
            if not documents:
                logger.warning("No documents provided to add")
                return True
            
            logger.info(f"Adding {len(documents)} documents to vector store")
            self.vector_store.add_documents(documents)
            logger.info("Documents added successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add documents to vector store: {str(e)}")
            return False
    
    def get_retriever(self, k: Optional[int] = None):
        """
        Get a retriever from the vector store.
        
        Args:
            k: Number of documents to retrieve (uses settings default if None)
            
        Returns:
            Retriever object or None if vector store is not available
        """
        if not self.vector_store:
            logger.error("No vector store available for retrieval")
            return None
        
        search_k = k or settings.SEARCH_K
        return self.vector_store.as_retriever(search_kwargs={"k": search_k})
    
    def similarity_search(self, query: str, k: Optional[int] = None) -> List[Document]:
        """
        Perform similarity search on the vector store.
        
        Args:
            query: Search query
            k: Number of documents to retrieve
            
        Returns:
            List of similar documents
        """
        if not self.vector_store:
            logger.error("No vector store available for search")
            return []
        
        try:
            search_k = k or settings.SEARCH_K
            results = self.vector_store.similarity_search(query, k=search_k)
            logger.debug(f"Similarity search returned {len(results)} documents for query: {query}")
            return results
            
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            return []
    
    def is_ready(self) -> bool:
        """
        Check if the vector store is ready for use.
        
        Returns:
            True if vector store is available, False otherwise
        """
        return self.vector_store is not None
