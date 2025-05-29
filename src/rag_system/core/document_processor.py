"""
Document processing utilities for loading and splitting documents.
"""
from pathlib import Path
from typing import List, Optional
from langchain_community.document_loaders import PyMuPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from config.settings import settings
from src.rag_system.utils.logger import get_logger

logger = get_logger()


class DocumentProcessor:
    """Handles document loading and text splitting operations."""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            length_function=len,
            separators=[
                "\n\n",    # Split by double newlines (paragraphs)
                "\n",      # Split by single newlines
                ". ",      # Split by period followed by space
                "? ",      # Split by question mark followed by space
                "! ",      # Split by exclamation mark followed by space
                "。 ",     # Chinese period followed by space
                "？ ",     # Chinese question mark followed by space
                "！ ",     # Chinese exclamation mark followed by space
                "。\n",    # Chinese period followed by newline
                "？\n",    # Chinese question mark followed by newline
                "！\n",    # Chinese exclamation mark followed by newline
                " ",       # Split by space as a fallback
                ""         # Finally, split by character
            ]
        )
    
    def load_document(self, file_path: Path) -> Optional[List[Document]]:
        """
        Load a single document based on its file extension.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            List of Document objects or None if loading fails
        """
        try:
            file_extension = file_path.suffix.lower()
            
            if file_extension == '.pdf':
                loader = PyMuPDFLoader(str(file_path))
            elif file_extension in ['.docx', '.doc']:
                loader = Docx2txtLoader(str(file_path))
            elif file_extension == '.txt':
                loader = TextLoader(str(file_path), encoding='utf-8')
            else:
                logger.warning(f"Unsupported file type: {file_extension} for {file_path}")
                return None
            
            documents = loader.load()
            logger.info(f"Successfully loaded {len(documents)} documents from {file_path}")
            return documents
            
        except Exception as e:
            logger.error(f"Failed to load document {file_path}: {str(e)}")
            return None
    
    def load_documents_from_directory(self, directory_path: Path) -> List[Document]:
        """
        Load all supported documents from a directory.
        
        Args:
            directory_path: Path to the directory containing documents
            
        Returns:
            List of all loaded Document objects
        """
        all_documents = []
        
        if not directory_path.exists():
            logger.warning(f"Document directory does not exist: {directory_path}")
            return all_documents
        
        supported_files = []
        for file_path in directory_path.iterdir():
            if (file_path.is_file() and 
                file_path.suffix.lower() in settings.SUPPORTED_FILE_EXTENSIONS):
                supported_files.append(file_path)
        
        if not supported_files:
            logger.info(f"No supported documents found in {directory_path}")
            return all_documents
        
        logger.info(f"Found {len(supported_files)} supported documents in {directory_path}")
        
        for file_path in supported_files:
            documents = self.load_document(file_path)
            if documents:
                all_documents.extend(documents)
        
        logger.info(f"Successfully loaded {len(all_documents)} total documents")
        return all_documents
    
    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to split
            
        Returns:
            List of split Document objects
        """
        try:
            if not documents:
                logger.warning("No documents provided for splitting")
                return []
            
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(chunks)} chunks")
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to split documents: {str(e)}")
            return []
    
    def get_document_list(self, directory_path: Path) -> List[str]:
        """
        Get a list of supported document filenames in a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of supported document filenames
        """
        if not directory_path.exists():
            return []
        
        return [
            file_path.name 
            for file_path in directory_path.iterdir()
            if (file_path.is_file() and 
                file_path.suffix.lower() in settings.SUPPORTED_FILE_EXTENSIONS)
        ]
