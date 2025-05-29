"""
Main RAG System orchestrating all components.
"""
import shutil
from pathlib import Path
from typing import List, Tuple, Iterator

from config.settings import settings
from src.rag_system.core.document_processor import DocumentProcessor
from src.rag_system.core.vector_store_manager import VectorStoreManager
from src.rag_system.core.llm_client import LLMClient
from src.rag_system.core.rag_chain_manager import RAGChainManager
from src.rag_system.utils.logger import get_logger

logger = get_logger()


class RAGSystem:
    """Main RAG system coordinating all components."""
    
    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.vector_store_manager = VectorStoreManager()
        self.llm_client = LLMClient()
        self.rag_chain_manager = None
        
        self._initialize_system()
    
    def _initialize_system(self) -> None:
        """Initialize the RAG system components."""
        try:
            logger.info("Initializing RAG system...")
            
            # Initialize RAG chain manager
            self.rag_chain_manager = RAGChainManager(
                self.vector_store_manager, 
                self.llm_client
            )
            
            # Load existing documents and build initial index
            self._build_initial_index()
            
            logger.info("RAG system initialization completed")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            raise
    
    def _build_initial_index(self) -> str:
        """Build initial vector index from existing documents."""
        try:
            logger.info("Building initial document index...")
            
            # Load documents from the documents directory
            documents = self.document_processor.load_documents_from_directory(
                settings.DOCUMENTS_DIR
            )
            
            if documents:
                # Split documents into chunks
                chunks = self.document_processor.split_documents(documents)
                
                if chunks:
                    # Create or load vector store
                    success = self.vector_store_manager.load_or_create_vector_store(chunks)
                    if success:
                        # Rebuild RAG chain
                        self.rag_chain_manager.rebuild_chain()
                        status = f"成功加载 {len(documents)} 个文档，生成 {len(chunks)} 个文本块"
                        logger.info(status)
                        return status
                    else:
                        status = "向量存储创建失败"
                        logger.error(status)
                        return status
                else:
                    status = "文档分割后未生成文本块"
                    logger.warning(status)
                    return status
            else:
                # Try to load existing vector store even without new documents
                success = self.vector_store_manager.load_or_create_vector_store()
                if success:
                    self.rag_chain_manager.rebuild_chain()
                    status = "未找到新文档，已加载现有索引"
                    logger.info(status)
                    return status
                else:
                    status = "未找到文档，系统将在上传文档后可用"
                    logger.info(status)
                    return status
                    
        except Exception as e:
            error_msg = f"构建初始索引时发生错误: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def upload_and_process_file(self, file_path: Path, destination_name: str) -> Tuple[str, List[str]]:
        """
        Upload and process a new file.
        
        Args:
            file_path: Source file path
            destination_name: Name for the destination file
            
        Returns:
            Tuple of (status_message, updated_document_list)
        """
        try:
            logger.info(f"Processing uploaded file: {destination_name}")
            
            # Ensure documents directory exists
            settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            
            # Copy file to documents directory
            destination_path = settings.DOCUMENTS_DIR / destination_name
            shutil.copy(str(file_path), str(destination_path))
            
            logger.info(f"File copied to: {destination_path}")
            
            # Rebuild index with all documents
            status = self.rebuild_index()
            
            # Get updated document list
            doc_list = self.get_document_list()
            
            final_status = f"文件 '{destination_name}' 上传成功。\n{status}"
            logger.info(f"File processing completed: {destination_name}")
            
            return final_status, doc_list
            
        except Exception as e:
            error_msg = f"文件上传或处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, self.get_document_list()
    
    def rebuild_index(self) -> str:
        """
        Rebuild the entire document index.
        
        Returns:
            Status message
        """
        try:
            logger.info("Rebuilding document index...")
            
            # Load all documents from directory
            documents = self.document_processor.load_documents_from_directory(
                settings.DOCUMENTS_DIR
            )
            
            if not documents:
                # Try to load existing vector store
                success = self.vector_store_manager.load_or_create_vector_store()
                if success:
                    self.rag_chain_manager.rebuild_chain()
                    return "未找到新文档，已使用现有数据重新加载系统"
                else:
                    return "错误：没有文档可加载，且没有现有的向量数据库"
            
            # Split documents
            chunks = self.document_processor.split_documents(documents)
            
            if not chunks:
                success = self.vector_store_manager.load_or_create_vector_store()
                if success:
                    self.rag_chain_manager.rebuild_chain()
                    return "警告：文档分割后未产生文本块，已使用现有数据"
                else:
                    return "错误：文档分割后未产生文本块，且无现有数据库"
            
            # Create new vector store (this will replace the existing one)
            success = self.vector_store_manager.load_or_create_vector_store(chunks)
            
            if not success:
                return "错误：创建向量数据库失败"
            
            # Rebuild RAG chain
            if not self.rag_chain_manager.rebuild_chain():
                return "警告：RAG链重建失败，但向量数据库已更新"
            
            status = f"索引重建完成：处理了 {len(documents)} 个文档，生成 {len(chunks)} 个文本块"
            logger.info(status)
            return status
            
        except Exception as e:
            error_msg = f"重建索引时发生错误: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def process_query(self, query: str) -> Iterator[str]:
        """
        Process a user query through the RAG system.
        
        Args:
            query: User question
            
        Yields:
            Response chunks
        """
        try:
            if not query or not query.strip():
                yield "请输入有效的问题。"
                return
            
            logger.info(f"Processing user query: {query}")
            
            if not self.is_ready():
                yield "错误：RAG 系统未就绪。请确保已上传文档并且系统正常运行。"
                return
            
            # Process query through RAG chain
            yield from self.rag_chain_manager.process_query(query.strip())
            
        except Exception as e:
            error_msg = f"处理查询时发生错误: {str(e)}"
            logger.error(error_msg)
            yield error_msg
    
    def get_document_list(self) -> List[str]:
        """
        Get list of loaded documents.
        
        Returns:
            List of document filenames
        """
        try:
            return self.document_processor.get_document_list(settings.DOCUMENTS_DIR)
        except Exception as e:
            logger.error(f"Error getting document list: {str(e)}")
            return []
    
    def get_document_list_markdown(self) -> str:
        """
        Get document list formatted as Markdown.
        
        Returns:
            Markdown formatted document list
        """
        try:
            doc_list = self.get_document_list()
            
            if not doc_list:
                return "当前没有已加载的文档。"
            
            markdown_list = "### 当前已加载文档:\n" + "\n".join([f"- {doc}" for doc in doc_list])
            return markdown_list
            
        except Exception as e:
            logger.error(f"Error formatting document list: {str(e)}")
            return "无法列出文档。"
    
    def is_ready(self) -> bool:
        """
        Check if the RAG system is ready for queries.
        
        Returns:
            True if system is ready, False otherwise
        """
        return (self.rag_chain_manager and 
                self.rag_chain_manager.is_ready())
    
    def get_system_status(self) -> dict:
        """
        Get comprehensive system status.
        
        Returns:
            Dictionary with system status information
        """
        try:
            doc_count = len(self.get_document_list())
            
            return {
                "ready": self.is_ready(),
                "documents_loaded": doc_count,
                "vector_store_ready": self.vector_store_manager.is_ready(),
                "llm_client_ready": self.llm_client.is_ready(),
                "rag_chain_ready": self.rag_chain_manager and self.rag_chain_manager.is_ready(),
                "documents_directory": str(settings.DOCUMENTS_DIR),
                "vector_store_directory": str(settings.PERSIST_DIR)
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            return {"error": str(e)}
