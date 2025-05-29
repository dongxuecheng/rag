"""
RAG chain management for combining retrieval and generation.
"""
from typing import Iterator, Optional, Dict, Any
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document

from config.settings import settings
from src.rag_system.core.vector_store_manager import VectorStoreManager
from src.rag_system.core.llm_client import LLMClient
from src.rag_system.utils.logger import get_logger

logger = get_logger()


class RAGChainManager:
    """Manages the RAG (Retrieval-Augmented Generation) chain."""
    
    def __init__(self, vector_store_manager: VectorStoreManager, llm_client: LLMClient):
        self.vector_store_manager = vector_store_manager
        self.llm_client = llm_client
        self.rag_chain = None
        self.retrieval_prompt = None
        self._initialize_chain()
    
    def _initialize_chain(self) -> None:
        """Initialize the RAG chain components."""
        try:
            if not self.vector_store_manager.is_ready():
                logger.warning("Vector store not ready, cannot initialize RAG chain")
                return
            
            if not self.llm_client.is_ready():
                logger.warning("LLM client not ready, cannot initialize RAG chain")
                return
            
            logger.info("Initializing RAG chain...")
            
            # Get retrieval prompt from LangChain Hub
            self.retrieval_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
            logger.debug("Retrieved QA prompt from LangChain Hub")
            
            # Create document combination chain
            combine_docs_chain = create_stuff_documents_chain(
                self.llm_client.client, 
                self.retrieval_prompt
            )
            
            # Get retriever from vector store
            retriever = self.vector_store_manager.get_retriever()
            if not retriever:
                logger.error("Failed to get retriever from vector store")
                return
            
            # Create the final retrieval chain
            self.rag_chain = create_retrieval_chain(retriever, combine_docs_chain)
            
            logger.info("RAG chain initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG chain: {str(e)}")
            self.rag_chain = None
    
    def rebuild_chain(self) -> bool:
        """
        Rebuild the RAG chain (useful after updating vector store).
        
        Returns:
            True if successful, False otherwise
        """
        try:
            logger.info("Rebuilding RAG chain...")
            self._initialize_chain()
            return self.is_ready()
            
        except Exception as e:
            logger.error(f"Failed to rebuild RAG chain: {str(e)}")
            return False
    
    def process_query(self, query: str) -> Iterator[str]:
        """
        Process a query through the RAG chain with streaming response.
        
        Args:
            query: User query
            
        Yields:
            Response chunks as they are generated
        """
        try:
            if not self.is_ready():
                yield "错误：RAG 系统未就绪，请检查配置。"
                return
            
            logger.info(f"Processing query: {query}")
            
            # Debug: Show retrieved documents
            if logger.level <= 10:  # DEBUG level
                self._log_retrieved_documents(query)
            
            # Stream response from RAG chain
            full_answer = ""
            
            try:
                response_stream = self.rag_chain.stream({"input": query})
                
                for chunk in response_stream:
                    # Extract answer from chunk
                    answer_part = chunk.get("answer", "")
                    if answer_part:
                        full_answer += answer_part
                        yield full_answer
                
                if not full_answer:
                    yield "抱歉，未能生成回答。请尝试重新表述您的问题。"
                else:
                    logger.info(f"Query processed successfully, response length: {len(full_answer)}")
                    
            except Exception as stream_error:
                logger.error(f"Error during streaming: {str(stream_error)}")
                yield f"处理查询时发生错误: {str(stream_error)}"
                
        except Exception as e:
            logger.error(f"Failed to process query: {str(e)}")
            yield f"处理查询时发生错误: {str(e)}"
    
    def _log_retrieved_documents(self, query: str) -> None:
        """Log retrieved documents for debugging purposes."""
        try:
            retrieved_docs = self.vector_store_manager.similarity_search(query)
            
            logger.debug(f"Retrieved {len(retrieved_docs)} documents for query: '{query}'")
            
            for i, doc in enumerate(retrieved_docs[:3]):  # Show only first 3 for brevity
                logger.debug(f"Doc {i+1}: {doc.page_content[:200]}...")
                logger.debug(f"Metadata: {doc.metadata}")
                
        except Exception as e:
            logger.error(f"Error logging retrieved documents: {str(e)}")
    
    def get_retrieval_info(self, query: str) -> Dict[str, Any]:
        """
        Get information about document retrieval for a query.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with retrieval information
        """
        try:
            if not self.vector_store_manager.is_ready():
                return {"error": "Vector store not ready"}
            
            retrieved_docs = self.vector_store_manager.similarity_search(query)
            
            return {
                "query": query,
                "num_retrieved": len(retrieved_docs),
                "documents": [
                    {
                        "content_preview": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                        "metadata": doc.metadata
                    }
                    for doc in retrieved_docs[:5]  # Return first 5 documents
                ]
            }
            
        except Exception as e:
            logger.error(f"Error getting retrieval info: {str(e)}")
            return {"error": str(e)}
    
    def is_ready(self) -> bool:
        """
        Check if the RAG chain is ready for use.
        
        Returns:
            True if chain is available, False otherwise
        """
        return (self.rag_chain is not None and 
                self.vector_store_manager.is_ready() and 
                self.llm_client.is_ready())
