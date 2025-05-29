"""
LLM client for interacting with VLLM server.
"""
from typing import Iterator, Optional
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import BaseMessage

from config.settings import settings
from src.rag_system.utils.logger import get_logger

logger = get_logger()


class StreamingCallbackHandler(BaseCallbackHandler):
    """Callback handler for streaming LLM responses."""
    
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        """Handle new token from LLM."""
        self.tokens.append(token)
    
    def reset(self):
        """Reset the token buffer."""
        self.tokens = []


class LLMClient:
    """Client for interacting with VLLM server through OpenAI-compatible API."""
    
    def __init__(self):
        self.client = None
        self.streaming_handler = StreamingCallbackHandler()
        self._initialize_client()
    
    def _initialize_client(self) -> None:
        """Initialize the ChatOpenAI client."""
        try:
            logger.info(f"Initializing LLM client for VLLM server at {settings.VLLM_BASE_URL}")
            logger.info(f"Using model: {settings.VLLM_MODEL_NAME}")
            
            self.client = ChatOpenAI(
                openai_api_base=settings.VLLM_BASE_URL,
                openai_api_key=settings.VLLM_API_KEY,
                model_name=settings.VLLM_MODEL_NAME,
                temperature=settings.VLLM_TEMPERATURE,
                max_tokens=settings.VLLM_MAX_TOKENS,
                streaming=True  # Enable streaming by default
            )
            
            logger.info("LLM client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM client: {str(e)}")
            raise
    
    def generate_response(self, messages: list) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of messages for the conversation
            
        Returns:
            Generated response text
        """
        try:
            if not self.client:
                raise ValueError("LLM client not initialized")
            
            logger.debug(f"Generating response for {len(messages)} messages")
            response = self.client.invoke(messages)
            
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Failed to generate response: {str(e)}")
            return f"Error generating response: {str(e)}"
    
    def stream_response(self, messages: list) -> Iterator[str]:
        """
        Stream a response from the LLM.
        
        Args:
            messages: List of messages for the conversation
            
        Yields:
            Response chunks as they are generated
        """
        try:
            if not self.client:
                raise ValueError("LLM client not initialized")
            
            logger.debug(f"Streaming response for {len(messages)} messages")
            
            # Use the streaming capability
            for chunk in self.client.stream(messages):
                if hasattr(chunk, 'content') and chunk.content:
                    yield chunk.content
                elif hasattr(chunk, 'message') and hasattr(chunk.message, 'content'):
                    yield chunk.message.content
                    
        except Exception as e:
            logger.error(f"Failed to stream response: {str(e)}")
            yield f"Error streaming response: {str(e)}"
    
    def is_ready(self) -> bool:
        """
        Check if the LLM client is ready for use.
        
        Returns:
            True if client is available, False otherwise
        """
        return self.client is not None
    
    def test_connection(self) -> bool:
        """
        Test the connection to the VLLM server.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            if not self.client:
                return False
            
            # Send a simple test message
            test_response = self.client.invoke([{"role": "user", "content": "Hello"}])
            logger.info("LLM connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"LLM connection test failed: {str(e)}")
            return False
