"""
RAG System Configuration Settings
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings
from enum import Enum


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ModelDevice(str, Enum):
    CPU = "cpu"
    CUDA_0 = "cuda:0"
    CUDA_1 = "cuda:1"
    AUTO = "auto"


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Project paths
    PROJECT_ROOT: Path = Path(__file__).parent.parent
    DOCUMENTS_DIR: Path = PROJECT_ROOT / "documents"
    PERSIST_DIR: Path = PROJECT_ROOT / "chroma_db"
    LOGS_DIR: Path = PROJECT_ROOT / "logs"
    
    # Embedding model configuration
    EMBEDDING_MODEL_PATH: str = Field(
        default="/mnt/dxc/model/bge-m3",
        description="Path to the embedding model"
    )
    EMBEDDING_DEVICE: ModelDevice = Field(
        default=ModelDevice.CUDA_1,
        description="Device for embedding model"
    )
    
    # VLLM server configuration
    VLLM_BASE_URL: str = Field(
        default="http://localhost:8000/v1",
        description="VLLM server base URL"
    )
    VLLM_API_KEY: str = Field(
        default="dummy-key",
        description="API key for VLLM server"
    )
    VLLM_MODEL_NAME: str = Field(
        default="/mnt/dxc/model/qwen3-8b",
        description="Model name for VLLM server"
    )
    VLLM_TEMPERATURE: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for text generation"
    )
    VLLM_MAX_TOKENS: int = Field(
        default=2048,
        ge=1,
        le=8192,
        description="Maximum tokens for response"
    )
    
    # Text processing configuration
    CHUNK_SIZE: int = Field(
        default=512,
        ge=128,
        le=2048,
        description="Size of text chunks for processing"
    )
    CHUNK_OVERLAP: int = Field(
        default=100,
        ge=0,
        le=512,
        description="Overlap between text chunks"
    )
    SEARCH_K: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Number of documents to retrieve"
    )
    
    # Supported file types
    SUPPORTED_FILE_EXTENSIONS: List[str] = Field(
        default=[".pdf", ".docx", ".doc", ".txt"],
        description="Supported file extensions for document upload"
    )
    
    # Gradio UI configuration
    GRADIO_SERVER_NAME: str = Field(
        default="0.0.0.0",
        description="Gradio server host"
    )
    GRADIO_SERVER_PORT: Optional[int] = Field(
        default=None,
        description="Gradio server port (None for auto)"
    )
    GRADIO_SHARE: bool = Field(
        default=False,
        description="Enable Gradio sharing"
    )
    
    # Logging configuration
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level"
    )
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    LOG_FILE_MAX_SIZE: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum log file size in bytes"
    )
    LOG_BACKUP_COUNT: int = Field(
        default=5,
        description="Number of backup log files to keep"
    )
    
    # Application metadata
    APP_NAME: str = "耀安科技-煤矿大模型知识问答系统"
    APP_VERSION: str = "2.0.0"
    APP_DESCRIPTION: str = "基于RAG技术的智能知识问答系统"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    def model_post_init(self, __context) -> None:
        """Create necessary directories after initialization."""
        self.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
        self.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
