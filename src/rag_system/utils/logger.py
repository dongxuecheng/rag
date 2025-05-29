"""
Logging utilities for the RAG system.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional
from datetime import datetime

from config.settings import settings


class RAGLogger:
    """Enhanced logger for RAG system with timestamped files and rotation."""
    
    _loggers = {}
    _startup_time = None
    
    @classmethod
    def _get_log_directory(cls, log_type: str) -> Path:
        """Get the appropriate log directory based on log type."""
        log_dirs = {
            'app': settings.LOGS_DIR / 'app',
            'error': settings.LOGS_DIR / 'error', 
            'startup': settings.LOGS_DIR / 'startup'
        }
        
        log_dir = log_dirs.get(log_type, settings.LOGS_DIR)
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir
    
    @classmethod
    def _get_startup_timestamp(cls) -> str:
        """Get or create startup timestamp for this session."""
        if cls._startup_time is None:
            cls._startup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        return cls._startup_time
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None) -> logging.Logger:
        """
        Get a configured logger instance.
        
        Args:
            name: Logger name
            log_file: Optional base log file name (without extension and timestamp)
            
        Returns:
            Configured logger instance
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, settings.LOG_LEVEL.value))
        
        # Prevent duplicate handlers
        if logger.handlers:
            return logger
        
        # Create formatter
        formatter = logging.Formatter(settings.LOG_FORMAT)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler with timestamped filename
        if log_file:
            timestamp = cls._get_startup_timestamp()
            log_filename = f"{timestamp}_{log_file}.log"
            
            # Determine log type and directory
            if log_file == "app":
                log_dir = cls._get_log_directory('app')
            elif log_file == "startup":
                log_dir = cls._get_log_directory('startup')
            else:
                log_dir = cls._get_log_directory('app')  # Default to app directory
            
            log_path = log_dir / log_filename
            
            file_handler = logging.handlers.RotatingFileHandler(
                filename=log_path,
                maxBytes=settings.LOG_FILE_MAX_SIZE,
                backupCount=settings.LOG_BACKUP_COUNT,
                encoding='utf-8'
            )
            file_handler.setLevel(getattr(logging, settings.LOG_LEVEL.value))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Create error-only log file in error directory
            if log_file == "app":
                error_log_filename = f"{timestamp}_error.log"
                error_log_dir = cls._get_log_directory('error')
                error_log_path = error_log_dir / error_log_filename
                error_handler = logging.handlers.RotatingFileHandler(
                    filename=error_log_path,
                    maxBytes=settings.LOG_FILE_MAX_SIZE,
                    backupCount=settings.LOG_BACKUP_COUNT,
                    encoding='utf-8'
                )
                error_handler.setLevel(logging.ERROR)
                error_handler.setFormatter(formatter)
                logger.addHandler(error_handler)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def setup_app_logging(cls) -> logging.Logger:
        """Setup main application logger with timestamped files."""
        return cls.get_logger("app", "app")
    
    @classmethod
    def setup_startup_logging(cls) -> logging.Logger:
        """Setup startup script logger with timestamped files."""
        return cls.get_logger("startup", "startup")
    
    @classmethod
    def get_current_log_files(cls) -> dict:
        """Get list of current session log files organized by type."""
        if cls._startup_time is None:
            return {}
        
        timestamp = cls._startup_time
        log_files = {
            'app': [],
            'error': [], 
            'startup': []
        }
        
        # Check each log directory for timestamped files
        for log_type in log_files.keys():
            log_dir = cls._get_log_directory(log_type)
            if log_dir.exists():
                for log_file in log_dir.glob(f"{timestamp}_*.log"):
                    log_files[log_type].append(str(log_file))
        
        return log_files


# Convenience function for getting the main logger
def get_logger() -> logging.Logger:
    """Get the main application logger."""
    return RAGLogger.setup_app_logging()
