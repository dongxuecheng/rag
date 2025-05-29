#!/usr/bin/env python3
"""
Main application entry point for the RAG System.
"""
import sys
import signal
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config.settings import settings
from src.rag_system.core.rag_system import RAGSystem
from src.rag_system.ui.gradio_interface import RAGInterface
from src.rag_system.utils.logger import get_logger, RAGLogger

logger = get_logger()


class RAGApplication:
    """Main application class for the RAG system."""
    
    def __init__(self):
        self.rag_system = None
        self.interface = None
        
    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            logger.info("Received shutdown signal, cleaning up...")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def validate_environment(self) -> bool:
        """Validate the environment and configuration."""
        try:
            logger.info("Validating environment...")
            
            # Check if embedding model path exists
            if not Path(settings.EMBEDDING_MODEL_PATH).exists():
                logger.error(f"Embedding model not found at: {settings.EMBEDDING_MODEL_PATH}")
                return False
            
            # Check if directories are accessible
            try:
                settings.DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
                settings.PERSIST_DIR.mkdir(parents=True, exist_ok=True)
                settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Cannot create required directories: {str(e)}")
                return False
            
            logger.info("Environment validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Environment validation failed: {str(e)}")
            return False
    
    def initialize_system(self) -> bool:
        """Initialize the RAG system."""
        try:
            logger.info("Initializing RAG system...")
            logger.info(f"Configuration: Embedding model: {settings.EMBEDDING_MODEL_PATH}")
            logger.info(f"Configuration: LLM server: {settings.VLLM_BASE_URL}")
            logger.info(f"Configuration: Documents directory: {settings.DOCUMENTS_DIR}")
            
            self.rag_system = RAGSystem()
            
            if not self.rag_system.is_ready():
                logger.warning("RAG system initialized but not ready for queries")
                logger.warning("This is normal if no documents are loaded yet")
            else:
                logger.info("RAG system initialized and ready")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {str(e)}")
            return False
    
    def initialize_interface(self) -> bool:
        """Initialize the Gradio interface."""
        try:
            logger.info("Initializing user interface...")
            self.interface = RAGInterface(self.rag_system)
            logger.info("User interface initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize interface: {str(e)}")
            return False
    
    def run(self) -> None:
        """Run the application."""
        try:
            logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
            logger.info("=" * 60)
            
            # Log session information
            log_files = RAGLogger.get_current_log_files()
            if log_files:
                logger.info(f"Log files for this session:")
                for log_type, files in log_files.items():
                    if files:
                        logger.info(f"  {log_type.capitalize()} logs:")
                        for log_file in files:
                            logger.info(f"    {log_file}")
            else:
                logger.info(f"Logs will be stored in: {settings.LOGS_DIR}")
                logger.info(f"  - App logs: {settings.LOGS_DIR}/app/")
                logger.info(f"  - Error logs: {settings.LOGS_DIR}/error/")
                logger.info(f"  - Startup logs: {settings.LOGS_DIR}/startup/")
            
            # Setup signal handlers
            self.setup_signal_handlers()
            
            # Validate environment
            if not self.validate_environment():
                logger.error("Environment validation failed, exiting")
                sys.exit(1)
            
            # Initialize system components
            if not self.initialize_system():
                logger.error("System initialization failed, exiting")
                sys.exit(1)
            
            if not self.initialize_interface():
                logger.error("Interface initialization failed, exiting")
                sys.exit(1)
            
            # Print system status
            status = self.rag_system.get_system_status()
            logger.info("System Status:")
            for key, value in status.items():
                logger.info(f"  {key}: {value}")
            
            logger.info("=" * 60)
            logger.info("Application startup completed successfully")
            logger.info(f"Starting web interface on {settings.GRADIO_SERVER_NAME}:{settings.GRADIO_SERVER_PORT or 'auto'}")
            
            # Launch the interface
            self.interface.launch()
            
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        except Exception as e:
            logger.error(f"Application error: {str(e)}")
            sys.exit(1)
        finally:
            logger.info("Application shutdown complete")


def main():
    """Main entry point."""
    try:
        app = RAGApplication()
        app.run()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
