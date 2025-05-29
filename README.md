# RAG System - Refactored v2

A modern, production-ready Retrieval-Augmented Generation (RAG) system built with LangChain, VLLM, and Gradio.

## 🚀 Features

- **Modular Architecture**: Clean separation of concerns with dedicated modules for each component
- **Modern UI**: Beautiful Gradio interface with ChatGPT-like styling and multiple tabs
- **Document Processing**: Support for PDF, DOCX, and TXT files with intelligent text splitting
- **Vector Storage**: Efficient document retrieval using Chroma vector database
- **LLM Integration**: Seamless integration with VLLM server for high-performance inference
- **Comprehensive Logging**: Structured logging with rotation and different log levels
- **Configuration Management**: Environment-based configuration with validation
- **Error Handling**: Robust error handling throughout the application

## 🏗️ Architecture

```
rag/
├── app.py                          # Main application entry point
├── start.sh                        # Startup script with health checks
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment configuration template
├── config/
│   └── settings.py                 # Pydantic-based configuration
└── src/rag_system/
    ├── core/                       # Core business logic
    │   ├── document_processor.py   # Document loading and processing
    │   ├── vector_store_manager.py # Vector database management
    │   ├── llm_client.py          # VLLM client with streaming
    │   ├── rag_chain_manager.py   # RAG chain coordination
    │   └── rag_system.py          # Main system orchestrator
    ├── ui/
    │   └── gradio_interface.py     # Modern Gradio interface
    └── utils/
        └── logger.py               # Enhanced logging utilities
```

## 🛠️ Environment Setup

### Prerequisites

#### Miniconda Installation
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash ~/Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

#### Miniconda Mirror Configuration (China)
```bash
conda config --set show_channel_urls yes
```

Edit `.condarc` file:
```yaml
channels:
  - defaults
show_channel_urls: true
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

Clear index cache:
```bash
conda clean -i
```

#### Pip Mirror Configuration (China)
```bash
python -m pip install -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --upgrade pip
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
```

#### Hugging Face Mirror Configuration (China)
Add to `~/.bashrc`:
```bash
export HF_ENDPOINT=https://hf-mirror.com
source ~/.bashrc
```

### Model Download

#### LLM Models (Qwen)
```bash
# Qwen3-8B for reasoning tasks
huggingface-cli download Qwen/Qwen3-8B --local-dir ~/model/qwen3-8b --local-dir-use-symlinks False
```

#### Embedding Models
```bash
# BGE-M3 (multilingual)
huggingface-cli download BAAI/bge-m3 --local-dir ~/model/bge-m3 --local-dir-use-symlinks False
```

### VLLM Server Setup

```bash
# Create conda environment
conda create -n rag python=3.12 -y
conda activate rag

# Install VLLM
pip install vllm

# Start VLLM server
vllm serve ~/model/qwen3-8b --enable-reasoning --reasoning-parser deepseek_r1 --max-model-len 16384 --host 0.0.0.0 --port 8000
```

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone https://github.com/dongxuecheng/rag.git
cd rag
git checkout refactor-rag-v2
```

### 2. Install Dependencies
```bash
# Install all required packages
pip install -r requirements.txt

# Or install manually:
pip install langchain langchain-community langchain-chroma langchain-huggingface langchain_openai
pip install pypdf docx2txt gradio
pip install pydantic python-dotenv chromadb sentence-transformers
```

### 3. Configure Environment
```bash
cp .env.example .env
# Edit .env file with your settings
```

### 4. Start the Application
```bash
# Method 1: Using startup script (recommended)
chmod +x start_app.sh
./start_app.sh

# Method 2: Direct Python execution
conda acitvate rag
python app.py
```

### 5. Access the Interface
Open your browser and navigate to `http://localhost:7860`

## ⚙️ Configuration

The application uses environment variables for configuration. Copy `.env.example` to `.env` and customize:

```env
# VLLM Server Configuration
VLLM_BASE_URL=http://localhost:8000/v1
VLLM_MODEL_NAME=qwen2.5-7b
VLLM_API_KEY=your_api_key_here

# Embedding Model
EMBEDDING_MODEL_NAME=BAAI/bge-large-zh-v1.5

# Vector Store
VECTOR_STORE_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Application Settings
APP_HOST=0.0.0.0
APP_PORT=7860
LOG_LEVEL=INFO
```

## 🎯 Usage

### Document Management
1. Navigate to the "Document Management" tab
2. Upload PDF, DOCX, or TXT files
3. Files are automatically processed and indexed

### Chat Interface
1. Use the "Chat" tab for conversations
2. Ask questions about uploaded documents
3. View streaming responses in real-time

### System Monitoring
1. Check the "System Status" tab for health information
2. Monitor document count and system status
3. View recent activity logs

## 🧪 Testing

Test the system components:

```bash
# Test the main application
python -c "
from src.rag_system.core.rag_system import RAGSystem
system = RAGSystem()
print('✅ System initialized successfully')
"

# Test document processing
python -c "
from src.rag_system.core.document_processor import DocumentProcessor
processor = DocumentProcessor()
print('✅ Document processor ready')
"

# Test vector store
python -c "
from src.rag_system.core.vector_store_manager import VectorStoreManager
manager = VectorStoreManager()
print('✅ Vector store ready')
"
```

## 📝 Logging

Logs are organized in the `logs/` directory with timestamped files and automatic rotation:

### Log Directory Structure
```
logs/
├── app/          # Application logs (YYYYMMDD_HHMMSS_app.log)
├── error/        # Error-only logs (YYYYMMDD_HHMMSS_error.log)
└── startup/      # Startup script logs (YYYYMMDD_HHMMSS_startup.log)
```

### Log File Features
- **Timestamped Files**: Each application session creates new log files with startup timestamp
- **Organized by Type**: Separate directories for different log types
- **Automatic Rotation**: Files rotate when they reach size limits
- **Error Filtering**: Dedicated error logs for easier debugging
- **Startup Tracking**: Script execution logs saved separately

### Log Management
Use the provided cleanup script to manage old log files:
```bash
# Clean logs older than 7 days (default)
./scripts/cleanup_logs.sh

# Clean logs older than 14 days
./scripts/cleanup_logs.sh --days 14

# Preview what would be deleted (dry run)
./scripts/cleanup_logs.sh --dry-run
```

## 🔧 Development

### Project Structure
- `config/` - Configuration management
- `src/rag_system/core/` - Core business logic
- `src/rag_system/ui/` - User interface components
- `src/rag_system/utils/` - Utility functions
- `logs/` - Application logs organized by type (auto-created)
  - `logs/app/` - Application session logs
  - `logs/error/` - Error-only logs
  - `logs/startup/` - Startup script logs
- `chroma_db/` - Vector database storage (auto-created)
- `scripts/` - Utility scripts (log cleanup, etc.)

### Adding New Features
1. Create new modules in appropriate directories
2. Update configuration in `config/settings.py` if needed
3. Add imports to `__init__.py` files
4. Test thoroughly before committing

## 🐛 Troubleshooting

### Common Issues

1. **VLLM Connection Error**
   - Ensure VLLM server is running on the correct port
   - Check `VLLM_BASE_URL` in your `.env` file

2. **Model Not Found**
   - Verify model paths in your environment configuration
   - Ensure models are downloaded correctly

3. **Memory Issues**
   - Reduce `CHUNK_SIZE` in configuration
   - Use lighter embedding models for testing

4. **Port Already in Use**
   - Change `APP_PORT` in `.env` file
   - Kill existing processes using the port

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the RAG framework
- [VLLM](https://github.com/vllm-project/vllm) for high-performance LLM inference
- [Gradio](https://github.com/gradio-app/gradio) for the web interface
- [Chroma](https://github.com/chroma-core/chroma) for vector storage
- [Qwen](https://github.com/QwenLM/Qwen) for the language models
- [BGE](https://github.com/FlagOpen/FlagEmbedding) for embedding models
