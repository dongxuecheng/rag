#!/bin/bash

# RAG System Startup Script
# This script helps you start the RAG system with proper environment setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python is available
check_python() {
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed or not in PATH"
        exit 1
    fi
    
    PYTHON_VERSION=$(python3 --version | awk '{print $2}')
    print_status "Found Python $PYTHON_VERSION"
}

# Check if required directories exist
check_directories() {
    print_status "Checking required directories..."
    
    # Create directories if they don't exist
    mkdir -p documents
    mkdir -p chroma_db
    mkdir -p logs
    
    print_success "Directories are ready"
}

# Activate conda rag environment
check_env() {
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "Conda is not installed or not in PATH"
        exit 1
    fi
    
    # Initialize conda for bash shell if needed
    if [ -z "$CONDA_EXE" ]; then
        # Try to find conda initialization script
        if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
            print_status "Initializing conda from $HOME/miniconda3/etc/profile.d/conda.sh"
            source "$HOME/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
            print_status "Initializing conda from /opt/miniconda3/etc/profile.d/conda.sh"
            source "/opt/miniconda3/etc/profile.d/conda.sh"
        elif [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
            print_status "Initializing conda from /usr/local/miniconda3/etc/profile.d/conda.sh"
            source "/usr/local/miniconda3/etc/profile.d/conda.sh"
        else
            print_error "Could not find conda initialization script"
            print_error "Please run 'conda init bash' and restart your shell"
            exit 1
        fi
    fi
    
    # Initialize conda in this shell session
    eval "$(conda shell.bash hook)"
    
    # Check if rag environment exists
    if ! conda env list | grep -q "rag"; then
        print_error "Conda environment 'rag' not found"
        print_error "Please create the environment first with: conda create -n rag python=3.11"
        exit 1
    fi
    
    # Activate rag environment
    print_status "Activating conda environment: rag"
    conda activate rag
    
    if [ "$CONDA_DEFAULT_ENV" = "rag" ]; then
        print_success "Successfully activated conda environment: rag"
        print_status "Python version in rag environment: $(python --version)"
    else
        print_error "Failed to activate conda environment: rag"
        exit 1
    fi
}

# Install dependencies
install_dependencies() {
    print_status "Skipping dependency installation (not required)"
    print_status "Dependencies should be managed through conda environment"
}

# Check environment configuration
check_config() {
    if [ ! -f ".env" ]; then
        if [ -f ".env.example" ]; then
            print_warning "No .env file found, copying from .env.example"
            cp .env.example .env
            print_warning "Please edit .env file with your specific configuration"
        else
            print_warning "No .env or .env.example file found"
            print_warning "Using default configuration settings"
        fi
    else
        print_success "Found .env configuration file"
    fi
}

# Check if VLLM server is running
check_vllm_server() {
    print_status "Checking VLLM server availability..."
    
    # Default URL, can be overridden by environment
    VLLM_URL=${VLLM_BASE_URL:-"http://localhost:8000"}
    
    if curl -s -f "${VLLM_URL}/health" > /dev/null 2>&1; then
        print_success "VLLM server is running at $VLLM_URL"
    else
        print_warning "VLLM server is not accessible at $VLLM_URL"
        print_warning "Please ensure VLLM server is running before starting the RAG system"
        print_warning "Example: vllm serve /path/to/model --host 0.0.0.0 --port 8000"
    fi
}

# Start the application
start_app() {
    print_status "Starting RAG System..."
    print_status "Access the web interface at http://localhost:7860 (or configured port)"
    print_status "Press Ctrl+C to stop the application"
    
    # Create startup logs directory
    STARTUP_LOG_DIR="logs/startup"
    mkdir -p "$STARTUP_LOG_DIR"
    
    # Generate timestamp for log file
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    STARTUP_LOG_FILE="${STARTUP_LOG_DIR}/${TIMESTAMP}_startup.log"
    
    print_status "Startup log will be saved to: $STARTUP_LOG_FILE"
    echo
    
    # Start the application with logging
    # Log both stdout and stderr to the startup log file
    python3 app.py 2>&1 | tee "$STARTUP_LOG_FILE"
}

# Main execution
main() {
    echo "========================================"
    echo "    RAG System Startup Script"
    echo "========================================"
    echo
    
    check_python
    check_directories
    check_env
    check_config
    
    # Parse command line arguments
    SKIP_DEPS=true  # Always skip dependencies
    SKIP_VLLM_CHECK=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --skip-vllm-check)
                SKIP_VLLM_CHECK=true
                shift
                ;;
            --help|-h)
                echo "Usage: $0 [options]"
                echo "Options:"
                echo "  --skip-vllm-check  Skip VLLM server check"
                echo "  --help, -h         Show this help message"
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                echo "Use --help for usage information"
                exit 1
                ;;
        esac
    done
    
    if [ "$SKIP_DEPS" = false ]; then
        install_dependencies
    fi
    
    if [ "$SKIP_VLLM_CHECK" = false ]; then
        check_vllm_server
    fi
    
    echo
    start_app
}

# Run main function
main "$@"
