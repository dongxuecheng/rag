#!/bin/bash

# Log Cleanup Script for RAG System
# This script helps manage log files by cleaning up old logs

set -e

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

# Configuration
LOGS_DIR="logs"
DEFAULT_KEEP_DAYS=7
LOG_SUBDIRS=("app" "error" "startup")

# Parse command line arguments
KEEP_DAYS=${DEFAULT_KEEP_DAYS}
DRY_RUN=false
HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --days)
            KEEP_DAYS="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            HELP=true
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            HELP=true
            shift
            ;;
    esac
done

if [ "$HELP" = true ]; then
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  --days N       Keep logs from last N days (default: $DEFAULT_KEEP_DAYS)"
    echo "  --dry-run      Show what would be deleted without actually deleting"
    echo "  --help, -h     Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Clean logs older than 7 days"
    echo "  $0 --days 14         # Keep logs from last 14 days"
    echo "  $0 --dry-run         # Preview what would be deleted"
    exit 0
fi

# Check if logs directory exists
if [ ! -d "$LOGS_DIR" ]; then
    print_error "Logs directory '$LOGS_DIR' not found"
    exit 1
fi

print_status "RAG System Log Cleanup"
print_status "Logs directory: $(realpath $LOGS_DIR)"
print_status "Keep logs from last: $KEEP_DAYS days"

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN MODE - No files will be deleted"
fi

echo

# Find log files in all subdirectories
cd "$LOGS_DIR"

# Count total log files across all subdirectories
total_logs=0
for subdir in "${LOG_SUBDIRS[@]}"; do
    if [ -d "$subdir" ]; then
        subdir_logs=$(find "$subdir" -name "*.log" -type f 2>/dev/null | wc -l)
        total_logs=$((total_logs + subdir_logs))
        if [ $subdir_logs -gt 0 ]; then
            print_status "Found $subdir_logs log files in $subdir/"
        fi
    else
        print_status "Directory $subdir/ does not exist yet"
    fi
done

# Also check for logs in root logs directory (legacy)
root_logs=$(find . -maxdepth 1 -name "*.log" -type f 2>/dev/null | wc -l)
if [ $root_logs -gt 0 ]; then
    total_logs=$((total_logs + root_logs))
    print_status "Found $root_logs log files in root logs directory"
fi

print_status "Found $total_logs log files total"

if [ $total_logs -eq 0 ]; then
    print_status "No log files found, nothing to clean"
    exit 0
fi

# Find old log files (older than KEEP_DAYS) in all directories
old_logs_list=""
old_count=0

# Check subdirectories
for subdir in "${LOG_SUBDIRS[@]}"; do
    if [ -d "$subdir" ]; then
        subdir_old_logs=$(find "$subdir" -name "*.log" -type f -mtime +${KEEP_DAYS} 2>/dev/null || true)
        if [ -n "$subdir_old_logs" ]; then
            old_logs_list="$old_logs_list$subdir_old_logs"$'\n'
            subdir_old_count=$(echo "$subdir_old_logs" | grep -c "\.log$" || echo "0")
            old_count=$((old_count + subdir_old_count))
        fi
    fi
done

# Check root directory for legacy logs
root_old_logs=$(find . -maxdepth 1 -name "*.log" -type f -mtime +${KEEP_DAYS} 2>/dev/null || true)
if [ -n "$root_old_logs" ]; then
    old_logs_list="$old_logs_list$root_old_logs"$'\n'
    root_old_count=$(echo "$root_old_logs" | grep -c "\.log$" || echo "0")
    old_count=$((old_count + root_old_count))
fi

if [ $old_count -eq 0 ]; then
    print_success "No old log files found (older than $KEEP_DAYS days)"
    exit 0
fi

print_status "Found $old_count log files older than $KEEP_DAYS days:"
echo "$old_logs_list" | while read -r file; do
    if [ -n "$file" ] && [ "$file" != "." ]; then
        file_size=$(ls -lh "$file" 2>/dev/null | awk '{print $5}' || echo "unknown")
        file_date=$(ls -l "$file" 2>/dev/null | awk '{print $6, $7, $8}' || echo "unknown")
        echo "  $file ($file_size, $file_date)"
    fi
done

echo

if [ "$DRY_RUN" = true ]; then
    print_warning "DRY RUN: Would delete $old_count files"
    exit 0
fi

# Ask for confirmation
read -p "Do you want to delete these $old_count old log files? (y/N): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    print_status "Operation cancelled"
    exit 0
fi

# Delete old log files
deleted_count=0
echo "$old_logs_list" | while read -r file; do
    if [ -n "$file" ] && [ "$file" != "." ] && [ -f "$file" ]; then
        rm -f "$file"
        deleted_count=$((deleted_count + 1))
        print_status "Deleted: $file"
    fi
done

print_success "Cleanup completed"
print_success "Deleted $old_count old log files"

# Show remaining logs in all directories
remaining_logs=0
for subdir in "${LOG_SUBDIRS[@]}"; do
    if [ -d "$subdir" ]; then
        subdir_remaining=$(find "$subdir" -name "*.log" -type f 2>/dev/null | wc -l)
        remaining_logs=$((remaining_logs + subdir_remaining))
    fi
done

# Also count root directory logs
root_remaining=$(find . -maxdepth 1 -name "*.log" -type f 2>/dev/null | wc -l)
remaining_logs=$((remaining_logs + root_remaining))

print_status "Remaining log files: $remaining_logs"

if [ $remaining_logs -gt 0 ]; then
    echo
    print_status "Current log files by directory:"
    
    # Show logs in each subdirectory
    for subdir in "${LOG_SUBDIRS[@]}"; do
        if [ -d "$subdir" ]; then
            subdir_logs=$(find "$subdir" -name "*.log" -type f 2>/dev/null)
            if [ -n "$subdir_logs" ]; then
                echo "  $subdir/:"
                echo "$subdir_logs" | while read -r file; do
                    if [ -n "$file" ]; then
                        file_size=$(ls -lh "$file" 2>/dev/null | awk '{print $5}' || echo "unknown")
                        file_date=$(ls -l "$file" 2>/dev/null | awk '{print $6, $7, $8}' || echo "unknown")
                        echo "    $(basename "$file") ($file_size, $file_date)"
                    fi
                done
            fi
        fi
    done
    
    # Show root directory logs if any
    root_logs=$(find . -maxdepth 1 -name "*.log" -type f 2>/dev/null)
    if [ -n "$root_logs" ]; then
        echo "  root:"
        echo "$root_logs" | while read -r file; do
            if [ -n "$file" ]; then
                file_size=$(ls -lh "$file" 2>/dev/null | awk '{print $5}' || echo "unknown")
                file_date=$(ls -l "$file" 2>/dev/null | awk '{print $6, $7, $8}' || echo "unknown")
                echo "    $(basename "$file") ($file_size, $file_date)"
            fi
        done
    fi
fi
