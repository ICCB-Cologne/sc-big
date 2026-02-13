#!/bin/bash
FILE_BASE_NAME="paper"
TEX_FILE="$FILE_BASE_NAME"".tex"
PDF_FILE="$FILE_BASE_NAME"".pdf"
BIBLIOGRAPHY_FILE="bibliography.bib"

RED="\033[0;31m"
GREEN="\033[0;32m"
YELLOW="\033[1;33m"
BLUE="\033[0;34m"
NC="\033[0m"

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

clean_build() {
    rm -f *.aux *.log *.nav *.out *.snm *.toc *.bbl *.bcf *.blg *.run.xml *.fdb_latexmk *.fls
    print_success "Build files cleaned"
}

build_paper() {
    print_status "Running pdflatex (first pass)..."
    if ! pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1; then
        print_error "First pdflatex run failed"
        return 1
    fi

    print_status "Running biber..."
    if ! biber "$FILE_BASE_NAME" > /dev/null 2>&1; then
        print_warning "Biber run failed (bibliography may not be available)"
    fi

    print_status "Running pdflatex (second pass)..."
    if ! pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1; then
        print_error "Second pdflatex run failed"
        return 1
    fi

    print_status "Running pdflatex (final pass)..."
    if ! pdflatex -interaction=nonstopmode "$TEX_FILE" > /dev/null 2>&1; then
        print_error "Final pdflatex run failed"
        return 1
    fi

    if [ -f "$PDF_FILE" ]; then
        print_success "Build completed successfully: $PDF_FILE"
    else
        print_error "Build failed: $PDF_FILE not found"
        return 1
    fi
}

watch_mode() {
    print_status "Starting watch mode (press Ctrl+C to stop)..."
    if ! command -v inotifywait &> /dev/null; then
        print_error "inotifywait not found. Please install inotify-tools package."
        print_status "On Ubuntu/Debian: sudo apt-get install inotify-tools"
        return 1
    fi

    build_paper
    while true; do
        inotifywait -e modify "$TEX_FILE" "$BIBLIOGRAPHY_FILE" 2>/dev/null
        print_status "File changed, rebuilding..."
        build_paper
    done
}

case "${1:-build}" in
    "clean")
        clean_build
        ;;
    "watch")
        watch_mode
        ;;
    "build"|"")
        build_paper
        ;;
    *)
        echo "Usage: $0 [build|clean|watch]"
        echo "  build  - Build the presentation (default)"
        echo "  clean  - Clean build files"
        echo "  watch  - Watch for changes and rebuild automatically"
        exit 1
        ;;
esac
