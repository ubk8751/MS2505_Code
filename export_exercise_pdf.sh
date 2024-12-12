#!/bin/bash

# Display help message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [--output OUTPUT_FILE] [--latex-dir LATEX_DIR]"
    echo
    echo "Arguments:"
    echo "  --output      Optional: The name of the output PDF file (default: 'output.pdf')."
    echo "  --latex-dir   Optional: The path to the LaTeX build directory (default: 'ExerciseCompilation/latex_build')."
    exit 0
fi

# Default values
output_file="output.pdf"
latex_dir="ExerciseCompilation/latex_build"

# Parse arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --output)
            output_file="$2"
            shift 2
            ;;
        --latex-dir)
            latex_dir="$2"
            shift 2
            ;;
        *)
            echo "Error: Unknown argument '$1'"
            echo "Use -h or --help for usage information."
            exit 1
            ;;
    esac
done

# Collect all .pdf files in the $latex_dir directory
pdf_files=("$latex_dir"/*.pdf)

# Additional fixed PDF files
additional_files=(
    "./ExerciseCompilation/form_new.pdf"    # Path to table of formulas
    "./ExerciseCompilation/Summary_main_results.pdf" # Path to main results list
)

# Combine all PDF files into one list
file_list=("${pdf_files[@]}" "${additional_files[@]}")

# Check if all files exist
for file in "${file_list[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "Error: File '$file' does not exist."
        exit 1
    fi
done

# Check if pdftk is installed
if ! command -v pdftk &> /dev/null; then
    echo "Error: pdftk is not installed. Please install it and try again."
    exit 1
fi

# Combine the PDF files using pdftk
pdftk "${file_list[@]}" cat output "$output_file"

# Check if the operation was successful
if [[ $? -eq 0 ]]; then
    echo "PDF files combined successfully into '$output_file'."
else
    echo "Error: Failed to combine PDF files."
    exit 1
fi
