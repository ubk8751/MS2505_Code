#!/bin/bash

# Display help message
if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [OUTPUT_FILE] [LATEX_DIR]"
    echo
    echo "Arguments:"
    echo "  OUTPUT_FILE    Optional: The name of the output PDF file (default: 'combined_exercises.pdf')."
    echo "  LATEX_DIR      Optional: The path to the LaTeX build directory (default: 'ExerciseCompilation/latex_build')."
    exit 0
fi

# Set default values for OUTPUT_FILE and LATEX_DIR
default_output_file="combined_exercises.pdf"
default_latex_dir="ExerciseCompilation/latex_build"

# Use provided arguments or defaults
output_file="${1:-$default_output_file}"
latex_dir="${2:-$default_latex_dir}"

# List of PDF files to combine
file_list=(
    "$latex_dir/exercises.pdf"               # Path to LaTeX project PDF file
    "./ExerciseCompilation/form_new.pdf"    # Path to table of formulas
    "./ExerciseCompilation/Summary_main_results.pdf" # Path to main results list
)

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
