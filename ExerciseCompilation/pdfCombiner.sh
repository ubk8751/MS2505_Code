#!/bin/bash

if [[ "$1" == "-h" || "$1" == "--help" ]]; then
    echo "Usage: $0 [OUTPUT_FILE] [LATEX_DIR]"
    echo
    echo "Arguments:"
    echo "  OUTPUT_FILE    Optional: The name of the output PDF file (default: 'combined_exercises.pdf')."
    echo "  LATEX_DIR      Optional: The path to the LaTeX build directory (default: 'latex_build')."
    exit 0
fi

# Check if the output file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <output.pdf>"
    exit 1
fi

# Set the output file
output_file="$1"

# Set the optional latex directory (use a default if not set)
latex_dir=${LATEX_DIR:-"latex_build"}

# List of PDF files
file_list=(
    "./ExerciseCompilation/$latex_dir/exercises.pdf" # Path to latex project pdf file
    "./ExerciseCompilation/form_new.pdf"            # Path to table of formulas
    "./ExerciseCompilation/Summary_main_results.pdf" # Path to main results list
)

# Check if all files exist
for file in "${file_list[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Error: File '$file' does not exist."
        exit 1
    fi
done

# Combine the PDF files using pdftk
pdftk "${file_list[@]}" cat output "$output_file"

# Check if the operation was successful
if [ $? -eq 0 ]; then
    echo "PDF files combined successfully into '$output_file'."
else
    echo "Error: Failed to combine PDF files."
    exit 1
fi