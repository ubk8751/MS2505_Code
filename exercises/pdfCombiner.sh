#!/bin/bash

# Check if the input file is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <output.pdf>"
    exit 1
fi

# Set the output file
output_file="$1"

# List of PDF files
file_list=(
    "./exercises/latex_build/exercises.pdf" # Path to latex project pdf file
    "./exercises/form_new.pdf"              # Path to table of formulas
    "./exercises/Summary_main_results.pdf"  # Path to main results list
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
