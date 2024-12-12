# MS2505_Code

Code related to the course MS2505 Bayesian Statistics at Blekinge Institute of Technology.

Additional course material can be found at the [course repo](https://github.com/avehtari/BDA_course_Aalto).

## Export the compiled exercise pdf

The `ExerciseCompilation` folder contains a latex project with a compilation of all provided exercises during the course, as well as the table of formulae and summary of main results. If changes are made to the pdf file, run

    bash ./export_exercise_pdf.sh [--latex-dir LATEX_DIR] [--output OUTPUT_FILE] [-h]

### Argumens

- `--output` (optional): Name of the output PDF file. Defaults to "output_pdf.pdf".
- `--latex-dir` (optional): Path to the LaTeX build directory. Defaults to "ExerciseCompilation/latex_build".
- `-h` (optional): Print options