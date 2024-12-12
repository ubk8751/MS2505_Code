# MS2505_Code

Code related to the course MS2505 Bayesian Statistics at Blekinge Institute of Technology.

Additional course material can be found at the [course repo](https://github.com/avehtari/BDA_course_Aalto).

## Export the compiled exercise pdf

The `ExerciseCompilation` folder contains a latex project with a compilation of all provided exercises during the course, as well as the table of formulae and summary of main results. If changes are made to the pdf file, run

    bash ./export_exercise_pdf.sh [OUTPUT_FILE] [LATEX_DIR] [-h]

### Argumens

- `OUTPUT_FILE` (optional): Name of the output PDF file. Defaults to "combined_exercises.pdf".
- `LATEX_DIR` (optional): Path to the LaTeX build directory. Defaults to "latex_build".
- `-h` (optional): Print options