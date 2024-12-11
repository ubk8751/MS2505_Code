# MS2505_Code

Code related to the course MS2505 Bayesian Statistics at Blekinge Institute of Technology.

Additional course material can be found at the [course repo](https://github.com/avehtari/BDA_course_Aalto).

## ExerciseCompilation folder

The folder contains a latex project with a compilation of all provided exercises during the course. If changes are made to the pdf file, run

    bash ./ExerciseCompilation/pdfCombiner.sh [OUTPUT_FILE] [LATEX_DIR] [-h]

### Argumens

- `OUTPUT_FILE` (optional): Name of the output PDF file. Defaults to "combined_exercises.pdf".
- `LATEX_DIR` (optional): Path to the LaTeX build directory. Defaults to "latex_build".
- `-h` (optional): Print options