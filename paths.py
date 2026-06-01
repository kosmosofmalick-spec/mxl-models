from pathlib import Path

R_PROJECT = Path(
    r"C:\Users\malik\Downloads\DynamicPhotosynthesis-master\DynamicPhotosynthesis-master"
)

R_OUTPUT = R_PROJECT / "Output"
R_INTERMEDIATE = R_PROJECT / "Intermediate"

PYTHON_PROJECT = Path(
    r"C:\Users\malik\OneDrive\Desktop\morales_mxlpy"
)

PYTHON_FIGURES = PYTHON_PROJECT / "python_figures"

PYTHON_FIGURES.mkdir(exist_ok=True)