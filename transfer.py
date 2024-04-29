import nbformat

# Load the notebook
notebook_path = 'train.ipynb'
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = nbformat.read(f, as_version=4)

# Extract and display the code cells
code_cells = []
for cell in nb.cells:
    if cell.cell_type == 'code':
        code_cells.append(cell.source)

code_cells

