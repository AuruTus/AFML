---
description: "Use when editing Jupyter notebook (.ipynb) files with the notebook edit tool. edit_notebook_file with editType=edit replaces the ENTIRE cell - always supply the complete cell content, never a fragment."
---
# Notebook Editing with edit_notebook_file

`edit_notebook_file` with `editType="edit"` replaces the **whole cell**: `newCode` must be the complete cell source, including all function definitions, comments, and imports.

## Rules

- Before a full-cell edit, read the current cell (or confirm you have its exact content in context).
- To change a few lines (add a row, fix one line), still rewrite the **entire cell** with the updated content - never pass only the fragment.
- To insert or delete cells without rewriting neighbors, use `editType="insert"` / `editType="delete"` with the adjacent cell id.
- `replace_string_in_file` replaces only the matched text and is safe for partial edits; do not reach for it on notebooks, use it in plain code files.
- After a full-cell edit, verify the cell still defines everything it references (functions, imports) before moving on.

## Failure history (do not repeat)

- Sending only the `pd.DataFrame([...])` block as `newCode` wiped the cell's four helper functions (`band_side`, `ma_cross_side`, `fracdiff_side`, `run_pipeline`) → `NameError` on every downstream run.
- The same whole-cell trap recurred when "fixing one line": a single-fragment replacement overwrote an entire function body.
