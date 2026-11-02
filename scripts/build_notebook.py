"""Build the self-contained notebook and full cell listing from percent-format source.

Maintainer utility using only the Python standard library. Users upload the
already-generated .ipynb to Colab; no local execution is needed.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "notebooks/01_model_evaluation.py"
parts = re.split(r"^# %%([^\n]*)\n", source.read_text(encoding="utf-8"), flags=re.MULTILINE)
cells = []
listing = ["# Complete Colab cells\n\nUpload the `.ipynb` to Colab, or copy each code block below into a separate code cell in this order. Do not run the source locally. All helper functions are included.\n"]
for index in range(1, len(parts), 2):
    kind, body = parts[index], parts[index + 1].rstrip() + "\n"
    markdown = "[markdown]" in kind
    if markdown:
        body = "\n".join(line[2:] if line.startswith("# ") else "" if line == "#" else line
                          for line in body.splitlines()) + "\n"
    cell = {"cell_type": "markdown" if markdown else "code", "metadata": {},
            "id": f"cell-{len(cells):02d}", "source": body.splitlines(keepends=True)}
    if not markdown:
        compile(body, f"cell-{len(cells):02d}", "exec")
        cell.update(execution_count=None, outputs=[])
    cells.append(cell)
    listing.append(body + "\n" if markdown else "```python\n" + body + "```\n\n")
notebook = {"cells": cells, "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": [], "name": "German Pathology AI — Model Evaluation Dashboard"},
}, "nbformat": 4, "nbformat_minor": 5}
(ROOT / "notebooks/01_model_evaluation.ipynb").write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n")
(ROOT / "COLAB_CELLS.md").write_text("\n".join(listing), encoding="utf-8")
print(f"Built {len(cells)} cells ({sum(c['cell_type'] == 'code' for c in cells)} code cells).")
