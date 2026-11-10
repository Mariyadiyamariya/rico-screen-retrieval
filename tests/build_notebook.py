"""Convert notebooks/project_2_2.py (cells separated by '# %%') into a Colab .ipynb."""
import json
import re
import sys
from pathlib import Path

src = Path(sys.argv[1] if len(sys.argv) > 1 else "notebooks/project_2_2.py").read_text()
dst = Path(sys.argv[2] if len(sys.argv) > 2 else "notebooks/Project_2_2_Implementation.ipynb")
cells = []
for block in re.split(r"^# %%", src, flags=re.M)[1:]:
    is_md = block.startswith(" [markdown]")
    body = block.split("\n", 1)[1] if "\n" in block else ""
    body = body.strip("\n")
    if is_md:
        body = "\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())
        cells.append({"cell_type": "markdown", "metadata": {}, "source": body.splitlines(True)})
    else:
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": body.splitlines(True)})
nb = {"cells": cells, "metadata": {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"},
                                   "kernelspec": {"name": "python3", "display_name": "Python 3"},
                                   "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 0}
dst.write_text(json.dumps(nb, indent=1))
print(f"wrote {dst} with {len(cells)} cells")
