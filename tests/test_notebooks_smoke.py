"""Notebook smoke tests by executing code cells as plain Python."""

from __future__ import annotations

import json
from pathlib import Path


def _execute_notebook(path: Path) -> None:
    nb = json.loads(path.read_text(encoding="utf-8"))
    ns = {}
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        exec(compile(src, str(path), "exec"), ns, ns)


def test_notebook_figure3_executes() -> None:
    _execute_notebook(Path("notebooks/01_figure3.ipynb"))


def test_notebook_figure6_executes() -> None:
    _execute_notebook(Path("notebooks/02_figure6.ipynb"))


def test_notebook_sweep_executes() -> None:
    _execute_notebook(Path("notebooks/03_sweeps.ipynb"))
