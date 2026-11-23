"""Notebooks: a cell-aware three-way merge, and a readable diff.

The merge is nbdime's, run as a command — the student's environment has it
(``cs-lab[course]``), or ``uvx`` fetches it. A plain text merge of two
notebooks breaks their JSON; nbdime merges cell by cell, and with the
``use-local`` strategy the student's side wins where both changed the same
lines. Outputs are the student's.

That strategy can drop a cell the course added right where the student added
one of their own. :func:`missing_cells` tells, so that the update also leaves
the course's version next to the merged one.
"""

from __future__ import annotations

import difflib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Sequence


def _merge_command() -> Optional[List[str]]:
    if importlib.util.find_spec("nbdime") is not None:
        return [sys.executable, "-m", "nbdime", "merge"]
    if shutil.which("uvx"):
        return ["uvx", "--quiet", "--from", "nbdime", "nbmerge"]
    return None


def can_merge() -> bool:
    return _merge_command() is not None


class MergeError(Exception):
    pass


def merge(base: Path, local: Path, remote: Path, out: Path) -> None:
    """Merge into ``out``; the student's (local) side wins conflicts."""
    command = _merge_command()
    if command is None:
        raise MergeError("nbdime is not available (install cs-lab[course], or uv)")
    proc = subprocess.run(
        [
            *command,
            "--merge-strategy",
            "use-local",
            "--output-strategy",
            "use-local",
            str(base),
            str(local),
            str(remote),
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0 or not out.is_file():
        raise MergeError(proc.stderr.strip() or "nbdime merge failed")
    try:
        json.loads(out.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise MergeError(f"the merge did not give a valid notebook: {exc}") from exc


def _cells(path: Path) -> List[tuple]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    cells = []
    for cell in notebook.get("cells", []):
        source = cell.get("source", "")
        if not isinstance(source, str):
            source = "".join(source)
        cells.append((cell.get("cell_type", ""), source))
    return cells


#: How close a cell must be to a base cell to count as an edit of it rather
#: than a new cell (difflib's ratio).
SIMILAR = 0.6


def _resembles(cell: tuple, others: List[tuple]) -> bool:
    kind, source = cell
    return any(
        other_kind == kind
        and difflib.SequenceMatcher(None, source, other_source).ratio() >= SIMILAR
        for other_kind, other_source in others
    )


def missing_cells(base: Path, remote: Path, merged: Path) -> List[str]:
    """Cells the course added that the merge lost — their first line each.

    A cell of ``remote`` is *added* when nothing in ``base`` resembles it (an
    edited cell does). The student never saw it, so it cannot have been
    edited: it is *lost* when ``merged`` does not have it as is.
    """
    before, after = _cells(base), set(_cells(merged))
    return [
        (cell[1].splitlines() or [""])[0]
        for cell in _cells(remote)
        if cell not in after and not _resembles(cell, before)
    ]


def as_text(path: Optional[Path]) -> List[str]:
    """A notebook as text — cell sources, outputs left out — for a diff."""
    if path is None or not path.is_file():
        return []
    try:
        cells = _cells(path)
    except ValueError:
        return path.read_text(encoding="utf-8", errors="replace").splitlines(True)
    lines = []
    for cell_type, source in cells:
        lines.append(f"# ---- {cell_type} cell\n")
        lines.extend(line + "\n" for line in source.splitlines())
    return lines


def diff(local: Optional[Path], new: Path, labels: Sequence[str]) -> str:
    """A unified diff of the student's copy and the course's new one."""
    read = as_text if new.suffix == ".ipynb" else _plain
    return "".join(difflib.unified_diff(read(local), read(new), labels[0], labels[1]))


def _plain(path: Optional[Path]) -> List[str]:
    if path is None or not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines(True)
