"""The environment files: ``pyproject.toml`` and ``uv.lock``.

They are not content to merge line by line — a text merge of ``uv.lock``
gives a file uv refuses to read. Instead:

* the student did not change them: take the course's;
* they did: take the course's, then replay the student's own changes to the
  dependency list with ``uv add`` / ``uv remove``. uv keeps every version the
  course's lock pins and resolves only what the student added. Their previous
  ``pyproject.toml`` is kept as ``pyproject.mine.toml``.

With a shared environment, the folder has no environment files at all; the
update only checks that the shared one was built from the release's lock.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .folder import Folder, sibling
from .release import sha256_file

PYPROJECT = "pyproject.toml"
LOCK = "uv.lock"

_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")


def project_name(requirement: str) -> str:
    match = _NAME.match(requirement)
    name = match.group(1) if match else requirement.strip()
    return re.sub(r"[-_.]+", "-", name).lower()


def dependencies(path: Path) -> List[str]:
    if not path.is_file():
        return []
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    return list(data.get("project", {}).get("dependencies", []))


@dataclass
class DependencyChanges:
    added: List[str] = field(default_factory=list)  # requirements, as written
    removed: List[str] = field(default_factory=list)  # project names

    def __bool__(self) -> bool:
        return bool(self.added or self.removed)


def student_changes(base: Path, local: Path) -> DependencyChanges:
    """What the student did to the dependency list since the base release."""
    before = {" ".join(r.split()) for r in dependencies(base)}
    after = [" ".join(r.split()) for r in dependencies(local)]
    names_after = {project_name(r) for r in after}
    return DependencyChanges(
        added=[r for r in after if r not in before],
        removed=sorted(
            {project_name(r) for r in before} - names_after,
        ),
    )


@dataclass
class EnvResult:
    changed: bool = False
    replayed: DependencyChanges = field(default_factory=DependencyChanges)
    backup: Optional[Path] = None
    error: str = ""


def _uv(args: List[str], cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(["uv", *args], cwd=cwd, capture_output=True, text=True)


def update_own(folder: Folder, release_dir: Path, reset: bool = False) -> EnvResult:
    """Install the release's environment files, keeping the student's
    dependency changes (unless ``reset``)."""
    root, base = folder.root, folder.base
    new_py, new_lock = release_dir / PYPROJECT, release_dir / LOCK
    if not new_py.is_file():
        return EnvResult()

    local_py, local_lock = root / PYPROJECT, root / LOCK
    touched = Folder.digest(local_py) != Folder.digest(base / PYPROJECT) or (
        Folder.digest(local_lock) != Folder.digest(base / LOCK)
    )
    upstream = Folder.digest(new_py) != Folder.digest(base / PYPROJECT) or (
        Folder.digest(new_lock) != Folder.digest(base / LOCK)
    )
    if not upstream and not reset:
        return EnvResult()

    changes = DependencyChanges()
    result = EnvResult(changed=True)
    if touched and local_py.is_file():
        changes = student_changes(base / PYPROJECT, local_py)
        result.backup = sibling(local_py, "mine")
        shutil.copy2(local_py, result.backup)

    shutil.copy2(new_py, local_py)
    if new_lock.is_file():
        shutil.copy2(new_lock, local_lock)
    if reset or not changes:
        return result

    result.replayed = changes
    return _replay(root, changes, result)


def _replay(root: Path, changes: DependencyChanges, result: EnvResult) -> EnvResult:
    if shutil.which("uv") is None:
        result.error = "uv is not installed: add your packages again with `uv add`"
        return result
    present = {project_name(r) for r in dependencies(root / PYPROJECT)}
    removed = [name for name in changes.removed if name in present]
    for args in (
        ["add", "--no-sync", *changes.added] if changes.added else None,
        ["remove", "--no-sync", *removed] if removed else None,
    ):
        if args is None:
            continue
        proc = _uv(args, root)
        if proc.returncode != 0:
            result.error = f"`uv {' '.join(args)}` failed:\n{proc.stderr.strip()}"
            break
    return result


def sync(root: Path) -> str:
    """``uv sync``; an error message, or "" when it went through."""
    if shutil.which("uv") is None:
        return "uv is not installed: run `uv sync` once it is"
    proc = _uv(["sync"], root)
    return "" if proc.returncode == 0 else proc.stderr.strip()


def shared_lock_matches(shared_env: Path, lock: Optional[str]) -> Optional[bool]:
    """Whether the shared environment was built from ``lock`` (a hash).

    The environment keeps a copy of the lock it was built from, as
    ``<env>/uv.lock``. None when that cannot be told.
    """
    copy = Path(shared_env) / LOCK
    if lock is None or not copy.is_file():
        return None
    return sha256_file(copy) == lock
