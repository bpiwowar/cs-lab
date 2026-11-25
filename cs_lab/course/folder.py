"""A student's course folder and what ``cs-lab course`` keeps in it.

    <folder>/
        notebooks/...            the student's working copy
        pyproject.toml, uv.lock  their environment (own-environment mode)
        .course/config.toml      where the releases come from, and choices
        .course/base/            the release as it was given to the student

``base/`` is what makes an update three-way: a file equal to its base copy was
not touched by the student, and can be replaced; one that differs is theirs.
"""

from __future__ import annotations

import shutil
import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Dict, Iterable, Optional

from .release import sha256_file

STATE_DIR = ".course"
CONFIG = "config.toml"
BASE = "base"

#: What to do with a file both the student and the course changed.
CHOICES = ("merge", "new", "replace", "keep")

#: The two ways a folder gets its Python environment.
MODES = ("own", "shared")


@dataclass
class Config:
    url: str
    name: str
    release: str = ""
    mode: str = "own"
    shared_env: str = ""
    #: The release's room the paths come from (``update`` follows it).
    room: str = ""
    #: In an own environment, the room's cache `course lab` points the
    #: notebooks at (CS_LAB_CACHE_PATH); empty: none.
    shared_cache: str = ""
    on_conflict: str = ""

    def dumps(self) -> str:
        lines = ["# Written by `cs-lab course`; on_conflict may be edited by hand."]
        for item in fields(self):
            value = getattr(self, item.name)
            escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
            lines.append(f'{item.name} = "{escaped}"')
        return "\n".join(lines) + "\n"

    @classmethod
    def loads(cls, text: str) -> "Config":
        data = tomllib.loads(text)
        known = {item.name for item in fields(cls)}
        return cls(**{k: str(v) for k, v in data.items() if k in known})


class FolderError(Exception):
    pass


class Folder:
    def __init__(self, root: Path):
        self.root = Path(root)

    @property
    def state(self) -> Path:
        return self.root / STATE_DIR

    @property
    def base(self) -> Path:
        return self.state / BASE

    @property
    def config_path(self) -> Path:
        return self.state / CONFIG

    def exists(self) -> bool:
        return self.config_path.is_file()

    def load(self) -> Config:
        if not self.exists():
            raise FolderError(
                f"{self.root} is not a course folder (no {STATE_DIR}/{CONFIG}): "
                "create one with `cs-lab course init`"
            )
        return Config.loads(self.config_path.read_text(encoding="utf-8"))

    def save(self, config: Config) -> None:
        self.state.mkdir(parents=True, exist_ok=True)
        # The base copy is bookkeeping, not the student's work.
        (self.state / ".gitignore").write_text(f"{BASE}/\n", encoding="utf-8")
        self.config_path.write_text(config.dumps(), encoding="utf-8")

    @staticmethod
    def digest(path: Path) -> Optional[str]:
        return sha256_file(path) if path.is_file() else None

    def local_digest(self, relative: str) -> Optional[str]:
        return self.digest(self.root / relative)

    def base_digest(self, relative: str) -> Optional[str]:
        return self.digest(self.base / relative)

    def base_files(self) -> Dict[str, str]:
        if not self.base.is_dir():
            return {}
        return {
            path.relative_to(self.base).as_posix(): sha256_file(path)
            for path in sorted(self.base.rglob("*"))
            if path.is_file()
        }

    def replace_base(self, release_dir: Path, paths: Iterable[str]) -> None:
        """Make ``release_dir`` (only ``paths`` of it) the new base."""
        staging = self.state / (BASE + ".new")
        shutil.rmtree(staging, ignore_errors=True)
        for relative in paths:
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(release_dir / relative, target)
        staging.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(self.base, ignore_errors=True)
        staging.rename(self.base)


def sibling(path: Path, tag: str) -> Path:
    """``attention.ipynb`` -> ``attention.<tag>.ipynb`` (no suffix: ``name.<tag>``)."""
    if path.suffix:
        return path.with_name(f"{path.stem}.{tag}{path.suffix}")
    return path.with_name(f"{path.name}.{tag}")


def free_sibling(path: Path, tag: str) -> Path:
    """Like :func:`sibling`, numbered when the name is taken."""
    candidate = sibling(path, tag)
    number = 2
    while candidate.exists():
        candidate = sibling(path, f"{tag}-{number}")
        number += 1
    return candidate
