"""Revisions in the cache: locking them, who uses which, and cleaning up.

A course *locks* its resources (``cs-lab cache lock``): each HuggingFace entry
it declares gets the commit hash it is to be read at, in a JSON file next to
its resource list — what ``uv.lock`` is to its packages. Downloading with that
lock fetches exactly those revisions; a shared environment points its Python
at it (``CS_LAB_CACHE_LOCK``), so that the course reads them back.

A cache shared by several courses keeps every revision any of them locked.
Each course *registers* its lock in the cache (``<root>/locks/<name>.lock``,
by ``cs-lab cache download --register NAME``): :func:`usage` then tells, for
each revision, which locks use it, and :func:`garbage` which revisions no lock
(nor ref) holds, for ``cs-lab cache gc`` to remove.

The former layout, without revisions, is moved aside by :func:`migrate`
(``<entry>/legacy/``) and, given a lock, renamed to the locked hash when it is
that revision: the Hub's ``main`` was at that commit when it was downloaded.
"""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from .config import (
    DOWNLOAD_MARKER,
    HF_KINDS,
    LEGACY,
    LOCKS_DIR,
    REFS_DIR,
    is_download_complete,
    is_sha,
    read_lock,
    write_ref,
)
from .resources import Resources, iter_all

logger = logging.getLogger("cs_lab")

PROVIDER_DIR = "huggingface"

#: ``(repo_id, repo_type, revision) -> sha``; the Hub by default.
Resolver = Callable[[str, str, Optional[str]], str]


# --------------------------------------------------------------------------- #
# Locking
# --------------------------------------------------------------------------- #


def lock_entries(resources: Resources) -> Dict[str, dict]:
    """The entries ``resources`` declare (optional ones included), by key."""
    entries: Dict[str, dict] = {}
    for resource in iter_all(resources):
        for entry in getattr(resource, "lock_entries", []):
            previous = entries.get(entry["key"])
            if previous and previous["revision"] != entry["revision"]:
                raise ValueError(
                    f"{entry['key']}: declared at two revisions "
                    f"({previous['revision']!r} and {entry['revision']!r})"
                )
            entries[entry["key"]] = entry
    return entries


def lock(
    resources: Resources,
    previous: Optional[Dict[str, dict]] = None,
    upgrade: Optional[Iterable[str]] = None,
    resolver: Optional[Resolver] = None,
) -> Dict[str, dict]:
    """The lock of ``resources``: each entry's commit hash.

    An entry already in ``previous`` keeps its hash, unless the revision it
    asks for changed or it is in ``upgrade`` (keys; an empty iterable: all).
    """
    if resolver is None:
        from .hf import resolve_revision as resolver
    previous = previous or {}
    upgrade_all = upgrade is not None and not list(upgrade)
    upgrade_keys = set(upgrade or ())
    locked: Dict[str, dict] = {}
    for key, entry in sorted(lock_entries(resources).items()):
        requested = entry["revision"] or "main"
        old = previous.get(key)
        if (
            old
            and old.get("requested") == requested
            and not upgrade_all
            and key not in upgrade_keys
        ):
            locked[key] = old
            continue
        sha = resolver(entry["id"], entry["repo_type"], entry["revision"])
        locked[key] = {
            "id": entry["id"],
            "repo_type": entry["repo_type"],
            "requested": requested,
            "revision": sha,
        }
        if old and old.get("revision") != sha:
            logger.info("%s: %s -> %s", key, old.get("revision"), sha)
    return locked


def register(root: Path, name: str, lock_path: Path) -> Path:
    """Copy a course's lock into the cache, as ``name``: :func:`usage` and
    :func:`garbage` count what it pins as used."""
    if not name or "/" in name or name.startswith("."):
        raise ValueError(f"invalid lock name {name!r}")
    read_lock(lock_path)  # a lock, not anything
    target = root / LOCKS_DIR / f"{name}.lock"
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}")
    shutil.copyfile(lock_path, tmp)
    os.replace(tmp, target)
    return target


def registered(root: Path) -> Dict[str, Dict[str, dict]]:
    """The locks registered in the cache, by name (unreadable ones skipped)."""
    locks: Dict[str, Dict[str, dict]] = {}
    folder = root / LOCKS_DIR
    if not folder.is_dir():
        return locks
    for path in sorted(folder.glob("*.lock")):
        try:
            locks[path.stem] = read_lock(path)
        except (OSError, ValueError) as exc:
            logger.warning("Lock %s skipped: %s", path, exc)
    return locks


# --------------------------------------------------------------------------- #
# What the cache holds
# --------------------------------------------------------------------------- #


@dataclass
class Entry:
    """One resource in the cache, and its revisions."""

    key: str  # models/org-name
    path: Path
    #: revision name (a hash, ``legacy``, or ``.`` for the former flat
    #: layout) -> its directory, complete ones only
    revisions: Dict[str, Path] = field(default_factory=dict)
    refs: Dict[str, str] = field(default_factory=dict)


def _complete(kind: str, path: Path) -> bool:
    if kind != "datasets":
        return is_download_complete(path)
    return any(
        child.is_dir() and is_download_complete(child) for child in path.iterdir()
    )


def _flat(kind: str, path: Path) -> bool:
    """Whether the entry itself holds content of the former layout."""
    if kind != "datasets":
        return is_download_complete(path)
    return any(
        child.is_dir()
        and child.name not in (REFS_DIR, LEGACY)
        and not is_sha(child.name)
        and is_download_complete(child)
        for child in path.iterdir()
    )


def entries(root: Path) -> List[Entry]:
    """Every entry of the cache, with its complete revisions and refs."""
    found = []
    base = root / PROVIDER_DIR
    for kind in HF_KINDS:
        folder = base / kind
        if not folder.is_dir():
            continue
        for path in sorted(folder.iterdir()):
            if not path.is_dir():
                continue
            entry = Entry(f"{kind}/{path.name}", path)
            if _flat(kind, path):
                entry.revisions["."] = path
            for child in sorted(path.iterdir()):
                if child.is_dir() and (is_sha(child.name) or child.name == LEGACY):
                    if _complete(kind, child):
                        entry.revisions[child.name] = child
            refs = path / REFS_DIR
            if refs.is_dir():
                for ref in sorted(refs.iterdir()):
                    if ref.is_file() and not ref.name.startswith("."):
                        entry.refs[ref.name] = ref.read_text().strip()
            found.append(entry)
    return found


@dataclass
class Use:
    """One revision of an entry, and what holds it."""

    entry: Entry
    revision: str
    path: Path
    locks: List[str]
    refs: List[str]

    @property
    def used(self) -> bool:
        return bool(self.locks or self.refs)


def usage(root: Path) -> Tuple[List[Use], List[Tuple[str, str, str]]]:
    """Each revision in the cache with the locks and refs holding it, and the
    ``(lock, key, hash)`` pins the cache does not hold (to download)."""
    locks = registered(root)
    pins: Dict[Tuple[str, str], List[str]] = {}
    for name, locked in locks.items():
        for key, entry in locked.items():
            pins.setdefault((key, entry["revision"]), []).append(name)

    uses = []
    held = set()
    for entry in entries(root):
        for revision, path in entry.revisions.items():
            refs = [ref for ref, sha in entry.refs.items() if sha == revision]
            names = pins.get((entry.key, revision), [])
            held.add((entry.key, revision))
            uses.append(Use(entry, revision, path, names, refs))
    missing = [
        (name, key, sha)
        for (key, sha), names in sorted(pins.items())
        if (key, sha) not in held
        for name in names
    ]
    return uses, missing


def garbage(root: Path, legacy: bool = False) -> List[Use]:
    """Revisions no registered lock pins and no ref points to.

    Content from before revisions were kept (``legacy``, or the former flat
    layout) is only included with ``legacy``: a cache with no lock reads it.
    """
    uses, _ = usage(root)
    return [use for use in uses if not use.used and (legacy or is_sha(use.revision))]


def remove(use: Use) -> None:
    """Delete an unused revision (and the refs pointing to it, if any)."""
    if use.revision == ".":
        for child in use.path.iterdir():
            if child.name in (REFS_DIR, LEGACY) or is_sha(child.name):
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    else:
        shutil.rmtree(use.path)
    entry = use.entry.path
    if entry.is_dir() and not any(child.name != REFS_DIR for child in entry.iterdir()):
        shutil.rmtree(entry)


# --------------------------------------------------------------------------- #
# From the former layout
# --------------------------------------------------------------------------- #


def migrate(root: Path) -> List[str]:
    """Move content of the former layout into ``<entry>/legacy/``.

    Renames only (cheap on a network file system). Returns the keys moved.
    """
    moved = []
    for entry in entries(root):
        if "." not in entry.revisions:
            continue
        target = entry.path / LEGACY
        if target.exists():
            logger.warning("%s: %s already exists, not migrated", entry.key, target)
            continue
        kind = entry.key.split("/", 1)[0]
        target.mkdir()
        for child in list(entry.path.iterdir()):
            if child.name in (REFS_DIR, LEGACY) or is_sha(child.name):
                continue
            if kind == "datasets" and not child.is_dir():
                continue
            os.rename(child, target / child.name)
        if not (target / DOWNLOAD_MARKER).exists() and kind != "datasets":
            logger.warning("%s: no completion marker after migration", entry.key)
        moved.append(entry.key)
    return moved


def _downloaded_at(kind: str, path: Path) -> datetime:
    """When a revision was downloaded: its completion marker's date (the
    earliest, for a dataset's splits)."""
    if kind == "datasets":
        markers = [p / DOWNLOAD_MARKER for p in path.iterdir() if p.is_dir()]
    else:
        markers = [path / DOWNLOAD_MARKER]
    times = [m.stat().st_mtime for m in markers if m.exists()]
    return datetime.fromtimestamp(min(times), tz=timezone.utc)


def main_at(repo_id: str, repo_type: str, when: datetime) -> Optional[str]:
    """The commit ``main`` was at, on the Hub, at ``when``."""
    from huggingface_hub import HfApi

    for commit in HfApi().list_repo_commits(repo_id, repo_type=repo_type):
        if commit.created_at <= when:
            return commit.commit_id
    return None


def adopt(
    root: Path,
    locked: Dict[str, dict],
    history: Callable[[str, str, datetime], Optional[str]] = main_at,
) -> List[Tuple[str, str]]:
    """Rename ``legacy/`` revisions to the hash the lock pins, when ``main``
    was at that hash on the day they were downloaded — no download needed.

    Returns ``(key, outcome)`` pairs, one per legacy revision considered.
    """
    outcomes = []
    for entry in entries(root):
        legacy = entry.revisions.get(LEGACY)
        pin = locked.get(entry.key)
        if legacy is None or pin is None:
            continue
        sha = pin["revision"]
        kind = entry.key.split("/", 1)[0]
        if sha in entry.revisions:
            outcomes.append((entry.key, f"{sha[:12]} already there"))
            continue
        try:
            was = history(pin["id"], pin["repo_type"], _downloaded_at(kind, legacy))
        except Exception as exc:  # the Hub, the network
            outcomes.append((entry.key, f"history unavailable: {exc}"))
            continue
        if was != sha:
            outcomes.append(
                (entry.key, f"legacy is {was and was[:12]}, not {sha[:12]}: kept")
            )
            continue
        os.rename(legacy, entry.path / sha)
        if pin.get("requested", "main") == "main" and "main" not in entry.refs:
            write_ref(entry.path, "main", sha)
        outcomes.append((entry.key, f"adopted as {sha[:12]}"))
    return outcomes
