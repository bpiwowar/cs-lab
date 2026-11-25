"""Cache location, layout and cache-miss policy.

Environment variables
---------------------
``CS_LAB_CACHE_PATH``
    Root of the shared local cache. When unset, this library adds no caching
    of its own: loaders forward to ``from_pretrained`` / ``load_dataset``
    unchanged, so HuggingFace's usual cache (``~/.cache/huggingface``) applies.
``CS_LAB_CACHE_ENFORCE``
    If set to any value, a cache miss raises :class:`CacheMissError` instead of
    falling back to the remote source. Useful to verify that everything a
    notebook needs is in the cache.
``CS_LAB_CACHE_LOCK``
    A resource lock (``cs-lab cache lock``): the revision each resource is
    read at, when the code does not name one. A shared environment sets it,
    so that a course reads the revisions it downloaded even when another
    course, sharing the cache, downloaded others.

Both were called something else before: ``CS_LAB_PATH`` / ``CS_LAB_ENFORCE``
until the cache became one part of this package rather than all of it, and
``CACHED_HUB_PATH`` / ``CACHED_HUB_ENFORCE`` before the rename. A classroom
machine exports them from a shell profile this code cannot edit, so they are
still read — once, with a word on stderr. A shared cache that silently stops
being used is an afternoon of downloads in a room of thirty.

On-disk layout
--------------
One directory per resource (an *entry*), one sub-directory per revision — the
Hub's commit hash — so that courses sharing a cache can each have their own::

    <root>/huggingface/models/<id>/<sha>/              model.save_pretrained()
    <root>/huggingface/tokenizers/<id>/<sha>/          tokenizer.save_pretrained()
    <root>/huggingface/processors/<id>/<sha>/          processor.save_pretrained()
    <root>/huggingface/datasets/<id>[-<name>]/<sha>/<split>/   dataset.save_to_disk()
    <root>/huggingface/<kind>/<entry>/refs/<ref>       the hash a ref (main) was
                                                       at, when downloaded
    <root>/huggingface/                                HF ``cache_dir`` for fallbacks
    <root>/locks/<name>.lock                           the locks of the courses
                                                       using the cache

``<id>`` is the HuggingFace identifier with ``/`` replaced by ``-``. A directory
counts as complete only when it contains the marker file ``.downloaded.ok``,
written after a successful download.

The former layout, without the revision level (``<entry>/`` itself complete,
or ``<entry>/<split>/`` for datasets), is still read; ``cs-lab cache migrate``
moves it to ``<entry>/legacy/``, then (``--adopt``) to its hash when the Hub
shows the repository unchanged since the download.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, List, Optional

logger = logging.getLogger("cs_lab")

ENV_PATH = "CS_LAB_CACHE_PATH"
ENV_ENFORCE = "CS_LAB_CACHE_ENFORCE"
ENV_LOCK = "CS_LAB_CACHE_LOCK"
#: Older spellings of each variable, tried in turn: ``CS_LAB_*`` from before
#: the cache commands were grouped, and ``CACHED_HUB_*`` from before the
#: package was renamed. A classroom machine exports them from a shell profile
#: this code cannot edit, and a shared cache that silently stops being used is
#: an afternoon of downloads in a room of thirty.
LEGACY_ENV = {
    ENV_PATH: ("CS_LAB_PATH", "CACHED_HUB_PATH"),
    ENV_ENFORCE: ("CS_LAB_ENFORCE", "CACHED_HUB_ENFORCE"),
}
_warned: set[str] = set()


def _environ(name: str) -> Optional[str]:
    """The variable's value, falling back to its older spellings."""
    value = os.environ.get(name)
    if value is not None:
        return value
    for legacy in LEGACY_ENV.get(name, ()):
        value = os.environ.get(legacy)
        if value is None:
            continue
        if legacy not in _warned:
            _warned.add(legacy)
            logger.warning("%s is the old name of %s; read it anyway", legacy, name)
        return value
    return None


#: Marker file written in a resource directory once its download completed.
DOWNLOAD_MARKER = ".downloaded.ok"


class CacheMissError(Exception):
    """Raised on a cache miss when enforce mode is on (``CS_LAB_CACHE_ENFORCE``)."""


def get_cache_path() -> Optional[Path]:
    """Return the cache root, or ``None`` when no cache is configured."""
    value = _environ(ENV_PATH)
    return Path(value) if value else None


def is_enforce_mode() -> bool:
    """Return True when a cache miss must raise instead of falling back."""
    return _environ(ENV_ENFORCE) is not None


def sanitize_id(resource_id: str) -> str:
    """Turn a hub identifier (``org/name``) into a directory name (``org-name``)."""
    return resource_id.replace("/", "-")


def get_resource_path(
    provider: str, resource_type: str, resource_id: str, name: Optional[str] = None
) -> Optional[Path]:
    """Directory where a resource lives in the cache, or ``None`` if no cache.

    Args:
        provider: e.g. ``"huggingface"``
        resource_type: e.g. ``"models"``, ``"tokenizers"``, ``"datasets"``
        resource_id: hub identifier, e.g. ``"distilbert-base-uncased"``
        name: optional configuration name (e.g. ``"sst2"`` for the glue dataset)
    """
    root = get_cache_path()
    if root is None:
        return None
    safe_id = sanitize_id(resource_id)
    if name:
        safe_id = f"{safe_id}-{name}"
    return root / provider / resource_type / safe_id


#: Kinds of HuggingFace entries, i.e. the directories under ``<root>/huggingface``.
HF_KINDS = ("models", "tokenizers", "processors", "datasets")
#: Sub-directory of an entry recording the hash each ref was at.
REFS_DIR = "refs"
#: Revision directory for content from before revisions were kept.
LEGACY = "legacy"
#: Directory of the cache root where the courses register their locks.
LOCKS_DIR = "locks"
LOCK_VERSION = 1

_SHA = re.compile(r"[0-9a-f]{40}")


def is_sha(revision: str) -> bool:
    """True for a full commit hash (40 hexadecimal digits)."""
    return bool(_SHA.fullmatch(revision))


def entry_key(resource_type: str, resource_id: str, name: Optional[str] = None) -> str:
    """The entry's path below ``<root>/huggingface``, e.g. ``models/org-m``: its
    key in a lock."""
    safe_id = sanitize_id(resource_id)
    if name:
        safe_id = f"{safe_id}-{name}"
    return f"{resource_type}/{safe_id}"


def _ref_file(entry: Path, ref: str) -> Path:
    return entry / REFS_DIR / sanitize_id(ref)


def read_ref(entry: Path, ref: str) -> Optional[str]:
    """The hash ``ref`` was at when last downloaded into ``entry``."""
    try:
        value = _ref_file(entry, ref).read_text().strip()
    except OSError:
        return None
    return value or None


def write_ref(entry: Path, ref: str, sha: str) -> None:
    """Record that ``ref`` was at ``sha`` (atomically: readers may be running)."""
    path = _ref_file(entry, ref)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(sha + "\n")
    os.replace(tmp, path)


# ---- locks ------------------------------------------------------------------

#: Lock set by the running program (``cs-lab cache download --lock``); wins
#: over CS_LAB_CACHE_LOCK.
_lock_override: Optional[Path] = None
_lock_cache: Dict[Path, tuple] = {}


def set_lock(path: Optional[Path]) -> None:
    """Use this lock (``None``: back to CS_LAB_CACHE_LOCK)."""
    global _lock_override
    _lock_override = Path(path) if path is not None else None


def get_lock_path() -> Optional[Path]:
    if _lock_override is not None:
        return _lock_override
    value = os.environ.get(ENV_LOCK)
    return Path(value) if value else None


def read_lock(path: Path) -> Dict[str, dict]:
    """A lock's entries: ``{entry key: {"id", "revision", ...}}``."""
    data = json.loads(Path(path).read_text())
    if data.get("version") != LOCK_VERSION:
        raise ValueError(f"{path}: unknown lock version {data.get('version')!r}")
    return dict(data.get("huggingface", {}))


def write_lock(path: Path, entries: Dict[str, dict]) -> None:
    data = {
        "version": LOCK_VERSION,
        "huggingface": {key: entries[key] for key in sorted(entries)},
    }
    Path(path).write_text(json.dumps(data, indent=2) + "\n")


def locked_revision(key: str) -> Optional[str]:
    """The hash the active lock pins the entry ``key`` at, if any.

    A lock that cannot be read is reported once and ignored: a notebook still
    loads (the cache's default revision), it does not fail on the lock.
    """
    path = get_lock_path()
    if path is None:
        return None
    try:
        mtime = path.stat().st_mtime
        cached = _lock_cache.get(path)
        if cached is None or cached[0] != mtime:
            cached = (mtime, read_lock(path))
            _lock_cache[path] = cached
    except (OSError, ValueError) as exc:
        if path not in _warned_locks:
            _warned_locks.add(path)
            logger.warning("Lock %s ignored: %s", path, exc)
        return None
    entry = cached[1].get(key)
    return entry.get("revision") if entry else None


_warned_locks: set = set()


def candidate_dirs(entry: Path, key: str, revision: Optional[str] = None) -> List[Path]:
    """Directories that may hold the entry, best first.

    A revision named by the code, or else pinned by the active lock, is the
    only candidate: reading another one would hide the difference. Otherwise:
    where ``main`` was last downloaded, then content from before revisions
    were kept (``legacy/``, then the entry itself).
    """
    pinned = revision or locked_revision(key)
    if pinned:
        sha = pinned if is_sha(pinned) else read_ref(entry, pinned)
        return [entry / sha] if sha else []
    candidates = []
    sha = read_ref(entry, "main")
    if sha:
        candidates.append(entry / sha)
    candidates += [entry / LEGACY, entry]
    return candidates


def get_hf_cache_dir() -> Optional[str]:
    """HuggingFace ``cache_dir`` inside the cache root (created on demand).

    Used when a resource is not in the local layout and has to be fetched from
    the Hub: it keeps the download inside the shared cache instead of
    ``~/.cache/huggingface``. Returns ``None`` when no cache is configured.
    """
    root = get_cache_path()
    if root is None:
        return None
    cache_dir = root / "huggingface"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return str(cache_dir)


def is_download_complete(path: Path) -> bool:
    """True when ``path`` holds a completed download (marker file present)."""
    return (path / DOWNLOAD_MARKER).exists()


def mark_download_complete(path: Path) -> None:
    """Write the completion marker in ``path``."""
    marker = path / DOWNLOAD_MARKER
    marker.touch()
    os.chmod(marker, 0o444)


def warn_cache_miss(
    resource_type: str, resource_id: str, cache_path: Optional[Path] = None
) -> None:
    """Report that ``resource_id`` is not in the local cache.

    With a configured cache this logs a warning, or raises
    :class:`CacheMissError` in enforce mode. Without a cache it only logs at
    debug level (nothing was expected locally).
    """
    if cache_path is None:
        cache_path = get_cache_path()
    if cache_path:
        message = (
            f"{resource_type} '{resource_id}' not found in local cache at {cache_path}"
        )
        if is_enforce_mode():
            raise CacheMissError(message)
        logger.warning(message)
    else:
        logger.debug(
            "%s '%s': no local cache configured, set %s to enable",
            resource_type,
            resource_id,
            ENV_PATH,
        )


@contextmanager
def hub_online() -> Iterator[None]:
    """Temporarily lift ``HF_HUB_OFFLINE`` so that downloads can reach the Hub."""
    previous = os.environ.pop("HF_HUB_OFFLINE", None)
    try:
        yield
    finally:
        if previous is not None:
            os.environ["HF_HUB_OFFLINE"] = previous


def describe() -> str:
    """One-line human description of the current cache configuration."""
    root = get_cache_path()
    if root is None:
        return f"no local cache ({ENV_PATH} is not set)"
    mode = "enforce" if is_enforce_mode() else "fallback to remote"
    return f"local cache at {root} ({mode})"
