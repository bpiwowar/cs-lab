"""Cache location, layout and cache-miss policy.

Environment variables
---------------------
``CS_LAB_PATH``
    Root of the shared local cache. When unset, this library adds no caching
    of its own: loaders forward to ``from_pretrained`` / ``load_dataset``
    unchanged, so HuggingFace's usual cache (``~/.cache/huggingface``) applies.
``CS_LAB_ENFORCE``
    If set to any value, a cache miss raises :class:`CacheMissError` instead of
    falling back to the remote source. Useful to verify that everything a
    notebook needs is in the cache.

The package used to be called ``cached-hub``, and a classroom machine may
still export ``CACHED_HUB_PATH`` / ``CACHED_HUB_ENFORCE`` from a shell profile
this code cannot edit. Those names are still read, once, with a word on
stderr — a shared cache that silently stops being used is an afternoon of
downloads in a room of thirty.

On-disk layout
--------------
Kept identical to the historical ``master_mind`` layout so existing caches
keep working::

    <root>/huggingface/models/<id>/                  model.save_pretrained()
    <root>/huggingface/tokenizers/<id>/              tokenizer.save_pretrained()
    <root>/huggingface/processors/<id>/              processor.save_pretrained()
    <root>/huggingface/datasets/<id>[-<name>]/<split>/   dataset.save_to_disk()
    <root>/huggingface/                              HF ``cache_dir`` for fallbacks

``<id>`` is the HuggingFace identifier with ``/`` replaced by ``-``. A directory
counts as complete only when it contains the marker file ``.downloaded.ok``,
written after a successful download.
"""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

logger = logging.getLogger("cs_lab")

ENV_PATH = "CS_LAB_PATH"
ENV_ENFORCE = "CS_LAB_ENFORCE"
#: What the same variables were called when the package was ``cached-hub``.
LEGACY_ENV = {ENV_PATH: "CACHED_HUB_PATH", ENV_ENFORCE: "CACHED_HUB_ENFORCE"}
_warned: set[str] = set()


def _environ(name: str) -> Optional[str]:
    """The variable's value, falling back to its ``CACHED_HUB_*`` spelling."""
    value = os.environ.get(name)
    if value is not None:
        return value
    legacy = LEGACY_ENV.get(name)
    if legacy is None:
        return None
    value = os.environ.get(legacy)
    if value is not None and legacy not in _warned:
        _warned.add(legacy)
        logger.warning("%s is the old name of %s; read it anyway", legacy, name)
    return value


#: Marker file written in a resource directory once its download completed.
DOWNLOAD_MARKER = ".downloaded.ok"


class CacheMissError(Exception):
    """Raised on a cache miss when enforce mode is on (``CS_LAB_ENFORCE``)."""


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
