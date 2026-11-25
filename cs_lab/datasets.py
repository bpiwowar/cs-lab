"""A drop-in replacement for ``datasets``, reading the local cache first.

::

    from cs_lab.datasets import load_dataset

    train = load_dataset("stanfordnlp/imdb", split="train")
    sst2 = load_dataset("glue", "sst2")    # a DatasetDict of the cached splits

:func:`load_dataset` takes the arguments of ``datasets.load_dataset``: a Hub
dataset comes from the course's cache when it holds it (see
:func:`cs_lab.hf.load_hf_dataset`); local files, packaged builders
(``"csv"``, ``"json"``…, with ``data_files``) and streaming go straight to
``datasets``. Every other name is ``datasets``' own.
"""

from __future__ import annotations

import os
from typing import Any, Optional

import datasets as _datasets

from .hf import load_hf_dataset

#: Arguments that make a load something other than a Hub dataset's splits.
#: (``revision`` is not one: the cache keeps each revision apart.)
_NOT_FROM_HUB = ("data_dir", "data_files", "streaming")


def _from_hub(path: str, args: tuple, split: Any, kwargs: dict) -> bool:
    """Whether this load is a Hub dataset's splits, which the cache may hold."""
    if args or any(kwargs.get(key) for key in _NOT_FROM_HUB):
        return False
    if not isinstance(split, (str, type(None))) or os.path.exists(path):
        return False
    try:
        from datasets.packaged_modules import _PACKAGED_DATASETS_MODULES
    except ImportError:  # pragma: no cover - moved in some version
        _PACKAGED_DATASETS_MODULES = {}
    return path not in _PACKAGED_DATASETS_MODULES  # "csv", "json", …


def load_dataset(
    path: str, name: Optional[str] = None, *args, split: Any = None, **kwargs
) -> Any:
    """``datasets.load_dataset``, from the cache when it holds the splits."""
    if not _from_hub(path, args, split, kwargs):
        return _datasets.load_dataset(path, name, *args, split=split, **kwargs)
    return load_hf_dataset(path, name, split=split, **kwargs)


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    return getattr(_datasets, name)


def __dir__():
    return sorted(set(dir(_datasets)) | set(globals()))
