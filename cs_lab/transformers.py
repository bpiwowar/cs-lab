"""A drop-in replacement for ``transformers``, reading the local cache first.

::

    from cs_lab.transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", dtype="auto")

Every name of ``transformers`` can be imported from here. A class with a
``from_pretrained`` comes back as a subclass of the same name whose
``from_pretrained`` looks in the course's cache first (see :mod:`cs_lab.hf`:
same layout as the ``make_hf_*_resource`` declarations, same fallback to the
Hub); anything else is ``transformers``' own.

What ``from_pretrained`` returns is the object ``transformers`` itself would
return — a ``GPT2LMHeadModel``, not a cs-lab type — so saving, ``peft`` and
``isinstance`` checks against ``transformers`` classes behave as usual. A
class that subclasses one of these keeps reading the cache, and gets
instances of itself.

Where a class reads from in the cache follows what it loads: tokenizers from
``tokenizers/``, processors (image processors and feature extractors too)
from ``processors/``, and models — and their configurations — from
``models/``.
"""

from __future__ import annotations

from typing import Any, Dict, Type

import transformers as _transformers

from .hf import load_pretrained

#: (kind, cache sub-folder), from the name of a class in the MRO.
_KINDS = (
    ("Tokenizer", ("tokenizer", "tokenizers")),
    ("Processor", ("processor", "processors")),
    ("ProcessingMixin", ("processor", "processors")),
    ("FeatureExtract", ("processor", "processors")),
    ("Config", ("config", "models")),
)

_wrapped: Dict[str, Any] = {}


def _kind(cls: Type) -> tuple:
    for base in cls.__mro__:
        for marker, kind in _KINDS:
            if marker in base.__name__:
                return kind
    return ("model", "models")


def cached_class(real: Type) -> Type:
    """``real`` with a ``from_pretrained`` that reads the cache first."""
    resource_type, subdir = _kind(real)

    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        # The class itself returns what `real` would; a subclass of it, as
        # with any class, instances of the subclass.
        target = real if cls is wrapper else super(wrapper, cls)
        return load_pretrained(
            resource_type, subdir, target, pretrained_model_name_or_path,
            *args, **kwargs,
        )  # fmt: skip

    from_pretrained.__doc__ = getattr(real.from_pretrained, "__doc__", None)
    wrapper = type(
        real.__name__,
        (real,),
        {
            "from_pretrained": classmethod(from_pretrained),
            "__module__": __name__,
            "__qualname__": real.__qualname__,
            "__doc__": real.__doc__,
            "__cs_lab_wraps__": real,
        },
    )
    return wrapper


def __getattr__(name: str) -> Any:
    if name.startswith("__"):
        raise AttributeError(name)
    if name not in _wrapped:
        value = getattr(_transformers, name)  # AttributeError: not there either
        if isinstance(value, type) and hasattr(value, "from_pretrained"):
            value = cached_class(value)
        _wrapped[name] = value
    return _wrapped[name]


def __dir__():
    return sorted(set(dir(_transformers)) | set(globals()))
