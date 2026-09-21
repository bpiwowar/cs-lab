"""cs-lab: what a teaching notebook needs from the machine it runs on.

Three questions a practical has to answer, kept apart:

*How much compute to spend* — a :class:`Profile` ladder the course declares,
selected with ``NOTEBOOK_PROFILE`` or from the in-notebook chooser::

    from cs_lab import Profile as BaseProfile

    class Profile(BaseProfile):
        FAST_TEST = 0
        SMALL = 1
        LOW_GPU = 2

*What the machine offers* — device, dtype, and whether ``bitsandbytes`` or
``vLLM`` actually work here (:func:`hardware`, see :mod:`cs_lab.machine`)::

    hw = hardware(Profile)              # also shows the chooser, in a notebook
    model = load_hf_model(Profile.pick(...), AutoModelForCausalLM, dtype=hw.dtype)
    model.to(hw.device)

*Where the weights come from* — a shared local cache, falling back to the Hub,
so thirty laptops in a room do not each download the same model.

Notebook side::

    from cs_lab import load_hf_model, load_hf_tokenizer, load_hf_dataset, HFModel

Course side (declare what to pre-download)::

    from cs_lab import make_hf_model_resource, make_hf_dataset_resource
    RESOURCES = {"practical1": [make_hf_model_resource("gpt2"), ...]}

then ``cs-lab cache download --from mycourse.resources:RESOURCES``.

For course data that is not a HuggingFace model/dataset (a tarball published
on a course web server, say), use :func:`cached_download` instead -- same
``CS_LAB_CACHE_PATH`` root and cache-miss policy, with a plain relative-path
layout rather than the HuggingFace one.

``cs-lab cache scan sources/`` reads the loader calls back out of the sources, so
that a declaration can be written from them (``--emit``) and kept honest
(``cs-lab cache check sources/ --declaration mycourse/resources.py``).

Configure the cache root with ``CS_LAB_CACHE_PATH`` (see :mod:`cs_lab.config`).

:class:`Profile` is the base for a course's compute ladder (see
:mod:`cs_lab.profile`); the scanner reads ``Profile.pick(...)`` calls to
learn which of several models a notebook may load.
"""

from .config import (
    ENV_ENFORCE,
    ENV_PATH,
    CacheMissError,
    get_cache_path,
    is_enforce_mode,
)
from .datamaestro import make_datamaestro_resource
from .hf import (
    HFModel,
    load_hf_dataset,
    load_hf_model,
    load_hf_processor,
    load_hf_tokenizer,
    make_hf_dataset_resource,
    make_hf_model_resource,
    make_hf_processor_resource,
    make_hf_tokenizer_resource,
)
from .machine import ENV_BACKEND, Hardware, add_state_note, hardware
from .notebook import is_notebook
from .profile import ENV_PROFILE, Profile, current_ladder
from .pyterrier import make_pyterrier_dataset_resource
from .resources import (
    DownloadableResource,
    FunctionalResource,
    Resource,
    Resources,
    download_resources,
    format_resources,
    merge_resources,
    select_resources,
)
from .scan import (
    ScannedResource,
    compare,
    emit_section,
    parse_declaration,
    scan_paths,
)
from .web import cached_download

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - not installed from a build
    __version__ = "0.0.0+unknown"

__all__ = [
    "__version__",
    # config
    "ENV_PATH",
    "ENV_ENFORCE",
    "CacheMissError",
    "get_cache_path",
    "is_enforce_mode",
    # profiles
    "ENV_PROFILE",
    "Profile",
    "current_ladder",
    # the machine underneath
    "ENV_BACKEND",
    "Hardware",
    "hardware",
    "add_state_note",
    "is_notebook",
    # loading
    "HFModel",
    "load_hf_dataset",
    "load_hf_model",
    "load_hf_processor",
    "load_hf_tokenizer",
    # resources
    "Resource",
    "DownloadableResource",
    "FunctionalResource",
    "Resources",
    "download_resources",
    "format_resources",
    "merge_resources",
    "select_resources",
    "make_hf_dataset_resource",
    "make_hf_model_resource",
    "make_hf_processor_resource",
    "make_hf_tokenizer_resource",
    "make_pyterrier_dataset_resource",
    "make_datamaestro_resource",
    # scanning the sources that load them
    "ScannedResource",
    "compare",
    "emit_section",
    "parse_declaration",
    "scan_paths",
    # generic URL/archive caching
    "cached_download",
]
