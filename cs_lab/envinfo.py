"""What a notebook runs in: which Python, which course, which cache.

A student — or the teacher helping one — should not have to guess whether a
kernel runs the room's shared environment or a venv of their own, nor whether
the models come from the shared cache or are being downloaded again::

    $ cs-lab env
    Python    3.12.16  /…/envs/llm-mind/e301e589a7f15a24/bin/python
              the room's shared environment (llm-mind, `current`, lock e301e589a7f15a24)
    Course    llm-mind, release 2026-11-23-1 (~/tp-llm, shared environment)
    Cache     /…/cache/llm-mind (fallback to the Hub on a miss)
              12 models, 10 tokenizers, 1 processor, 9 dataset splits
    HF hub    ~/.cache/huggingface (what is not in the cache above)

:func:`environment` returns it, :func:`summary` in one line (what
:func:`cs_lab.hardware` adds to its note in a notebook).

``cs-lab env`` (and ``cs-lab course status``) describe the environment of
the course folder they are run in — the room's shared one, or the folder's
``.venv`` — and not the Python cs-lab itself runs on, which under ``uvx`` is
a throwaway one: they ask that interpreter (a few lines of standard library,
whatever cs-lab version it holds).

A shared environment is recognised by what jupytext-notebook-helper's
``make shared-env`` leaves in it: the lock it was built from, its
``constraints.txt`` and a ``.complete`` marker. A student's own environment
on top of one (a venv whose ``.pth`` adds the shared site-packages) is an
*overlay* of it.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ENV_ENFORCE, ENV_PATH, LEGACY_ENV

#: What ``make shared-env`` writes in each environment it builds.
SHARED_MARKERS = (".complete", "uv.lock", "constraints.txt")
#: The links a shared environment is published under.
SHARED_LINKS = ("current", "current-dev")
#: Course folder state (cs_lab.course.folder), read without importing it.
COURSE_CONFIG = Path(".course") / "config.toml"


@dataclass
class SharedEnv:
    path: Path  # the environment (a lock-hash directory)
    links: List[str] = field(default_factory=list)  # `current`… pointing to it

    @property
    def name(self) -> str:
        return self.path.parent.name

    @property
    def lock(self) -> str:
        return self.path.name

    def __str__(self) -> str:
        links = ", ".join(f"`{link}`" for link in self.links) or "no longer current"
        return f"{self.name}, {links}, lock {self.lock}"


#: What is asked of an interpreter, in its own terms: the standard library only.
_FACTS = """
def facts(names):
    import os, platform, sys
    return {
        "executable": sys.executable,
        "version": platform.python_version(),
        "prefix": sys.prefix,
        "base_prefix": sys.base_prefix,
        "path": sys.path,
        "environ": {name: os.environ[name] for name in names if name in os.environ},
    }
"""
#: The variables that say where things come from.
_VARIABLES = (
    ENV_PATH,
    *LEGACY_ENV[ENV_PATH],
    ENV_ENFORCE,
    *LEGACY_ENV[ENV_ENFORCE],
    "HF_HOME",
    "XDG_CACHE_HOME",
    "IR_DATASETS_HOME",
)


class ProbeError(Exception):
    """An interpreter that could not be asked about itself."""


def _facts_here() -> dict:
    namespace: dict = {}
    exec(_FACTS, namespace)  # noqa: S102 - our own constant
    return namespace["facts"](_VARIABLES)


def _facts_of(python: Path) -> dict:
    script = _FACTS + f"\nimport json; print(json.dumps(facts({list(_VARIABLES)!r})))"
    try:
        out = subprocess.run(
            [str(python), "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        ).stdout
        return json.loads(out)
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise ProbeError(f"cannot run {python}: {exc}") from exc


def _variable(environ: dict, name: str) -> Optional[str]:
    for candidate in (name, *LEGACY_ENV.get(name, ())):
        if candidate in environ:
            return environ[candidate]
    return None


@dataclass
class Course:
    root: Path
    name: str
    release: str
    mode: str
    shared_env: str = ""
    #: The room's cache `course lab` points an own environment at.
    shared_cache: str = ""

    @property
    def python(self) -> Path:
        """The interpreter the folder's notebooks run on."""
        env = Path(self.shared_env) if self.mode == "shared" else self.root / ".venv"
        return _interpreter(env)

    def __str__(self) -> str:
        mode = "shared environment" if self.mode == "shared" else "own environment"
        release = self.release or "no release"
        return f"{self.name}, release {release} ({_home(self.root)}, {mode})"


@dataclass
class Environment:
    python: str
    version: str
    prefix: Path
    #: "shared" (the room's), "overlay" (one's own on top of it), "venv", "system"
    kind: str
    shared: Optional[SharedEnv]
    course: Optional[Course]
    cache: Optional[Path]
    enforce: bool
    #: complete entries of the cache, per kind (models, tokenizers, ...)
    cached: Dict[str, int]
    hf_home: Path
    ir_datasets_home: Optional[str]

    def python_line(self) -> str:
        if self.kind == "shared":
            return f"the room's shared environment ({self.shared})"
        if self.kind == "overlay":
            return (
                f"your environment ({_home(self.prefix)}) on top of the room's "
                f"shared one ({self.shared})"
            )
        if self.kind == "venv":
            return f"a virtual environment ({_home(self.prefix)})"
        return "no virtual environment"

    def cache_line(self) -> str:
        if self.cache is None:
            return f"none ({ENV_PATH} is not set): models come from the HF hub cache"
        if not self.cache.is_dir():
            return f"{self.cache} — does not exist: everything is downloaded"
        mode = "a miss is an error" if self.enforce else "fallback to the Hub on a miss"
        return f"{_home(self.cache)} ({mode})"

    def cached_line(self) -> str:
        if not self.cached:
            return "empty"
        return ", ".join(
            f"{count} {kind if count > 1 else kind.rstrip('s')}"
            for kind, count in self.cached.items()
        )

    def __str__(self) -> str:
        lines = [
            f"Python    {self.version}  {self.python}",
            f"          {self.python_line()}",
            f"Course    {self.course or 'none here (no .course/ above this folder)'}",
            f"Cache     {self.cache_line()}",
        ]
        if self.cache is not None and self.cache.is_dir():
            lines.append(f"          {self.cached_line()}")
        hub = "what is not in the cache above" if self.cache else "downloads"
        lines.append(f"HF hub    {_home(self.hf_home)} ({hub})")
        if self.ir_datasets_home:
            lines.append(f"ir_datasets {_home(Path(self.ir_datasets_home))}")
        return "\n".join(lines)

    def summary(self) -> str:
        """One line, for the note a notebook shows."""
        python = {
            "shared": f"room environment {self.shared.name} ({self.shared.lock[:8]})"
            if self.shared
            else "",
            "overlay": f"your venv over room environment {self.shared.name}"
            if self.shared
            else "",
            "venv": f"venv {_home(self.prefix)}",
            "system": "system Python",
        }[self.kind]
        if self.cache is None:
            cache = "no shared cache"
        elif not self.cache.is_dir():
            cache = f"cache {self.cache} missing"
        else:
            cache = f"cache {_home(self.cache)}"
        return f"Environment: {python} · {cache}"


def _home(path: Path) -> str:
    """``path``, with the home directory written ``~``."""
    try:
        return "~/" + str(Path(path).relative_to(Path.home()))
    except ValueError:
        return str(path)


def _is_shared(prefix: Path) -> bool:
    return all((prefix / marker).exists() for marker in SHARED_MARKERS)


def _shared_env(prefix: Path) -> SharedEnv:
    prefix = prefix.resolve()
    links = []
    for link in SHARED_LINKS:
        try:
            if (prefix.parent / link).resolve() == prefix:
                links.append(link)
        except OSError:
            pass
    return SharedEnv(prefix, links)


def _interpreter(env: Path) -> Path:
    if os.name == "nt":
        return env / "Scripts" / "python.exe"
    return env / "bin" / "python"


def _site_prefix(entry: str) -> Optional[Path]:
    """The environment a ``…/lib/pythonX.Y/site-packages`` entry belongs to."""
    path = Path(entry)
    if path.name == "site-packages" and len(path.parents) > 2:
        return path.parents[2]
    return None


def _python(facts: dict) -> tuple:
    prefix = Path(facts["prefix"])
    if _is_shared(prefix):
        return "shared", _shared_env(prefix)
    for entry in facts["path"]:
        other = _site_prefix(entry)
        if other is not None and other != prefix and _is_shared(other):
            return "overlay", _shared_env(other)
    if facts["prefix"] != facts["base_prefix"]:
        return "venv", None
    return "system", None


def _course(start: Path) -> Optional[Course]:
    import tomllib

    for directory in (start, *start.parents):
        config = directory / COURSE_CONFIG
        if config.is_file():
            try:
                data = tomllib.loads(config.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return None
            return Course(
                directory,
                str(data.get("name", directory.name)),
                str(data.get("release", "")),
                str(data.get("mode", "own")),
                str(data.get("shared_env", "")),
                str(data.get("shared_cache", "")),
            )
    return None


def _cached(root: Path) -> Dict[str, int]:
    """Entries with a complete revision under ``<root>/huggingface/<kind>/``;
    for datasets, their splits (any revision, former layout included)."""
    from .revisions import entries

    counts: Dict[str, int] = {}
    for entry in entries(root):
        if not entry.revisions:
            continue
        kind = entry.key.split("/", 1)[0]
        if kind == "datasets":
            splits = {
                child.name
                for path in entry.revisions.values()
                for child in path.iterdir()
                if child.is_dir() and (child / ".downloaded.ok").exists()
            }
            label, count = "dataset splits", len(splits)
        else:
            label, count = kind, 1
        if count:
            counts[label] = counts.get(label, 0) + count
    custom = root / "custom"
    if custom.is_dir():
        count = sum(1 for _ in custom.rglob(".downloaded.ok"))
        if count:
            counts["other downloads"] = count
    return counts


def _hf_home(environ: dict) -> Path:
    if environ.get("HF_HOME"):
        return Path(environ["HF_HOME"])
    xdg = environ.get("XDG_CACHE_HOME")
    return (Path(xdg) if xdg else Path.home() / ".cache") / "huggingface"


def _describe(facts: dict, course: Optional[Course]) -> Environment:
    environ = facts["environ"]
    kind, shared = _python(facts)
    value = _variable(environ, ENV_PATH)
    if not value and course is not None and course.mode != "shared":
        # what `cs-lab course lab` sets for the notebooks
        value = course.shared_cache or None
    cache = Path(value) if value else None
    try:
        cached = _cached(cache) if cache is not None and cache.is_dir() else {}
    except OSError:
        cached = {}
    return Environment(
        python=facts["executable"],
        version=facts["version"],
        prefix=Path(facts["prefix"]),
        kind=kind,
        shared=shared,
        course=course,
        cache=cache,
        enforce=_variable(environ, ENV_ENFORCE) is not None,
        cached=cached,
        hf_home=_hf_home(environ),
        ir_datasets_home=environ.get("IR_DATASETS_HOME"),
    )


def find_course(directory: Optional[Path] = None) -> Optional[Course]:
    """The course folder ``directory`` (the current one by default) is in."""
    return _course(Path(directory or Path.cwd()).resolve())


def environment(directory: Optional[Path] = None) -> Environment:
    """This process's environment (what a notebook's kernel runs in)."""
    return _describe(_facts_here(), find_course(directory))


def environment_of(python: Path, directory: Optional[Path] = None) -> Environment:
    """Another interpreter's environment (:class:`ProbeError` if it cannot run)."""
    return _describe(_facts_of(Path(python)), find_course(directory))


def for_folder(
    directory: Optional[Path] = None, python: Optional[Path] = None
) -> Tuple[Environment, Optional[str]]:
    """The environment the notebooks of a folder run in, and a note when it is
    not that but this process's (no course folder, or its environment is
    missing): ``python`` overrides the choice."""
    if python is not None:
        return environment_of(python, directory), None
    course = find_course(directory)
    if course is None:
        return environment(directory), (
            "not in a course folder: this is the Python cs-lab itself runs on"
        )
    if not course.python.exists():
        return environment(directory), (
            f"the folder's environment ({course.python}) is not there"
            + (" — not on this machine?" if course.mode == "shared" else
               " — `uv sync` makes it")
            + ": this is the Python cs-lab itself runs on"
        )  # fmt: skip
    try:
        return environment_of(course.python, directory), None
    except ProbeError as exc:
        return environment(directory), f"{exc}: this is the Python cs-lab runs on"


def summary() -> str:
    """:func:`environment` in one line."""
    return environment().summary()


__all__ = [
    "Course",
    "Environment",
    "ProbeError",
    "SharedEnv",
    "environment",
    "environment_of",
    "find_course",
    "for_folder",
    "summary",
]
