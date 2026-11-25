# cs-lab

**The runtime of a course's notebooks: the student's folder and its
environment, what the machine offers, how much compute to spend, and where the
models come from — from a cache shared by a room, or by several courses.**

A practical hands the same notebook to a laptop, a teaching-room machine and a
Colab GPU. `cs-lab` is what makes that work, for three kinds of users:

| Who | What for | Where |
|---|---|---|
| **Students** | get the practicals in a folder of their own, run them in the right environment, and update them without losing their work | [`cs-lab course`](#a-folder-that-follows-the-course), [`cs-lab env`](#which-environment-which-cache) |
| **Notebook authors** | ask the machine what it offers, size the work to it, load models and datasets through the cache | [`hardware()`](#the-machine-underneath), [`Profile`](#compute-profiles), [the loaders](#supported-libraries) |
| **Teachers running a room** | declare what a course needs, fill a shared cache, pin revisions, and clean up what no course uses any more | [`cs-lab cache`](#declaring-and-downloading-resources), [locks and `gc`](#revisions-locks-and-cleaning-up) |

Three settings carry it, all optional:

| | |
|---|---|
| `CS_LAB_CACHE_PATH` | the shared cache; unset, the loaders are plain `transformers` / `datasets` |
| `NOTEBOOK_PROFILE` | how much work a notebook does (`fast-test` … the course's largest rung) |
| `NOTEBOOK_BACKEND` | force a backend (`cpu`, …) on a machine with an accelerator |

It has **no heavy dependency**: `torch`, `transformers`, `datasets`,
`pyterrier` and `datamaestro` are imported only by the functions that use
them. Building the notebooks themselves is a separate package,
[jupytext-notebook-helper](https://github.com/bpiwowar/jupytext-notebook-helper),
which students never install; it also builds a course's shared environment in
a teaching room, and its `release.json`.

## Install

Students need nothing but [uv](https://docs.astral.sh/uv/): `uvx cs-lab …`
runs the latest version (`uvx cs-lab@latest …` when an older one is cached).
`cs-lab version` prints the one you have. Notebooks depend on it like on any
package:

```sh
pip install cs-lab            # loaders only (bring your own transformers/datasets)
pip install "cs-lab[hf]"      # + transformers, datasets
pip install "cs-lab[course]"  # + nbdime, for `course update`'s merges
```

## A folder that follows the course

A course built with jupytext-notebook-helper publishes, next to its zip, a
`release.json` listing every file with its hash. `cs-lab course` turns that
into a folder a student can work in and still update:

```sh
uvx cs-lab course init https://example.org/course/practical/  # -> ./<course name>
cd <course name>
uvx cs-lab course lab             # JupyterLab, from the folder's environment
uvx cs-lab course update          # later, whenever the course says so
uvx cs-lab course status          # the release you have, what you changed
uvx cs-lab env                    # which Python and cache the notebooks use
```

An update compares three versions of each file — the release the student was
given (kept in `.course/base/`), their copy, and the new release:

| | |
|---|---|
| the course did not change it | the student's copy stays as it is |
| the student did not touch it | replaced (added, or removed when the course dropped it) |
| both changed it | **the student chooses**, below |

| Choice | |
|---|---|
| `merge` | notebooks: a cell-by-cell merge ([nbdime](https://nbdime.readthedocs.io)), the student's side winning where both changed the same lines; their copy is kept as `name.mine.ipynb`. A cell of the course that could not be placed also leaves the course's version as `name.new.ipynb` |
| `new` | the student's file stays; the course's is put next to it, `name.new.ipynb` |
| `replace` | the course's version; the student's is kept as `name.mine-<date>.ipynb` |
| `keep` | nothing |

In a terminal it asks for each file (a capital letter answers for all the
rest, `d` shows a diff); `--on-conflict` answers once for all, and so does
`on_conflict` in `.course/config.toml`. Without a terminal or a choice, `new`:
it never touches the student's file. `--dry-run` says what would happen.

The environment is not merged as text — a text merge of `uv.lock` is a file uv
refuses. The course's `pyproject.toml` and `uv.lock` are taken, and the
dependencies the student added (or removed) are replayed with `uv add` /
`uv remove`, so uv keeps every version the course pinned and resolves only
theirs; then `uv sync`. `--reset-env` takes the course's environment as is.

If the folder is a git repository, an update commits the student's work
before, and the new release after.

`merge` needs nbdime: `cs-lab[course]`, or `uvx` fetches it.

### In a teaching room

The same commands, with no option. The release lists the *rooms* — sets of
machines sharing a disk — where the course installed things for everyone, as
absolute paths (`rooms` in `release.json`; jupytext-notebook-helper's
`make shared-env` builds one):

```json
"rooms": [
  {"name": "ppti", "shared_env": "/…/envs/llm-mind/current",
   "shared_cache": "/…/cache/cs-lab"},
  {"name": "lab", "shared_cache": "/data/cs-lab-cache"}
]
```

`init` takes the first room whose paths exist on the machine, and the folder
records its name; `update` follows that room's paths as later releases give
them.

- **A shared environment** (`shared_env`): the folder uses it, and has no
  environment files of its own — nothing to install, and a home quota spared.
  `--own-env` refuses it, `--shared-env PATH` names another. An update warns
  when it was built from another release's lock (it keeps a copy as
  `PATH/uv.lock`). Its Python already points at the room's cache.
- **A model cache** (`shared_cache`): a folder with its own environment there
  (`--own-env`, or a room with a cache only) still reads the models from it.
  `init` records it (`--no-shared-cache` refuses it), and `cs-lab course lab`
  starts Jupyter with `CS_LAB_CACHE_PATH` pointing at it — and
  `CS_LAB_CACHE_LOCK` at the folder's `resources.lock`, when the course ships
  one, so that the notebooks read the revisions the course pinned.

A release from before `rooms` names one room with top-level `shared_env` /
`shared_cache`, which is still read (and still written, for the first room,
for cs-lab before 2.2).

## Which environment, which cache

```console
$ uvx cs-lab env
Python    3.12.16  /…/envs/llm-mind/current-dev/bin/python
          the room's shared environment (llm-mind, `current-dev`, lock 2aa8421e9fd66685)
Course    llm-mind, release 94956bfc65d28b4e (~/tp-llm, shared environment)
Cache     ~/student-data/cache/cs-lab (fallback to the Hub on a miss)
          13 models, 11 tokenizers, 3 processors, 10 dataset splits
HF hub    ~/.cache/huggingface (what is not in the cache above)
ir_datasets ~/student-data/cache/cs-lab/ir_datasets
```

In a course folder, `cs-lab env` describes the environment its notebooks run
in — the room's shared one, or the folder's `.venv` (with the room's cache
when the folder reads it) — rather than the Python
`cs-lab` itself runs on (a throwaway one under `uvx`): it asks that
interpreter, with a few lines of standard library, so any cs-lab version there
will do. Elsewhere, or with `--python PATH`, it describes that Python.
`cs-lab course status` says the same in two lines; in code,
`cs_lab.environment()`.

## In notebooks

```python
from cs_lab.transformers import AutoModelForCausalLM, AutoTokenizer
from cs_lab.datasets import load_dataset

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, device_map="auto")
train = load_dataset("stanfordnlp/imdb", split="train")
sst2 = load_dataset("glue", "sst2")  # DatasetDict of the cached splits
```

Each load checks the local cache first, then falls back to the Hub with a
warning. Going back to plain HuggingFace is a matter of the import lines.

Each load says where it came from (the cache folder, or the Hub and where it
was kept), and the note `hardware()` shows in a notebook says which Python
the kernel runs — the room's shared environment, your own on top of it, a
venv — and which cache it reads.

## Supported libraries

`cs_lab.transformers` and `cs_lab.datasets` are drop-in replacements for
`transformers` and `datasets`: same names, same arguments, the same objects
returned.

| Library                | Load with                                         | Cached under `$CS_LAB_CACHE_PATH`        | Pre-download with              |
|------------------------|---------------------------------------------------|----------------------------------------|--------------------------------|
| transformers           | `cs_lab.transformers.<AnyModel>.from_pretrained(id, **kw)` | `huggingface/models/<id>/<sha>/`    | `make_hf_model_resource`       |
| transformers           | `cs_lab.transformers.<AnyTokenizer>.from_pretrained(id, **kw)` | `huggingface/tokenizers/<id>/<sha>/` | `make_hf_tokenizer_resource` |
| transformers           | `cs_lab.transformers.<AnyProcessor>.from_pretrained(id, **kw)` (image processors, feature extractors too) | `huggingface/processors/<id>/<sha>/` | `make_hf_processor_resource` |
| transformers           | `cs_lab.transformers.<AnyConfig>.from_pretrained(id)` | the model's folder                 | (comes with the model)         |
| datasets               | `cs_lab.datasets.load_dataset(id, name=None, split=None, **kw)` | `huggingface/datasets/<id>[-<name>]/<sha>/<split>/` | `make_hf_dataset_resource` |
| pyterrier / ir-datasets| (use `pt.get_dataset` directly)                   | pyterrier's own home (`PYTERRIER_HOME`, `IR_DATASETS_HOME`) | `make_pyterrier_dataset_resource` |
| datamaestro            | (use `datamaestro.prepare_dataset` directly)      | datamaestro's own store (`DATAMAESTRO_DIR`) | `make_datamaestro_resource` |

Every other name of the two modules (`pipeline`, `TrainingArguments`,
`Dataset`…) is the library's own. A class from `cs_lab.transformers` is a
subclass of the `transformers` one, of the same name, whose `from_pretrained`
reads the cache and then calls the `transformers` one: what comes back is a
plain `GPT2LMHeadModel`, so `save_pretrained`, `peft` and `isinstance` against
`transformers` classes behave as usual (a class of your own deriving from it
keeps the cache, and gets instances of itself). A local folder is loaded as
given. `cs_lab.datasets.load_dataset` goes straight to `datasets` for local
files, packaged builders (`"csv"`, `"json"`…), `data_files` / `data_dir`
and `streaming`; a `revision=` is read from the cache like the rest.

The functions these are built on are still there, and kept for existing code:

| cs-lab function                                  | Same as                                 |
|--------------------------------------------------|-----------------------------------------|
| `load_hf_model(id, cls=AutoModel, **kw)`         | `cls.from_pretrained(id, **kw)`         |
| `load_hf_tokenizer(id, cls=AutoTokenizer, **kw)` | `cls.from_pretrained(id, **kw)`         |
| `load_hf_processor(id, cls=AutoProcessor, **kw)` | `cls.from_pretrained(id, **kw)`         |
| `load_hf_dataset(id, name=None, split=None, **kw)` | `load_dataset(id, name, split=split, **kw)` |
| `HFModel(id, tok_cls, model_cls, **kw)`          | lazy `.tokenizer` / `.model`, loaded on first access |

The loaders differ from their equivalents in one way only: with
`CS_LAB_CACHE_PATH` set, they first look for the resource in the cache layout
above — at the revision asked for, or the one the course's lock pins (see
[Configuration](#configuration)) — log where it came from, and on a miss forward to the equivalent call
with `cache_dir=$CS_LAB_CACHE_PATH/huggingface` added (unless you passed one), or
raise `CacheMissError` in enforce mode. PyTerrier and datamaestro manage their
own caches, so there is no loader for them: `cs-lab` only declares them as
resources so that `cs-lab cache download` fetches everything a course needs in
one go, and you keep calling those libraries as usual.

## The machine underneath

`Profile` says how much work to do; `hardware()` says what to do it on. The two
kept being confused — a notebook writing `device.type == "cuda"` to mean "is
this machine big" when it meant "does `bitsandbytes` exist here", and an Apple
Silicon laptop with 128 GB of unified memory coming out on the wrong side of
both.

```python
from cs_lab import hardware
from mycourse.profiles import Profile

hw = hardware(Profile)        # in a notebook: also shows the profile chooser
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=hw.dtype).to(hw.device)

if hw.has_vllm:               # the library imports *and* the backend supports it
    ...
```

- `hw.device` / `hw.backend` (`cuda` | `mps` | `cpu`) / `hw.total_memory_gb`.
  **MPS counts as a GPU.**
- `hw.dtype` for inference (bf16 on CUDA, fp16 on MPS, fp32 on CPU) and
  `hw.train_dtype` for training (bf16 on CUDA, fp32 elsewhere — LoRA in pure
  fp16, without bf16's range, is unstable).
- `hw.has_bitsandbytes` / `hw.has_vllm` / `hw.has_flash_attention` /
  `hw.supports_bf16`: each is *the library imports* **and** *the backend
  supports it*.
- `hw.synchronize()` / `hw.empty_cache()` / `hw.memory_used_gb()`, so a
  notebook stops writing the per-device branches by hand.
- `NOTEBOOK_BACKEND=cpu` forces a backend, to reproduce a CPU-only run on a
  machine that has an accelerator.

Handing the ladder to `hardware()` is what connects the two: its `detect()`
maps this machine to a rung, recorded as the default — `NOTEBOOK_PROFILE` and
the chooser still win. In a notebook it also prints where things stand:

```
Profile: FAST_TEST — tiny budgets
```

`add_state_note(callable)` adds to that line. That is how
jupytext-notebook-helper appends `· images: off` when an author runs a source
with figures turned off; a student's notebook, which does not have that
package, simply shows the profile.

`torch` is imported lazily, so a course that installs this package for its
cache alone never pays for it.

## Compute profiles

How much compute a notebook should spend is a choice a reader makes — a smoke
test on a laptop, a full run on an A100 — and it is separate from what the
machine offers (CUDA or MPS, which dtype, whether `bitsandbytes` imports).
`Profile` covers the first question only.

The ladder is course material, not library material, so each course declares
its own rungs by subclassing:

```python
from cs_lab import Profile as BaseProfile


class Profile(BaseProfile):
    FAST_TEST = 0
    SMALL = 1
    LOW_GPU = 2
    HIGH_GPU = 3
```

Notebooks then size themselves with `pick`:

```python
MODEL_NAME = Profile.pick(
    fast_test="HuggingFaceTB/SmolLM2-135M-Instruct",
    small="Qwen/Qwen2.5-0.5B-Instruct",
    low_gpu="Qwen/Qwen2.5-1.5B-Instruct",
)
n_queries = Profile.pick(200, fast_test=8, small=40)
```

`pick` resolves when it is called, never at import, so changing profile and
re-running a cell does what it looks like it does. A rung with no value of its
own takes the nearest one below it — above, if there is nothing below — and a
positional default stands for every rung left unspecified. Rungs are ordered,
so `Profile.current() >= Profile.LOW_GPU` is the way to gate a section.

`NOTEBOOK_PROFILE` gives the starting rung by name (`fast-test`, `FAST_TEST`
and `fast gpu` are all read the same way); `Profile.set(...)` still wins
afterwards, and `Profile.select()` shows an `ipywidgets` chooser in a notebook.
Set `NOTEBOOK_PROFILE_WIDGET=0` to suppress it. Detecting the machine is
somebody else's job: whoever does it calls `Profile.set_detected(...)`, or a
course overrides `detect(hardware)` to map its own hardware to a rung.

Keeping `Profile` here, rather than in each course, is what lets `scan` read a
ladder without importing anything: it finds the `class X(Profile)` statement,
learns the rung names and their order from it, and resolves every `X.pick(...)`
call into the models it may load. Of those, the largest rung is required — it
is what the ladder resolves to when nothing selects a profile — and the smaller
ones come out `optional=True`.

## Why a shared cache

In a classroom, every student pulling `gpt2`, `SmolLM2` and `imdb` from the Hub
at the same minute is slow, fragile, and sometimes impossible (no home directory
quota, shaky proxy, offline lab). The usual answer is a shared read-only
directory pre-filled by the instructor. `cs-lab` makes that directory a
first-class thing:

- notebook code imports `transformers` and `datasets` from `cs_lab` instead —
  `from cs_lab.transformers import AutoModel` — and
  `AutoModel.from_pretrained("gpt2")` gets the cached copy when it exists, the Hub otherwise, with a one-line note saying which;
- the instructor declares the resources once, and `cs-lab cache download` fills
  the cache (each shared resource once, optional ones on demand);
- an *enforce* mode turns any cache miss into an error, so you can check that
  every notebook runs fully from the cache before the session.

## Configuration

| Variable             | Effect                                                                 |
|----------------------|------------------------------------------------------------------------|
| `CS_LAB_CACHE_PATH`    | Root of the shared cache. Unset: the library does nothing (see below). |
| `CS_LAB_CACHE_ENFORCE` | If set (any value), a cache miss raises `CacheMissError` instead of falling back. |
| `CS_LAB_CACHE_LOCK`    | A resource lock (`cs-lab cache lock`): the revision each resource is read at when the code names none. |

Layout under the root (`org/name` becomes `org-name`):

```
$CS_LAB_CACHE_PATH/huggingface/models/<id>/<sha>/           model.save_pretrained()
$CS_LAB_CACHE_PATH/huggingface/tokenizers/<id>/<sha>/
$CS_LAB_CACHE_PATH/huggingface/processors/<id>/<sha>/
$CS_LAB_CACHE_PATH/huggingface/datasets/<id>[-<name>]/<sha>/<split>/   dataset.save_to_disk()
$CS_LAB_CACHE_PATH/huggingface/<kind>/<id>/refs/main        the hash main was at, when downloaded
$CS_LAB_CACHE_PATH/huggingface/                             HF cache_dir used for fallbacks
$CS_LAB_CACHE_PATH/locks/<name>.lock                        the locks of the courses using it
```

A directory is used only when it contains the marker `.downloaded.ok`, written
after a successful download, so a half-copied model is never picked up.

Each revision (`<sha>`, the Hub's commit hash) has its own directory, so that
courses sharing a cache can each read the one they were written for. A loader
reads the `revision=` it is given, or else the one the lock in
`CS_LAB_CACHE_LOCK` pins, or else the one `main` was last downloaded at (see
[Revisions](#revisions-locks-and-cleaning-up)). A cache from before revisions
(`<id>/` itself complete) is still read; `cs-lab cache migrate` moves it.

**Without `CS_LAB_CACHE_PATH`, `cs-lab` adds no caching of its own.**
`cs_lab.transformers.X.from_pretrained(...)` is then exactly
`transformers.X.from_pretrained(...)`, and `cs_lab.datasets.load_dataset(...)`
exactly `datasets.load_dataset(...)`: the usual
HuggingFace cache (`~/.cache/huggingface`, `HF_HOME`) applies as it always does,
and `cs-lab cache download` merely warms it. Notebooks can therefore import from
`cs_lab` unconditionally and run unchanged on a laptop or on Colab; only the
classroom machines set the variable.

## Declaring and downloading resources

A course lists what it needs as `{section: [resources]}`:

```python
# mycourse/resources.py
from cs_lab import (
    make_hf_model_resource,
    make_hf_tokenizer_resource,
    make_hf_processor_resource,
    make_hf_dataset_resource,
    make_pyterrier_dataset_resource,
    make_datamaestro_resource,
)

RESOURCES = {
    "practical1": [
        make_hf_model_resource("gpt2", model_class="GPT2LMHeadModel"),
        make_hf_tokenizer_resource("gpt2", tokenizer_class="GPT2Tokenizer"),
        make_hf_dataset_resource("stanfordnlp/imdb", ["train", "test"]),
    ],
    "practical2": [
        make_hf_model_resource(
            "Qwen/Qwen2.5-7B-Instruct",
            model_class="AutoModelForCausalLM",
            optional=True,
        ),
        make_pyterrier_dataset_resource(
            "irds:lotte/technology/dev/search", "LoTTE technology"
        ),
    ],
}
```

then, on the machine that hosts the cache:

```sh
export CS_LAB_CACHE_PATH=/shared/cache
cs-lab cache info
cs-lab cache list     --from mycourse.resources:RESOURCES
cs-lab cache download --from mycourse.resources:RESOURCES               # everything but optional
cs-lab cache download --from mycourse.resources:RESOURCES --section practical2 --optional
cs-lab cache download --from mycourse.resources:RESOURCES --key gpt2
```

`--from MODULE:ATTR` imports `MODULE` and reads `ATTR` from it: a
`{section: [resources]}` mapping, or a zero-argument callable returning one
(dotted attributes such as `plugin.Course.resources` are followed). A `.py`
path works too and needs nothing on `sys.path`, which is the easy way to reach
a declaration that lives in a course's `src/`:

```sh
cs-lab cache list     --from src/mycourse/resources.py            # reads RESOURCES
cs-lab cache download --from src/mycourse/resources.py:get_resources
```

Each `make_hf_*_resource` also takes `revision=` (a branch, tag or commit
hash; `main` by default).

`RESOURCES` above is only a naming convention. The option can be repeated. Resources are
identified by `(type, key)`, so a model shared by several practicals is
downloaded once. `HF_HUB_OFFLINE` is lifted for the duration of a download.

The same helpers are available from Python (`download_resources`,
`select_resources`, `format_resources`, `merge_resources`), and any object with
`resource_type`, `key`, `description`, `optional` and `download()` is a valid
resource (`FunctionalResource` wraps a plain function).

## Revisions: locks, and cleaning up

A course pins its resources the way `uv.lock` pins its packages: `cs-lab cache
lock` resolves each HuggingFace entry to a commit hash, in a file to commit
next to the declaration; it keeps the existing pins, unless the revision a
resource asks for changed or `--upgrade` (all, or the keys given) says
otherwise. Downloading with `--lock` fetches exactly those revisions;
`--register NAME` records the lock in the cache (`<root>/locks/NAME.lock`),
which is how the cache knows who uses what:

```sh
cs-lab cache lock --from src/mycourse/resources.py -o resources.lock
cs-lab cache lock --from src/mycourse/resources.py -o resources.lock --upgrade models/gpt2

cs-lab cache download --from src/mycourse/resources.py \
    --lock resources.lock --register mycourse-2026 --cache /shared/cache
```

To read back the locked revisions, notebooks run with `CS_LAB_CACHE_LOCK`
pointing at the lock (a shared environment sets it, next to
`CS_LAB_CACHE_PATH`).

A cache shared by several courses then tells which lock holds which revision,
and removes what none holds — nor a `refs/` file (`main` as last downloaded):

```sh
cs-lab cache usage --cache /shared/cache            # each revision and its users; --size, --unused
cs-lab cache gc    --cache /shared/cache            # what would go
cs-lab cache gc    --cache /shared/cache --yes      # remove it
cs-lab cache unregister --cache /shared/cache mycourse-2025   # a course that ended
```

`--cache DIR` stands for `CS_LAB_CACHE_PATH` in every `cache` command. `gc`
refuses to run on a cache with no lock registered, and leaves content from
before revisions alone unless given `--legacy`.

A cache from before revisions is moved with `migrate`: each entry goes to
`<id>/legacy/`, still read when nothing pins a revision. `--adopt` then
renames it to the hash a registered lock (or `--lock FILE`) pins, when the
Hub's history shows `main` was at that commit on the day it was downloaded —
no download again:

```sh
cs-lab cache migrate --cache /shared/cache --adopt
```

## Keeping the declaration honest

The declaration is written by hand, so it drifts: a notebook gains a model, an
old one stops being loaded, and the classroom cache is wrong on the morning it
matters. `scan` reads the loader calls back out of the sources, and `check`
compares them with the declaration:

```sh
cs-lab cache scan  sources/                      # what the sources load
cs-lab cache scan  sources/ --emit practical2    # a declaration skeleton to fill in
cs-lab cache check sources/ --declaration mycourse/resources.py   # exit 1 on drift
```

A section is a source file name (`sources/02-generation.py` -> `02-generation`),
which is what `--section` takes. Both commands accept files or directories
(a directory contributes its top-level `*.py`, `_`-prefixed excluded), and
`--search-path DIR` lets an imported helper module be scanned as part of the
file that imports it — for course code split between a notebook and a library.

What the scan understands: `X.from_pretrained` and `load_dataset` on names
imported from `cs_lab.transformers` / `cs_lab.datasets` (`from … import X`,
`import cs_lab.transformers as T`), `load_hf_model` / `load_hf_tokenizer` /
`load_hf_processor` / `load_hf_dataset` / `HFModel` / `pt.get_dataset` /
`prepare_dataset`, with module-level string constants resolved
(`MODEL = "gpt2"` … `AutoModel.from_pretrained(MODEL)`). A configuration
(`AutoConfig`, `GenerationConfig`) comes with its model, and is not reported. A constant rebound under a
guard — `if test_mode:` by default, `--guard NAME` for another one — becomes an
`optional=True` resource, since a small stand-in used while testing has no
business filling a classroom cache. A profile ladder (below) says the same
thing in one expression and is read the same way. The same calls on
`transformers` / `datasets` themselves are reported as *bypasses*: they do not go
through the cache, so `check` never asks for them to be declared.

`check` reports as **errors** anything loaded but not declared, or declared but
never loaded, and as **warnings** a class or `optional` mismatch. Descriptions
and dataset *splits* are left alone: code that loads every split says nothing
about their names.

In a Makefile:

```make
check-resources:
	cs-lab cache check sources/ --declaration mycourse/resources.py
```

## Checking a cache before class

```sh
CS_LAB_CACHE_PATH=/shared/cache CS_LAB_CACHE_ENFORCE=1 python practical1.py
```

fails at the first resource that would have gone to the Hub.

## Former name

Renamed from **cached-hub**, which said only what the cache does. Imports
change from `cached_hub` to `cs_lab`; the CLI is `cs-lab`, with the cache
commands under a `cache` group (`cached-hub list` -> `cs-lab cache list`).
`CACHED_HUB_PATH` / `CACHED_HUB_ENFORCE` (and `CS_LAB_PATH` / `CS_LAB_ENFORCE`)
are still read, so a classroom machine that exports them keeps working.
