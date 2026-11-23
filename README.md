# cs-lab

**What a teaching notebook needs from the machine it runs on: how much compute
to spend, what the hardware offers, and where the models come from.**

A practical hands the same file to a laptop, a lab machine and a Colab GPU. It
has to answer three questions, and `cs-lab` answers one each:

| | | |
|---|---|---|
| How much work should I do? | [`Profile`](#compute-profiles), a ladder the course declares | `NOTEBOOK_PROFILE` |
| What does this machine offer? | [`hardware()`](#the-machine-underneath) — device, dtype, which libraries work | `NOTEBOOK_BACKEND` |
| Where do the weights come from? | [the loaders](#supported-libraries) — a shared cache, the Hub on a miss | `CS_LAB_CACHE_PATH` |

They are one package because a notebook asks all three in the same breath
(`hardware(Profile)`, then `load_hf_model(Profile.pick(...), dtype=hw.dtype)`)
and because a student installs them together. It has **no heavy
dependency**: `torch`, `transformers`, `datasets`, `pyterrier` and
`datamaestro` are imported only by the functions that use them.

It also keeps a student's copy of the practicals up to date:
[`cs-lab course`](#a-folder-that-follows-the-course).

Building the notebooks themselves is a separate concern, and a separate
package:
[jupytext-notebook-helper](https://github.com/bpiwowar/jupytext-notebook-helper),
which students never install.

> Renamed from **cached-hub**, which said only what the third row does.
> Imports change from `cached_hub` to `cs_lab`; the CLI is `cs-lab`, with the
> cache commands under a `cache` group (`cached-hub list` -> `cs-lab cache
> list`), leaving room for the rest of what the package now covers; `CACHED_HUB_PATH` / `CACHED_HUB_ENFORCE` are still read, so a
> classroom machine that exports them keeps working.

## Why a shared cache

In a classroom, every student pulling `gpt2`, `SmolLM2` and `imdb` from the Hub
at the same minute is slow, fragile, and sometimes impossible (no home directory
quota, shaky proxy, offline lab). The usual answer is a shared read-only
directory pre-filled by the instructor. `cs-lab` makes that directory a
first-class thing:

- notebook code calls `load_hf_model("gpt2")` and gets the cached copy when it
  exists, the Hub otherwise, with a one-line note saying which;
- the instructor declares the resources once, and `cs-lab cache download` fills
  the cache (each shared resource once, optional ones on demand);
- an *enforce* mode turns any cache miss into an error, so you can check that
  every notebook runs fully from the cache before the session.

## Install

```sh
pip install cs-lab            # loaders only (bring your own transformers/datasets)
pip install "cs-lab[hf]"      # + transformers, datasets
```

## Supported libraries

| Library                | cs-lab function                | Equivalent to                                   | Cached under `$CS_LAB_CACHE_PATH`        | Pre-download with              |
|------------------------|------------------------------------|-------------------------------------------------|----------------------------------------|--------------------------------|
| transformers           | `load_hf_model(id, cls=AutoModel, **kw)`        | `cls.from_pretrained(id, **kw)`        | `huggingface/models/<id>/`             | `make_hf_model_resource`       |
| transformers           | `load_hf_tokenizer(id, cls=AutoTokenizer, **kw)`| `cls.from_pretrained(id, **kw)`        | `huggingface/tokenizers/<id>/`         | `make_hf_tokenizer_resource`   |
| transformers           | `load_hf_processor(id, cls=AutoProcessor, **kw)`| `cls.from_pretrained(id, **kw)`        | `huggingface/processors/<id>/`         | `make_hf_processor_resource`   |
| datasets               | `load_hf_dataset(id, name=None, split=None, **kw)` | `datasets.load_dataset(id, name, split=split, **kw)` | `huggingface/datasets/<id>[-<name>]/<split>/` | `make_hf_dataset_resource` |
| transformers           | `HFModel(id, tok_cls, model_cls, **kw)`         | lazy `.tokenizer` / `.model` via the two loaders above | as above                    | model + tokenizer resources    |
| pyterrier / ir-datasets| (use `pt.get_dataset` directly)                 | `pt.get_dataset(id)`                   | pyterrier's own home (`PYTERRIER_HOME`, `IR_DATASETS_HOME`) | `make_pyterrier_dataset_resource` |
| datamaestro            | (use `datamaestro.prepare_dataset` directly)    | `prepare_dataset(id)`                  | datamaestro's own store (`DATAMAESTRO_DIR`) | `make_datamaestro_resource` |

The loaders differ from their equivalents in one way only: with
`CS_LAB_CACHE_PATH` set, they first look for the resource in the cache layout
above, log where it came from, and on a miss forward to the equivalent call
with `cache_dir=$CS_LAB_CACHE_PATH/huggingface` added (unless you passed one), or
raise `CacheMissError` in enforce mode. PyTerrier and datamaestro manage their
own caches, so there is no loader for them: `cs-lab` only declares them as
resources so that `cs-lab cache download` fetches everything a course needs in
one go, and you keep calling those libraries as usual.

## In notebooks

```python
from cs_lab import load_hf_model, load_hf_tokenizer, load_hf_dataset, HFModel
from transformers import AutoModelForCausalLM

tokenizer = load_hf_tokenizer("HuggingFaceTB/SmolLM2-1.7B-Instruct")
model = load_hf_model(
    "HuggingFaceTB/SmolLM2-1.7B-Instruct", AutoModelForCausalLM, device_map="auto"
)
train = load_hf_dataset("imdb", split="train")
sst2 = load_hf_dataset("glue", name="sst2")  # DatasetDict of the cached splits

hf = HFModel("gpt2")  # lazy: nothing is loaded yet
hf.tokenizer, hf.model  # AutoTokenizer / AutoModel, loaded on first access
```

Each loader checks the local cache first, then falls back to the Hub with a
warning. Extra keyword arguments go to `from_pretrained` / `load_dataset`.

## Configuration

| Variable             | Effect                                                                 |
|----------------------|------------------------------------------------------------------------|
| `CS_LAB_CACHE_PATH`    | Root of the shared cache. Unset: the library does nothing (see below). |
| `CS_LAB_CACHE_ENFORCE` | If set (any value), a cache miss raises `CacheMissError` instead of falling back. |

Layout under the root (`org/name` becomes `org-name`):

```
$CS_LAB_CACHE_PATH/huggingface/models/<id>/                 model.save_pretrained()
$CS_LAB_CACHE_PATH/huggingface/tokenizers/<id>/
$CS_LAB_CACHE_PATH/huggingface/processors/<id>/
$CS_LAB_CACHE_PATH/huggingface/datasets/<id>[-<name>]/<split>/   dataset.save_to_disk()
$CS_LAB_CACHE_PATH/huggingface/                             HF cache_dir used for fallbacks
```

A directory is used only when it contains the marker `.downloaded.ok`, written
after a successful download, so a half-copied model is never picked up.

**Without `CS_LAB_CACHE_PATH`, `cs-lab` adds no caching of its own.**
`load_hf_model("gpt2", cls, **kw)` is then exactly `cls.from_pretrained("gpt2", **kw)`,
and `load_hf_dataset(...)` exactly `datasets.load_dataset(...)`: the usual
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
        make_hf_dataset_resource("imdb", ["train", "test"]),
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

`RESOURCES` above is only a naming convention. The option can be repeated. Resources are
identified by `(type, key)`, so a model shared by several practicals is
downloaded once. `HF_HUB_OFFLINE` is lifted for the duration of a download.

The same helpers are available from Python (`download_resources`,
`select_resources`, `format_resources`, `merge_resources`), and any object with
`resource_type`, `key`, `description`, `optional` and `download()` is a valid
resource (`FunctionalResource` wraps a plain function).

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
model = load_hf_model(MODEL, AutoModelForCausalLM, dtype=hw.dtype).to(hw.device)

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

What the scan understands: `load_hf_model` / `load_hf_tokenizer` /
`load_hf_processor` / `load_hf_dataset` / `HFModel` / `pt.get_dataset` /
`prepare_dataset`, with module-level string constants resolved
(`MODEL = "gpt2"` … `load_hf_model(MODEL, …)`). A constant rebound under a
guard — `if test_mode:` by default, `--guard NAME` for another one — becomes an
`optional=True` resource, since a small stand-in used while testing has no
business filling a classroom cache. A profile ladder (below) says the same
thing in one expression and is read the same way. Plain `load_dataset` and
`Class.from_pretrained` calls are reported as *bypasses*: they do not go through
the cache, so `check` never asks for them to be declared.

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

## A folder that follows the course

A course built with jupytext-notebook-helper publishes, next to its zip, a
`release.json` listing every file with its hash. `cs-lab course` turns that
into a folder a student can work in and still update:

```sh
uvx cs-lab course init https://example.org/course/practical/  # -> ./<course name>
cd <course name>
uvx cs-lab course update          # later, whenever the course says so
uvx cs-lab course status          # the release you have, what you changed
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

In a teaching room, `--shared-env PATH` uses an environment the teacher built
instead: the folder then has no environment files, and an update warns when
the shared environment was built from another release's lock (it keeps a copy
as `PATH/uv.lock`). If the folder is a git repository, an update commits the
student's work before, and the new release after.

`merge` needs nbdime: `cs-lab[course]`, or `uvx` fetches it.
