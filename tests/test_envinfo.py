"""Tests for cs_lab.environment: which Python, which course, which cache."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from cs_lab import envinfo as env_module
from cs_lab.cli import main
from cs_lab.envinfo import environment, environment_of, for_folder


def _facts(prefix, path=(), base_prefix="/usr", environ=None):
    return {
        "executable": f"{prefix}/bin/python",
        "version": "3.12.0",
        "prefix": str(prefix),
        "base_prefix": base_prefix,
        "path": [str(p) for p in path],
        "environ": environ or {},
    }


def _shared(root, lock="0123456789abcdef", links=("current",)):
    """A room environment, as `make shared-env` leaves it."""
    env = root / lock
    site = env / "lib" / "python3.12" / "site-packages"
    site.mkdir(parents=True)
    for marker in env_module.SHARED_MARKERS:
        (env / marker).touch()
    for link in links:
        (root / link).symlink_to(lock)
    return env, site


def _cache(root):
    for kind in ("models/org-a", "models/org-b", "tokenizers/org-a"):
        (root / "huggingface" / kind).mkdir(parents=True)
        (root / "huggingface" / kind / ".downloaded.ok").touch()
    (root / "huggingface" / "models" / "partial").mkdir()
    split = root / "huggingface" / "datasets" / "org-data" / "train"
    split.mkdir(parents=True)
    (split / ".downloaded.ok").touch()
    return root


def _describe(facts, course=None):
    return env_module._describe(facts, course)


def test_the_rooms_shared_environment(tmp_path):
    env, site = _shared(tmp_path / "llm-mind")
    cache = _cache(tmp_path / "cache")
    described = _describe(
        _facts(env, [site], environ={"CS_LAB_CACHE_PATH": str(cache)})
    )
    assert described.kind == "shared"
    assert described.shared.name == "llm-mind"
    assert described.shared.links == ["current"]
    assert described.cached == {"models": 2, "tokenizers": 1, "dataset splits": 1}
    text = str(described)
    assert "the room's shared environment (llm-mind, `current`" in text
    assert "2 models, 1 tokenizer, 1 dataset split" in text
    assert "fallback to the Hub" in text
    assert described.summary().startswith("Environment: room environment llm-mind")


def test_an_overlay_on_the_shared_environment(tmp_path):
    _, shared_site = _shared(tmp_path / "llm-mind")
    mine = tmp_path / "llm-extra"
    mine_site = mine / "lib" / "python3.12" / "site-packages"
    mine_site.mkdir(parents=True)
    described = _describe(_facts(mine, [mine_site, shared_site]))
    assert described.kind == "overlay"
    assert "on top of the room's shared one (llm-mind" in described.python_line()


def test_a_plain_venv_without_a_cache(tmp_path):
    described = _describe(_facts(tmp_path / ".venv"))
    assert described.kind == "venv" and described.cache is None
    assert "CS_LAB_CACHE_PATH is not set" in str(described)
    assert described.summary().endswith("no shared cache")


def test_enforce_and_legacy_names(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    described = _describe(
        _facts(
            tmp_path / ".venv",
            environ={"CACHED_HUB_PATH": str(cache), "CS_LAB_CACHE_ENFORCE": "1"},
        )
    )
    assert described.cache == cache and described.enforce
    assert "a miss is an error" in described.cache_line()


def test_a_missing_cache_says_so(tmp_path):
    described = _describe(
        _facts(tmp_path, environ={"CS_LAB_CACHE_PATH": str(tmp_path / "nope")})
    )
    assert "does not exist" in described.cache_line()


def _course_folder(root, mode="own", shared_env=""):
    (root / ".course").mkdir(parents=True)
    (root / ".course" / "config.toml").write_text(
        f'url = "u"\nname = "a-course"\nrelease = "r1"\nmode = "{mode}"\n'
        f'shared_env = "{shared_env}"\n'
    )
    return root


def test_this_process_and_its_course_folder(tmp_path):
    folder = _course_folder(tmp_path / "tp")
    (folder / "notebooks").mkdir()
    with patch.dict(os.environ, {"CS_LAB_CACHE_PATH": str(tmp_path)}, clear=True):
        described = environment(folder / "notebooks")
    assert described.python == sys.executable
    assert described.course.name == "a-course" and described.course.root == folder
    assert "a-course, release r1" in str(described)


def test_another_interpreter_is_asked(tmp_path):
    with patch.dict(os.environ, {"CS_LAB_CACHE_PATH": str(tmp_path)}):
        described = environment_of(Path(sys.executable), tmp_path)
    assert described.cache == tmp_path
    assert described.course is None


def test_a_folder_is_described_by_its_own_python(tmp_path):
    """Not by the one cs-lab runs on (uvx's): the shared env's, in shared mode."""
    env, _ = _shared(tmp_path / "llm-mind")
    (env / "bin").mkdir()
    (env / "bin" / "python").symlink_to(sys.executable)
    folder = _course_folder(
        tmp_path / "tp", "shared", str(tmp_path / "llm-mind/current")
    )
    asked = []

    def facts_of(python):
        asked.append(python)
        return _facts(env, environ={"CS_LAB_CACHE_PATH": "/room/cache"})

    with patch.object(env_module, "_facts_of", facts_of):
        described, note = for_folder(folder)
    assert note is None
    assert asked == [tmp_path / "llm-mind/current/bin/python"]
    assert described.kind == "shared" and described.cache == Path("/room/cache")


def test_a_folder_whose_environment_is_missing(tmp_path):
    folder = _course_folder(tmp_path / "tp")
    described, note = for_folder(folder)
    assert "is not there" in note and "uv sync" in note
    assert described.python == sys.executable


def test_cli_env(tmp_path, capsys):
    assert main(["env", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "not in a course folder" in out
    assert "Python " in out and "Cache " in out


@pytest.mark.skipif(os.name == "nt", reason="POSIX layout")
def test_cli_env_python(tmp_path, capsys):
    assert main(["env", "--python", sys.executable, str(tmp_path)]) == 0
    assert "not in a course folder" not in capsys.readouterr().out
