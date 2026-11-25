"""Revisions in the cache: reading, locking, usage, gc, migration."""

import os
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

from cs_lab import cli, config
from cs_lab.config import ENV_LOCK, ENV_PATH, write_lock, write_ref
from cs_lab.hf import (
    load_hf_dataset,
    load_hf_model,
    make_hf_dataset_resource,
    make_hf_model_resource,
    make_hf_tokenizer_resource,
)
from cs_lab.revisions import (
    adopt,
    garbage,
    lock,
    migrate,
    register,
    registered,
    remove,
    usage,
)

OLD = "1" * 40
NEW = "2" * 40
OTHER = "3" * 40


@pytest.fixture(autouse=True)
def no_override():
    config.set_lock(None)
    yield
    config.set_lock(None)


def _complete(path):
    path.mkdir(parents=True, exist_ok=True)
    (path / ".downloaded.ok").touch()
    return path


def _model(root, name, revision):
    return _complete(root / "huggingface" / "models" / name / revision)


def _lock_file(path, **pins):
    """A lock pinning models/<name> at the given hashes."""
    write_lock(
        path,
        {
            f"models/{name}": {
                "id": name,
                "repo_type": "model",
                "requested": "main",
                "revision": sha,
            }
            for name, sha in pins.items()
        },
    )
    return path


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #


class TestLoading:
    def test_main_ref(self, tmp_path):
        _model(tmp_path, "m", OLD)
        new = _model(tmp_path, "m", NEW)
        write_ref(tmp_path / "huggingface" / "models" / "m", "main", NEW)
        cls = MagicMock()
        with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
            load_hf_model("m", cls)
        cls.from_pretrained.assert_called_once_with(str(new))

    def test_lock_wins_over_main(self, tmp_path):
        old = _model(tmp_path, "m", OLD)
        _model(tmp_path, "m", NEW)
        write_ref(tmp_path / "huggingface" / "models" / "m", "main", NEW)
        lock_path = _lock_file(tmp_path / "course.lock", m=OLD)
        cls = MagicMock()
        env = {ENV_PATH: str(tmp_path), ENV_LOCK: str(lock_path)}
        with patch.dict(os.environ, env, clear=True):
            load_hf_model("m", cls)
        cls.from_pretrained.assert_called_once_with(str(old))

    def test_locked_but_missing_goes_to_hub_at_that_revision(self, tmp_path):
        _model(tmp_path, "m", NEW)
        lock_path = _lock_file(tmp_path / "course.lock", m=OLD)
        cls = MagicMock()
        env = {ENV_PATH: str(tmp_path), ENV_LOCK: str(lock_path)}
        with patch.dict(os.environ, env, clear=True):
            load_hf_model("m", cls)
        cls.from_pretrained.assert_called_once_with(
            "m", revision=OLD, cache_dir=str(tmp_path / "huggingface")
        )

    def test_explicit_revision_by_ref(self, tmp_path):
        tagged = _model(tmp_path, "org-m", OLD)
        write_ref(tmp_path / "huggingface" / "models" / "org-m", "v1", OLD)
        cls = MagicMock()
        with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
            load_hf_model("org/m", cls, revision="v1")
        cls.from_pretrained.assert_called_once_with(str(tagged))

    def test_legacy_then_flat(self, tmp_path):
        flat = _complete(tmp_path / "huggingface" / "models" / "m")
        cls = MagicMock()
        with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
            load_hf_model("m", cls)
        cls.from_pretrained.assert_called_once_with(str(flat))
        legacy = _model(tmp_path, "m", "legacy")
        cls.reset_mock()
        with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
            load_hf_model("m", cls)
        cls.from_pretrained.assert_called_once_with(str(legacy))

    def test_unreadable_lock_is_ignored(self, tmp_path, caplog):
        flat = _complete(tmp_path / "huggingface" / "models" / "m")
        bad = tmp_path / "bad.lock"
        bad.write_text("{")
        cls = MagicMock()
        env = {ENV_PATH: str(tmp_path), ENV_LOCK: str(bad)}
        with patch.dict(os.environ, env, clear=True):
            load_hf_model("m", cls)
        cls.from_pretrained.assert_called_once_with(str(flat))
        assert "ignored" in caplog.text

    def test_dataset_split_at_locked_revision(self, tmp_path):
        entry = tmp_path / "huggingface" / "datasets" / "org-d"
        _complete(entry / NEW / "train")
        old = _complete(entry / OLD / "train")
        lock_path = tmp_path / "course.lock"
        write_lock(lock_path, {"datasets/org-d": {"id": "org/d", "revision": OLD}})
        datasets = MagicMock()
        env = {ENV_PATH: str(tmp_path), ENV_LOCK: str(lock_path)}
        with patch.dict(os.environ, env, clear=True):
            with patch.dict("sys.modules", {"datasets": datasets}):
                load_hf_dataset("org/d", split="train")
        datasets.load_from_disk.assert_called_once_with(str(old))

    def test_dataset_drop_in_keeps_revision_in_cache(self, tmp_path):
        pytest.importorskip("datasets")
        from cs_lab import datasets as dropin

        with patch.object(dropin, "load_hf_dataset") as load:
            dropin.load_dataset("org/d", split="train", revision=OLD)
        load.assert_called_once_with("org/d", None, split="train", revision=OLD)


# --------------------------------------------------------------------------- #
# Locking
# --------------------------------------------------------------------------- #


def _resources():
    return {
        "p1": [
            make_hf_model_resource("org/m"),
            make_hf_tokenizer_resource("org/m"),
            make_hf_dataset_resource("org/d", ["train"], revision="v2"),
        ],
        "p2": [make_hf_model_resource("org/m", optional=True)],
    }


class TestLock:
    def test_resolves_each_entry_once(self):
        resolver = MagicMock(return_value=NEW)
        locked = lock(_resources(), resolver=resolver)
        assert set(locked) == {"models/org-m", "tokenizers/org-m", "datasets/org-d"}
        assert locked["datasets/org-d"] == {
            "id": "org/d",
            "repo_type": "dataset",
            "requested": "v2",
            "revision": NEW,
        }
        assert resolver.call_count == 3

    def test_keeps_previous_pins_unless_upgraded(self):
        previous = lock(_resources(), resolver=lambda *a: OLD)
        kept = lock(_resources(), previous, resolver=lambda *a: NEW)
        assert {e["revision"] for e in kept.values()} == {OLD}
        one = lock(_resources(), previous, ["models/org-m"], resolver=lambda *a: NEW)
        assert one["models/org-m"]["revision"] == NEW
        assert one["tokenizers/org-m"]["revision"] == OLD
        every = lock(_resources(), previous, [], resolver=lambda *a: NEW)
        assert {e["revision"] for e in every.values()} == {NEW}

    def test_changed_request_is_resolved_again(self):
        previous = lock(_resources(), resolver=lambda *a: OLD)
        previous["datasets/org-d"]["requested"] = "v1"
        locked = lock(_resources(), previous, resolver=lambda *a: NEW)
        assert locked["datasets/org-d"]["revision"] == NEW
        assert locked["models/org-m"]["revision"] == OLD

    def test_conflicting_revisions(self):
        resources = {
            "a": [make_hf_model_resource("m", revision="v1")],
            "b": [make_hf_model_resource("m", revision="v2")],
        }
        with pytest.raises(ValueError, match="two revisions"):
            lock(resources, resolver=lambda *a: NEW)

    def test_download_follows_lock(self, tmp_path):
        lock_path = _lock_file(tmp_path / "course.lock", m=OLD)
        config.set_lock(lock_path)
        transformers = MagicMock()
        with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
            with patch.dict("sys.modules", {"transformers": transformers}):
                make_hf_model_resource("m").download()
        transformers.AutoModel.from_pretrained.assert_called_once_with(
            "m", revision=OLD
        )
        entry = tmp_path / "huggingface" / "models" / "m"
        assert (entry / OLD / ".downloaded.ok").exists()
        # a locked hash is not where main is
        assert not (entry / "refs" / "main").exists()


# --------------------------------------------------------------------------- #
# Usage, gc
# --------------------------------------------------------------------------- #


class TestUsage:
    def test_who_uses_what(self, tmp_path):
        _model(tmp_path, "a", OLD)
        _model(tmp_path, "a", NEW)
        _model(tmp_path, "b", OTHER)
        write_ref(tmp_path / "huggingface" / "models" / "b", "main", OTHER)
        register(tmp_path, "course1", _lock_file(tmp_path / "l1", a=OLD))
        register(tmp_path, "course2", _lock_file(tmp_path / "l2", a=OLD, c=NEW))
        uses, missing = usage(tmp_path)
        by = {(u.entry.key, u.revision): u for u in uses}
        assert by[("models/a", OLD)].locks == ["course1", "course2"]
        assert not by[("models/a", NEW)].used
        assert by[("models/b", OTHER)].refs == ["main"]
        assert missing == [("course2", "models/c", NEW)]

    def test_gc_removes_unused_revisions_only(self, tmp_path):
        _model(tmp_path, "a", OLD)
        unused = _model(tmp_path, "a", NEW)
        _model(tmp_path, "a", "legacy")
        gone = _model(tmp_path, "z", NEW)
        register(tmp_path, "course1", _lock_file(tmp_path / "l1", a=OLD))
        found = garbage(tmp_path)
        assert {u.path for u in found} == {unused, gone}
        for use in found:
            remove(use)
        assert not unused.exists()
        assert not gone.parent.exists()  # the emptied entry too
        assert (tmp_path / "huggingface" / "models" / "a" / OLD).exists()
        assert [u.revision for u in garbage(tmp_path, legacy=True)] == ["legacy"]

    def test_unregistered_lock_frees_its_revisions(self, tmp_path):
        _model(tmp_path, "a", OLD)
        register(tmp_path, "course1", _lock_file(tmp_path / "l1", a=OLD))
        assert not garbage(tmp_path)
        (tmp_path / "locks" / "course1.lock").unlink()
        assert registered(tmp_path) == {}
        assert [u.revision for u in garbage(tmp_path)] == [OLD]

    def test_register_rejects_bad_names(self, tmp_path):
        lock_path = _lock_file(tmp_path / "l1", a=OLD)
        for name in ("", "a/b", ".hidden"):
            with pytest.raises(ValueError):
                register(tmp_path, name, lock_path)


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #


class TestMigrate:
    def test_flat_entries_move_to_legacy(self, tmp_path):
        model = _complete(tmp_path / "huggingface" / "models" / "m")
        (model / "config.json").write_text("{}")
        dataset = tmp_path / "huggingface" / "datasets" / "d"
        _complete(dataset / "train")
        _model(tmp_path, "done", NEW)
        assert sorted(migrate(tmp_path)) == ["datasets/d", "models/m"]
        assert (model / "legacy" / "config.json").exists()
        assert (model / "legacy" / ".downloaded.ok").exists()
        assert not (model / ".downloaded.ok").exists()
        assert (dataset / "legacy" / "train" / ".downloaded.ok").exists()
        assert migrate(tmp_path) == []

    def test_adopt_when_main_was_at_the_locked_hash(self, tmp_path):
        legacy = _model(tmp_path, "m", "legacy")
        _model(tmp_path, "n", "legacy")
        locked = config.read_lock(_lock_file(tmp_path / "l", m=OLD, n=NEW))
        downloaded = datetime.fromtimestamp(
            (legacy / ".downloaded.ok").stat().st_mtime, tz=timezone.utc
        )
        commits = {  # newest first, as the Hub lists them
            "m": [(downloaded + timedelta(days=1), NEW), (downloaded, OLD)],
            "n": [(downloaded - timedelta(days=1), OLD)],
        }

        def history(repo_id, repo_type, when):
            return next(sha for date, sha in commits[repo_id] if date <= when)

        outcomes = dict(adopt(tmp_path, locked, history))
        assert outcomes["models/m"].startswith("adopted")
        assert outcomes["models/n"].endswith("kept")
        entry = tmp_path / "huggingface" / "models" / "m"
        assert (entry / OLD / ".downloaded.ok").exists()
        assert (entry / "refs" / "main").read_text().strip() == OLD
        assert (tmp_path / "huggingface" / "models" / "n" / "legacy").exists()


# --------------------------------------------------------------------------- #
# Command line
# --------------------------------------------------------------------------- #


def test_cli_gc_with_cache_option(tmp_path, capsys):
    _model(tmp_path, "a", OLD)
    unused = _model(tmp_path, "a", NEW)
    register(tmp_path, "course1", _lock_file(tmp_path / "l1", a=OLD))
    with patch.dict(os.environ, {}, clear=True):
        assert cli.main(["cache", "gc", "--cache", str(tmp_path)]) == 0
        assert unused.exists()
        assert "would remove models/a @ 222222222222" in capsys.readouterr().out
        assert cli.main(["cache", "gc", "--cache", str(tmp_path), "--yes"]) == 0
    assert not unused.exists()


def test_cli_gc_refuses_without_locks(tmp_path):
    _model(tmp_path, "a", OLD)
    with pytest.raises(SystemExit, match="no lock registered"):
        cli.main(["cache", "gc", "--cache", str(tmp_path)])


def test_cli_usage(tmp_path, capsys):
    _model(tmp_path, "a", OLD)
    register(tmp_path, "course1", _lock_file(tmp_path / "l1", a=OLD, b=NEW))
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        assert cli.main(["cache", "usage"]) == 0
    out = capsys.readouterr().out
    assert "models/a @ 111111111111  course1" in out
    assert "models/b @ 222222222222  (course1)" in out


def test_cli_lock_writes_file(tmp_path, capsys):
    resources = tmp_path / "resources.py"
    resources.write_text(
        "from cs_lab import make_hf_model_resource\n"
        "RESOURCES = {'p': [make_hf_model_resource('org/m')]}\n"
    )
    output = tmp_path / "resources.lock"
    with patch("cs_lab.hf.resolve_revision", return_value=NEW):
        argv = ["cache", "lock", "--from", str(resources), "-o", str(output)]
        assert cli.main(argv) == 0
    assert config.read_lock(output)["models/org-m"]["revision"] == NEW
    assert "1 new or changed" in capsys.readouterr().out
