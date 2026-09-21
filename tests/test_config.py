"""Tests for cache configuration and layout helpers."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cs_lab import config
from cs_lab.config import ENV_ENFORCE, ENV_PATH, CacheMissError


def test_no_cache_when_unset():
    with patch.dict(os.environ, {}, clear=True):
        assert config.get_cache_path() is None
        assert config.get_hf_cache_dir() is None
        assert config.get_resource_path("huggingface", "models", "gpt2") is None
        assert not config.is_enforce_mode()


def test_cache_path_from_env():
    with patch.dict(os.environ, {ENV_PATH: "/some/path"}, clear=True):
        assert config.get_cache_path() == Path("/some/path")


def test_legacy_master_mind_vars_are_ignored():
    env = {"MASTER_MIND_DATA_PATH": "/legacy", "MASTER_MIND_DATA_ENFORCE": "1"}
    with patch.dict(os.environ, env, clear=True):
        assert config.get_cache_path() is None
        assert not config.is_enforce_mode()


def test_hf_cache_dir_is_created(tmp_path):
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        assert config.get_hf_cache_dir() == str(tmp_path / "huggingface")
        assert (tmp_path / "huggingface").is_dir()


@pytest.mark.parametrize(
    "resource_id,expected",
    [("org/model", "org-model"), ("org/sub/model", "org-sub-model"), ("m", "m")],
)
def test_sanitize_id(resource_id, expected):
    assert config.sanitize_id(resource_id) == expected


def test_resource_path_layout(tmp_path):
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        assert (
            config.get_resource_path("huggingface", "models", "org/m")
            == tmp_path / "huggingface" / "models" / "org-m"
        )
        assert (
            config.get_resource_path("huggingface", "datasets", "glue", "sst2")
            == tmp_path / "huggingface" / "datasets" / "glue-sst2"
        )


def test_download_marker(tmp_path):
    assert not config.is_download_complete(tmp_path)
    config.mark_download_complete(tmp_path)
    assert config.is_download_complete(tmp_path)
    assert (tmp_path / config.DOWNLOAD_MARKER).exists()


def test_warn_cache_miss_warns_with_cache(tmp_path, caplog):
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        with caplog.at_level("WARNING", logger="cs_lab"):
            config.warn_cache_miss("Model", "gpt2")
    assert "gpt2" in caplog.text and str(tmp_path) in caplog.text


def test_warn_cache_miss_raises_in_enforce_mode(tmp_path):
    env = {ENV_PATH: str(tmp_path), ENV_ENFORCE: "1"}
    with patch.dict(os.environ, env, clear=True):
        with pytest.raises(CacheMissError):
            config.warn_cache_miss("Model", "gpt2")


def test_warn_cache_miss_silent_without_cache(caplog):
    with patch.dict(os.environ, {ENV_ENFORCE: "1"}, clear=True):
        with caplog.at_level("WARNING", logger="cs_lab"):
            config.warn_cache_miss("Model", "gpt2")  # no cache: never raises
    assert caplog.text == ""


def test_hub_online_restores_offline_flag():
    with patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"}, clear=True):
        with config.hub_online():
            assert "HF_HUB_OFFLINE" not in os.environ
        assert os.environ["HF_HUB_OFFLINE"] == "1"


def test_describe():
    with patch.dict(os.environ, {}, clear=True):
        assert ENV_PATH in config.describe()
    with patch.dict(os.environ, {ENV_PATH: "/x", ENV_ENFORCE: "1"}, clear=True):
        assert "enforce" in config.describe()


def test_the_older_names_of_the_cache_variables_are_still_read(monkeypatch, caplog):
    """A lab machine exports them from a profile this code cannot edit."""
    from cs_lab import config as config_module

    for name in ("CS_LAB_CACHE_PATH", "CS_LAB_PATH", "CACHED_HUB_PATH"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(config_module, "_warned", set())

    monkeypatch.setenv("CACHED_HUB_PATH", "/srv/from-the-old-name")
    assert str(config.get_cache_path()) == "/srv/from-the-old-name"
    assert "CACHED_HUB_PATH" in caplog.text

    # the intermediate spelling wins over the oldest one
    monkeypatch.setenv("CS_LAB_PATH", "/srv/intermediate")
    assert str(config.get_cache_path()) == "/srv/intermediate"

    # and the current one over both
    monkeypatch.setenv("CS_LAB_CACHE_PATH", "/srv/current")
    assert str(config.get_cache_path()) == "/srv/current"
