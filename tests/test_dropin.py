"""Tests for the drop-in modules, cs_lab.transformers and cs_lab.datasets.

Neither library is installed for the tests: small stand-ins take their place.
"""

import importlib
import os
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

from cs_lab.config import ENV_ENFORCE, ENV_PATH, CacheMissError


def _cached(tmp_path, *parts):
    path = tmp_path.joinpath(*parts)
    path.mkdir(parents=True)
    (path / ".downloaded.ok").touch()
    return path


class _Loadable:
    """Records what ``from_pretrained`` was given, in an instance of ``cls``."""

    def __init__(self, source, args=(), kwargs=None):
        self.source, self.args, self.kwargs = source, args, kwargs or {}

    @classmethod
    def from_pretrained(cls, source, *args, **kwargs):
        return cls(source, args, kwargs)


def _fake_transformers():
    module = types.ModuleType("transformers")

    class PreTrainedModel(_Loadable):
        pass

    class GPT2LMHeadModel(PreTrainedModel):
        pass

    class PreTrainedTokenizerBase(_Loadable):
        pass

    class GPT2Tokenizer(PreTrainedTokenizerBase):
        pass

    class AutoTokenizer:
        @classmethod
        def from_pretrained(cls, source, *args, **kwargs):
            return GPT2Tokenizer(source, args, kwargs)

    class ProcessorMixin(_Loadable):
        pass

    class CLIPProcessor(ProcessorMixin):
        pass

    class AutoImageProcessor(_Loadable):
        pass

    class PretrainedConfig(_Loadable):
        pass

    class AutoConfig(PretrainedConfig):
        pass

    def pipeline(*args, **kwargs):
        return "a pipeline"

    for value in (
        PreTrainedModel, GPT2LMHeadModel, PreTrainedTokenizerBase, GPT2Tokenizer,
        AutoTokenizer, ProcessorMixin, CLIPProcessor, AutoImageProcessor,
        PretrainedConfig, AutoConfig, pipeline,
    ):  # fmt: skip
        setattr(module, value.__name__, value)
    module.TOKENIZER_CONSTANT = 42
    return module


@pytest.fixture
def T():
    """``cs_lab.transformers``, over the stand-in ``transformers``."""
    fake = _fake_transformers()
    with patch.dict("sys.modules", {"transformers": fake}):
        sys.modules.pop("cs_lab.transformers", None)
        module = importlib.import_module("cs_lab.transformers")
        module.real = fake
        yield module
    sys.modules.pop("cs_lab.transformers", None)


@pytest.fixture
def cache(tmp_path):
    with patch.dict(os.environ, {ENV_PATH: str(tmp_path)}, clear=True):
        yield tmp_path


# --------------------------------------------------------------------------- #
# cs_lab.transformers
# --------------------------------------------------------------------------- #


def test_model_from_cache_is_the_real_class(T, cache):
    local = _cached(cache, "huggingface", "models", "org-model")
    model = T.GPT2LMHeadModel.from_pretrained("org/model", dtype="auto")
    assert type(model) is T.real.GPT2LMHeadModel
    assert model.source == str(local)
    assert model.kwargs == {"dtype": "auto"}


def test_same_name_and_a_subclass(T):
    assert T.GPT2LMHeadModel.__name__ == "GPT2LMHeadModel"
    assert issubclass(T.GPT2LMHeadModel, T.real.GPT2LMHeadModel)
    assert T.GPT2LMHeadModel is T.GPT2LMHeadModel  # made once


@pytest.mark.parametrize(
    "name, folder",
    [
        ("AutoTokenizer", "tokenizers"),
        ("GPT2Tokenizer", "tokenizers"),
        ("CLIPProcessor", "processors"),
        ("AutoImageProcessor", "processors"),
        ("AutoConfig", "models"),
    ],
)
def test_cache_folder_follows_what_is_loaded(T, cache, name, folder):
    local = _cached(cache, "huggingface", folder, "org-model")
    assert getattr(T, name).from_pretrained("org/model").source == str(local)


def test_miss_falls_back_to_hub_inside_cache(T, cache, caplog):
    with caplog.at_level("WARNING", logger="cs_lab"):
        model = T.GPT2LMHeadModel.from_pretrained("org/model")
    assert model.source == "org/model"
    assert model.kwargs == {"cache_dir": str(cache / "huggingface")}
    assert "not found in local cache" in caplog.text


def test_miss_in_enforce_mode_raises(T, cache):
    with patch.dict(os.environ, {ENV_ENFORCE: "1"}):
        with pytest.raises(CacheMissError):
            T.AutoTokenizer.from_pretrained("org/model")


def test_without_a_cache_it_is_transformers(T):
    with patch.dict(os.environ, {}, clear=True):
        model = T.GPT2LMHeadModel.from_pretrained("org/model", "extra", a=1)
    assert (model.source, model.args, model.kwargs) == (
        "org/model",
        ("extra",),
        {"a": 1},
    )


def test_a_local_folder_is_loaded_as_given(T, cache, caplog):
    folder = cache / "my-finetuned"
    folder.mkdir()
    with caplog.at_level("WARNING", logger="cs_lab"):
        model = T.GPT2LMHeadModel.from_pretrained(str(folder))
    assert model.source == str(folder) and model.kwargs == {}
    assert caplog.text == ""


def test_a_subclass_keeps_the_cache_and_its_own_type(T, cache):
    local = _cached(cache, "huggingface", "models", "org-model")

    class Mine(T.GPT2LMHeadModel):
        pass

    model = Mine.from_pretrained("org/model")
    assert type(model) is Mine and model.source == str(local)


def test_other_names_are_transformers_own(T):
    assert T.pipeline is T.real.pipeline
    assert T.TOKENIZER_CONSTANT == 42
    with pytest.raises(AttributeError):
        T.NotInTransformers
    from cs_lab.transformers import pipeline  # noqa: F401 - `from` works too


# --------------------------------------------------------------------------- #
# cs_lab.datasets
# --------------------------------------------------------------------------- #


@pytest.fixture
def D():
    """``cs_lab.datasets``, over a stand-in ``datasets``."""
    fake = MagicMock()
    packaged = types.ModuleType("datasets.packaged_modules")
    packaged._PACKAGED_DATASETS_MODULES = {"csv": None, "json": None}
    modules = {"datasets": fake, "datasets.packaged_modules": packaged}
    with patch.dict("sys.modules", modules):
        sys.modules.pop("cs_lab.datasets", None)
        module = importlib.import_module("cs_lab.datasets")
        module.real = fake
        yield module
    sys.modules.pop("cs_lab.datasets", None)


def test_dataset_split_from_cache(D, cache):
    split = _cached(cache, "huggingface", "datasets", "org-data-cfg", "test")
    D.load_dataset("org/data", "cfg", split="test")
    D.real.load_from_disk.assert_called_once_with(str(split))
    D.real.load_dataset.assert_not_called()


def test_dataset_miss_goes_to_the_hub(D, cache):
    D.load_dataset("org/data", split="train")
    D.real.load_dataset.assert_called_once_with(
        "org/data", None, split="train", cache_dir=str(cache / "huggingface")
    )


@pytest.mark.parametrize(
    "args, kwargs",
    [
        (("csv",), {"data_files": "a.csv"}),  # a packaged builder
        (("org/data", None, "some/dir"), {}),  # data_dir, positionally
        (("org/data",), {"streaming": True}),
        (("org/data",), {"split": ["train", "test"]}),
    ],
)
def test_other_loads_go_straight_to_datasets(D, cache, args, kwargs):
    D.load_dataset(*args, **kwargs)
    D.real.load_from_disk.assert_not_called()
    (call,) = D.real.load_dataset.call_args_list
    assert "cache_dir" not in call.kwargs


def test_other_dataset_names_are_datasets_own(D):
    assert D.Dataset is D.real.Dataset
