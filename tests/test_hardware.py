"""Tests for hardware detection, and for how it meets a profile ladder."""

import builtins
import sys
import types

import pytest

from cs_lab import machine as hw_module
from cs_lab.machine import ENV_BACKEND, hardware


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv(ENV_BACKEND, raising=False)
    monkeypatch.delenv("NOTEBOOK_PROFILE", raising=False)
    hw_module._reset_for_tests()
    yield
    hw_module._reset_for_tests()


# ---------------------------------------------------------------------------
# Hardware
# ---------------------------------------------------------------------------


def test_hardware_without_torch_falls_back_to_cpu(monkeypatch):
    """This package installs without torch, and must import without it."""
    real_import = builtins.__import__

    def no_torch(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("no torch here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_torch)
    hw = hardware(refresh=True)
    assert hw.backend == "cpu"
    assert hw.total_memory_gb == 0.0


@pytest.mark.parametrize("backend", ["cuda", "mps", "cpu"])
def test_the_backend_can_be_forced(monkeypatch, backend):
    monkeypatch.setenv(ENV_BACKEND, backend)
    assert hardware(refresh=True).backend == backend


def test_an_unknown_backend_warns_and_is_detected_instead(monkeypatch, caplog):
    monkeypatch.setenv(ENV_BACKEND, "tpu")
    hardware(refresh=True)
    assert "tpu" in caplog.text


def test_hardware_is_cached():
    assert hardware() is hardware()


def test_capability_flags_are_false_off_cuda(monkeypatch):
    monkeypatch.setenv(ENV_BACKEND, "mps")
    hw = hardware(refresh=True)
    assert not hw.has_bitsandbytes
    assert not hw.has_vllm
    assert not hw.has_flash_attention
    assert not hw.supports_bf16


def test_dtypes_follow_the_backend(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    hw = hardware(refresh=True)
    assert hw.dtype is torch.float32
    assert hw.train_dtype is torch.float32

    monkeypatch.setenv(ENV_BACKEND, "mps")
    hw = hardware(refresh=True)
    assert hw.dtype is torch.float16
    # LoRA in pure fp16 is unstable, so training stays in fp32 off CUDA.
    assert hw.train_dtype is torch.float32


def test_synchronize_and_empty_cache_are_safe_on_cpu(monkeypatch):
    pytest.importorskip("torch")
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    hw = hardware(refresh=True)
    hw.synchronize()
    hw.empty_cache()
    assert hw.memory_used_gb() == 0.0


# ---------------------------------------------------------------------------
# Hardware meets a profile ladder
# ---------------------------------------------------------------------------


def _ladder():
    from cs_lab import Profile as base

    class Profile(base):
        FAST_TEST = 0
        SMALL = 1
        LOW_GPU = 2
        HIGH_GPU = 3

        @classmethod
        def detect(cls, hardware=None):
            if hardware is None or hardware.backend == "cpu":
                return cls.SMALL
            if hardware.backend == "cuda" and hardware.total_memory_gb >= 32:
                return cls.HIGH_GPU
            return cls.LOW_GPU

    Profile.reset()
    return Profile


def test_hardware_seeds_the_profile_from_the_machine(monkeypatch):
    Profile = _ladder()
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    hw = hardware(Profile, refresh=True)
    assert hw.profile is Profile.SMALL


def test_mps_is_a_gpu(monkeypatch):
    """An Apple Silicon laptop is not a machine without a GPU."""
    Profile = _ladder()
    monkeypatch.setenv(ENV_BACKEND, "mps")
    hw = hardware(Profile, refresh=True)
    assert hw.profile is Profile.LOW_GPU


def test_the_environment_still_wins_over_detection(monkeypatch):
    Profile = _ladder()
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    monkeypatch.setenv("NOTEBOOK_PROFILE", "fast-test")
    hw = hardware(Profile, refresh=True)
    assert hw.profile is Profile.FAST_TEST


def test_hardware_pick_delegates_to_the_ladder(monkeypatch):
    Profile = _ladder()
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    hw = hardware(Profile, refresh=True)
    assert hw.pick(200, fast_test=8, small=40) == 40


def test_pick_without_a_ladder_says_so(monkeypatch):
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    hw = hardware(refresh=True)
    with pytest.raises(RuntimeError, match="without a profile ladder"):
        hw.pick(200, small=40)


# ---------------------------------------------------------------------------
# In a notebook: the chooser (just the profile) plus whatever was registered
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_notebook(monkeypatch):
    """Fake being inside a kernel, for both `hardware()` and `Profile.select`."""
    import cs_lab.profile as profile_module

    monkeypatch.setattr(hw_module, "is_notebook", lambda: True)
    monkeypatch.setattr(profile_module, "is_notebook", lambda: True)


@pytest.fixture(autouse=True)
def no_state_notes(monkeypatch):
    """Start from no registered note: importing jupytext-notebook-helper in the
    same interpreter would otherwise add its own."""
    monkeypatch.setattr(hw_module, "_state_notes", [])


class _FakeWidget:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def observe(self, handler, names=None):
        pass


def _install_fake_ipywidgets(monkeypatch, displayed):
    """Fake ``ipywidgets`` (used by ``Profile.select``) and ``IPython.display``
    (used by both the chooser and the note this module shows)."""
    widgets = types.ModuleType("ipywidgets")
    widgets.ToggleButtons = _FakeWidget
    widgets.HTML = _FakeWidget
    widgets.VBox = lambda children: children

    display_module = types.ModuleType("IPython.display")
    display_module.display = displayed.append
    display_module.HTML = lambda text: ("HTML", text)

    monkeypatch.setitem(sys.modules, "ipywidgets", widgets)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)


def test_hardware_reports_the_active_rung_in_a_notebook(monkeypatch, fake_notebook):
    displayed = []
    _install_fake_ipywidgets(monkeypatch, displayed)
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    Profile = _ladder()

    hardware(Profile, refresh=True)

    # The chooser (profile only) and the note both show.
    assert len(displayed) == 2
    assert displayed[-1] == ("HTML", "<i>Profile: SMALL</i>")


def test_the_environment_shows_in_the_note_too(monkeypatch, fake_notebook):
    displayed = []
    _install_fake_ipywidgets(monkeypatch, displayed)
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    monkeypatch.setenv("NOTEBOOK_PROFILE", "fast-test")
    Profile = _ladder()

    hardware(Profile, refresh=True)

    assert displayed[-1] == ("HTML", "<i>Profile: FAST_TEST</i>")


def test_a_registered_note_is_appended(monkeypatch, fake_notebook):
    """What the authoring side adds — e.g. whether figures are drawn."""
    displayed = []
    _install_fake_ipywidgets(monkeypatch, displayed)
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    Profile = _ladder()

    hw_module.add_state_note(lambda: "images: off")
    hw_module.add_state_note(lambda: None)  # says nothing, shows nothing
    hardware(Profile, refresh=True)

    assert displayed[-1] == ("HTML", "<i>Profile: SMALL · images: off</i>")


def test_a_note_that_raises_never_stops_the_run(monkeypatch, fake_notebook):
    displayed = []
    _install_fake_ipywidgets(monkeypatch, displayed)
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    Profile = _ladder()

    def boom():
        raise RuntimeError("no")

    hw_module.add_state_note(boom)
    hardware(Profile, refresh=True)

    assert displayed[-1] == ("HTML", "<i>Profile: SMALL</i>")


def test_hardware_is_silent_outside_a_notebook(monkeypatch, capsys):
    Profile = _ladder()
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    hardware(Profile, refresh=True)
    assert capsys.readouterr().out == ""


def test_hardware_falls_back_to_a_print_without_ipython_display(
    monkeypatch, fake_notebook, capsys
):
    """No ``ipywidgets``/``IPython.display``: no chooser, but the note still
    reaches the (teacher's) console instead of being silently dropped."""
    real_import = builtins.__import__

    def no_widgets(name, *args, **kwargs):
        if name in ("ipywidgets", "IPython.display"):
            raise ImportError(f"no {name} here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_widgets)
    monkeypatch.delitem(sys.modules, "ipywidgets", raising=False)
    monkeypatch.delitem(sys.modules, "IPython.display", raising=False)
    monkeypatch.setenv(ENV_BACKEND, "cpu")
    Profile = _ladder()

    hardware(Profile, refresh=True)

    assert "Profile: SMALL" in capsys.readouterr().out
