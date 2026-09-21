"""Detect whether code is running inside a Jupyter kernel."""

from __future__ import annotations

__all__ = ["is_notebook"]

#: Shell classes that mean "a notebook": Jupyter's, and Colab's own.
_KERNEL_SHELLS = ("ZMQInteractiveShell", "Shell")


def is_notebook() -> bool:
    """Return True when running under an IPython kernel (Jupyter, Colab)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        return shell.__class__.__name__ in _KERNEL_SHELLS
    except (ImportError, AttributeError):
        return False
