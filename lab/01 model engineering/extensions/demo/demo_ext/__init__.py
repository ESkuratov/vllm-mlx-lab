"""Python-обёртка над собранным _ext."""
from . import _ext

add = _ext.add

__all__ = ["add"]
