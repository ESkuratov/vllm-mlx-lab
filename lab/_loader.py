# SPDX-License-Identifier: Apache-2.0
"""Поиск и загрузка лабораторий независимо от раскладки папок.

Лаборатории разложены по темам (``01 model engineering/``,
``02 inference engineering/``, ``03 context engineering/``, …) и
пронумерованы **внутри своей папки**, поэтому номер сам по себе не уникален.
Некоторым лабораториям нужно загрузить другую (19 → 17, 20 → 13 и 16) или найти
файл по имени, чтобы посчитать хэш.

Ссылки задаются по имени файла без ``.py`` — оно уникально, — а поиск идёт по
всему ``lab/``: перенос между папками ничего не ломает.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent

#: Каталоги, в которых не ищем лаборатории.
SKIP_DIRS = {"extensions", "__pycache__"}


def _walk() -> list[Path]:
    out = []
    for path in sorted(LAB_ROOT.rglob("*.py")):
        rel = path.relative_to(LAB_ROOT)
        if any(
            part in SKIP_DIRS or part.startswith("__")
            for part in rel.parts[:-1]
        ):
            continue
        out.append(path)
    return out


@lru_cache(maxsize=None)
def _index() -> dict[str, Path]:
    """Ключ — имя файла без .py (уникально), плюс путь от папки lab/."""
    index: dict[str, Path] = {}
    for path in _walk():
        index[path.stem] = path
        index[path.relative_to(LAB_ROOT).with_suffix("").as_posix()] = path
    return index


def lab_path(name: str) -> Path:
    """Путь к лаборатории по имени файла ("13_load_qwen3").

    Также понимает форму с папкой ("01 model engineering/13_load_qwen3")
    и с расширением ".py".
    Номер без имени ("13") не принимается: нумерация теперь по папкам и
    неоднозначна.
    """
    key = Path(name).with_suffix("").as_posix()
    path = _index().get(key)
    if path is None:
        path = _index().get(key.rsplit("/", 1)[-1])
    if path is None:
        raise FileNotFoundError(
            f"лаборатория не найдена: {name!r}; укажите имя файла, например "
            f"'01 model engineering/13_load_qwen3'"
        )
    return path


def lab_file(name: str) -> Path:
    """Путь к произвольному файлу lab/ (например, к общему модулю)."""
    key = Path(name).with_suffix("").as_posix()
    path = _index().get(key) or _index().get(key.rsplit("/", 1)[-1])
    if path is not None:
        return path
    for candidate in _walk():
        if candidate.name == name:
            return candidate
    raise FileNotFoundError(f"файл не найден в lab/: {name!r}")


def load_lab(name: str, module_name: str | None = None):
    """Импортировать лабораторию как модуль (кэшируется в sys.modules)."""
    path = lab_path(name)
    mod_name = module_name or f"lab_{path.stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"не удалось загрузить {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def all_labs() -> list[Path]:
    """Все скрипты-лаборатории (файлы, начинающиеся с номера)."""
    return [p for p in _walk() if p.name[0].isdigit()]
