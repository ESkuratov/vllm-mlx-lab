# SPDX-License-Identifier: Apache-2.0
"""
Общие утилиты для экспериментов в lab/.

Запускать из корня репозитория с активированным окружением:

    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/01_prefill_vs_decode.py"

Модель по умолчанию — Qwen2.5-Coder-3B-Instruct-4bit. Переопределить можно
переменной окружения LAB_MODEL:

    LAB_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit python "lab/02 inference engineering/01_prefill_vs_decode.py"
"""
from __future__ import annotations

import os
import statistics
import time

import mlx.core as mx

DEFAULT_MODEL = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
MODEL_ID = os.environ.get("LAB_MODEL", DEFAULT_MODEL)


# --------------------------------------------------------------------------- #
# Загрузка модели
# --------------------------------------------------------------------------- #


def load_model(model_id: str = MODEL_ID):
    """Загрузить модель и токенизатор через mlx-lm."""
    from mlx_lm import load

    return load(model_id)


def base_token_ids(tokenizer, length: int) -> mx.array:
    """Детерминированный массив из `length` реальных токенов (текст по кругу)."""
    text = (
        "The quick brown fox jumps over the lazy dog. "
        "Pack my box with five dozen liquor jugs. "
        "How vexingly quick daft zebras jump! "
    )
    ids = tokenizer.encode(text)
    reps = (length + len(ids) - 1) // len(ids)
    return mx.array((ids * reps)[:length])


# --------------------------------------------------------------------------- #
# Измерение
# --------------------------------------------------------------------------- #


def sync() -> None:
    """Дождаться завершения всех операций на GPU (MLX вычисляет лениво)."""
    mx.synchronize()


def timeit(fn):
    """Вернуть (результат, время_сек) для вызова fn()."""
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t0


def bench(fn, *, repeats: int = 5) -> float:
    """Медианное время выполнения fn (сек) по `repeats` прогонам."""
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def model_nbytes(model) -> int:
    """Размер весов модели в байтах (учитывает упакованную квантизацию)."""
    from mlx.utils import tree_flatten

    return sum(a.nbytes for _, a in tree_flatten(model.parameters()))


def cache_nbytes(cache) -> int:
    """Суммарный размер KV-кэша по всем слоям (байт)."""
    return sum(getattr(layer, "nbytes", 0) for layer in cache)


def cache_offset(cache) -> int:
    """Сколько токенов реально лежит в кэше (по первому слою)."""
    off = getattr(cache[0], "offset", 0)
    try:
        return int(off)
    except TypeError:
        return int(mx.max(off))


# --------------------------------------------------------------------------- #
# Вывод
# --------------------------------------------------------------------------- #


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def subsection(title: str) -> None:
    print()
    print(f"— {title}")


def note(text: str) -> None:
    print(f"  → {text}")


def gb(nbytes: int) -> str:
    return f"{nbytes / 1024**3:.3f} GB"


def mb(nbytes: int) -> str:
    return f"{nbytes / 1024**2:.1f} MB"
