# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 2: как устроен KV-кэш.

Идея
----
Внимание токена t зависит от всех предыдущих K/V. Без кэша на каждом новом
токене пришлось бы заново считать K/V всего префикса → суммарно O(n²).
KV-кэш сохраняет K/V всех прошедших токенов, поэтому генерация становится
O(n): один forward на токен, который читает ровно один новый столбец K/V.

Что измеряем/показываем:
  1. генерация без кэша (квадратично) vs с кэшем (линейно);
  2. структуру кэша: форма тензоров, число слоёв, память на токен;
  3. рост кэша по мере prefill;
  4. ограничение памяти: trim() и max_kv_size (скользящее окно).

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/02_kv_cache.py"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from common import (
    MODEL_ID,
    base_token_ids,
    bench,
    cache_nbytes,
    cache_offset,
    load_model,
    mb,
    note,
    section,
    subsection,
)


def generate_no_cache(model, ids, n: int) -> None:
    """«Наивная» генерация: на каждом шаге пересчитываем весь префикс."""
    seq = ids.tolist()
    for _ in range(n):
        logits = model(mx.array(seq)[None])
        mx.eval(logits)
        seq.append(int(mx.argmax(logits[0, -1])))


def generate_with_cache(model, ids, n: int) -> None:
    """Генерация с KV-кэшем: prefill один раз, потом по одному токену."""
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    tok = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    for _ in range(n):
        logits = model(tok, cache=cache)
        mx.eval(logits)
        tok = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)


def main() -> None:
    section("2. Как устроен KV-кэш")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    generate_with_cache(model, base_token_ids(tok, 8), 4)  # прогрев

    # ------------------------------------------------------- зачем нужен кэш
    subsection("A. Без кэша — O(n²), с кэшем — O(n)")
    prefill_len, n_new = 64, 24
    ids = base_token_ids(tok, prefill_len)
    t_no = bench(lambda: generate_no_cache(model, ids, n_new), repeats=2)
    t_yes = bench(lambda: generate_with_cache(model, ids, n_new), repeats=2)
    print(f"  prefill {prefill_len} ток. + {n_new} новых")
    print(f"  без кэша: {t_no * 1000:>9.1f} ms  ({t_no / (t_no + t_yes):.0%} времени)")
    print(f"  с кэшем : {t_yes * 1000:>9.1f} ms")
    print(f"  ускорение: {t_no / t_yes:.1f}x")
    note("Без кэша каждый шаг заново считает весь префикс → лишняя работа.")

    # ----------------------------------------------------------- структура
    subsection("B. Что лежит внутри кэша")
    ids = base_token_ids(tok, 128)
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)
    mx.eval(logits)

    layer0 = cache[0]
    k, v = layer0.keys_and_values()
    n_layers = len(cache)
    n_kv_heads, head_dim = k.shape[1], k.shape[3]
    itemsize = k.dtype.size
    print(f"  слоёв: {n_layers}")
    print(f"  на слой: K{k.shape}  V{v.shape}  dtype={k.dtype}")
    print(f"  KV-голов: {n_kv_heads}, head_dim: {head_dim}")
    print(f"  offset (токенов): {cache_offset(cache)}")

    per_token = 2 * n_layers * n_kv_heads * head_dim * itemsize
    print(f"  формула: 2 (K,V) × {n_layers} слоёв × {n_kv_heads} гол. × {head_dim} × {itemsize} б = {per_token} б/токен")
    print(f"  итого кэш: {mb(cache_nbytes(cache))} на 128 токенов")
    note(
        "Отсюда видно, почему KV-кэш дорогой: он линеен по контексту и "
        "не зависит от размера промпта. GQA (мало KV-голов) экономит в разы."
    )

    # -------------------------------------------------------- рост кэша
    subsection("C. Кэш растёт по мере prefill")
    cache = make_prompt_cache(model)
    total = 1024
    ids = base_token_ids(tok, total)
    print(f"  {'скормлено':>9} {'offset':>7} {'выделено':>9} {'размер':>10}")
    for end in range(64, total + 1, 64):
        logits = model(ids[end - 64:end][None], cache=cache)
        mx.eval(logits)
        layer0 = cache[0]
        allocated = layer0.keys.shape[2]
        print(
            f"  {end:>9} {cache_offset(cache):>7} "
            f"{allocated:>9} {mb(cache_nbytes(cache)):>10}"
        )
    note(
        "offset растёт каждые 64 токена, а выделенная память — ступеньками: "
        "KVCache.step=256, поэтому память резервируется блоками по 256 слотов."
    )

    # ---------------------------------------------------- ограничение памяти
    subsection("D. Ограничение памяти: trim() и скользящее окно")
    ids = base_token_ids(tok, 256)
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    offset_before = cache_offset(cache)
    trimmed = cache[0].trim(128)
    print(f"  trim(128): offset {offset_before} → {cache_offset(cache)} "
          f"(срезано {trimmed} токенов в слое 0)")
    note("trim срезает старые K/V, но требует, чтобы слой это поддерживал (is_trimmable).")

    # Скользящее окно: важно кормить по частям, иначе один большой prefill
    # разом заполнит кэш (внутри чанка trim не срабатывает).
    window = 64
    chunk = 32
    long_ids = base_token_ids(tok, 512)
    cache_win = make_prompt_cache(model, max_kv_size=window)
    cache_full = make_prompt_cache(model)
    for end in range(chunk, 513, chunk):
        mx.eval(model(long_ids[end - chunk:end][None], cache=cache_win))
        mx.eval(model(long_ids[end - chunk:end][None], cache=cache_full))
    print(f"  обработано 512 токенов по {chunk} за шаг:")
    print(f"    без окна:         память={mb(cache_nbytes(cache_full))}, "
          f"слотов в слое 0={cache_full[0].keys.shape[2]}")
    print(f"    max_kv_size={window}: память={mb(cache_nbytes(cache_win))}, "
          f"слотов в слое 0={cache_win[0].keys.shape[2]}, offset={cache_win[0].offset}")
    note(
        "Скользящее окно (RotatingKVCache) держит память константной: старые "
        "K/V перетираются, хотя offset (сколько токенов модель видела) растёт. "
        "Так работает sliding-window attention."
    )

    section("Вывод")
    note("KV-кэш убирает повторный prefill → генерация линейна по длине.")
    note("Плата — память O(контекст × слои × KV-головы), отсюда:")
    note("  · квантизация KV-кэша (эксп. 5);")
    note("  · блочное управление и переиспользование префиксов (эксп. 4);")
    note("  · вытеснение/окно для длинных контекстов.")


if __name__ == "__main__":
    main()
