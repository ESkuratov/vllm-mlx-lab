# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 1: чем prefill отличается от decode.

Идея
----
Инференс LLM состоит из двух принципиально разных фаз:

* **prefill** — прогон всего промпта через модель за один forward. Все токены
  обрабатываются параллельно, доминируют матричные умножения → фаза упирается
  в *compute*.
* **decode** — генерация по одному токену за шаг. На каждом шаге модель читает
  все свои веса, но полезной работы мало (один токен) → фаза упирается в
  *пропускную способность памяти*.

Что измеряем:
  1. как масштабируется время prefill с длиной промпта;
  2. как меняется время одного decode-шага при росте контекста;
  3. сравниваем достигнутый decode tokens/s с теоретическим пределом
     bandwidth / размер_весов.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/01_prefill_vs_decode.py"
"""
from __future__ import annotations

import statistics
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
    gb,
    load_model,
    model_nbytes,
    note,
    section,
    subsection,
)


def prefill(model, ids):
    """Обработать весь промпт за один forward с чистым кэшем."""
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    return logits, cache


def decode_steps(model, cache, logits, n):
    """Greedy-декод n токенов, вернуть время каждого шага (сек)."""
    tok = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    steps = []
    for _ in range(n):
        t0 = time.perf_counter()
        logits = model(tok, cache=cache)
        mx.eval(logits)
        steps.append(time.perf_counter() - t0)
        tok = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    return steps


def main() -> None:
    section("1. Prefill vs decode")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()

    # Прогрев — первый запуск всегда медленнее (компиляция ядер Metal и т.п.).
    prefill(model, base_token_ids(tok, 32))
    prefill(model, base_token_ids(tok, 256))

    # ---------------------------------------------------------------- prefill
    subsection("A. Время prefill растёт линейно с длиной промпта")
    print(f"  {'длина':>8} {'время':>12} {'tokens/s':>12}")
    for length in (16, 32, 64, 128, 256, 512, 1024, 2048):
        ids = base_token_ids(tok, length)
        t = bench(lambda: prefill(model, ids), repeats=3)
        print(f"  {length:>8} {t * 1000:>10.1f} ms {length / t:>12.0f}")
    note("Стоимость на токен почти постоянна → prefill считается параллельно.")

    # ----------------------------------------------------------------- decode
    subsection("B. Время decode-шага почти не зависит от длины контекста")
    print(f"  {'контекст':>9} {'шаг(медиана)':>14} {'tokens/s':>10} {'шагов':>7}")
    decode_tps = None
    for length in (16, 128, 512, 1024, 2048):
        ids = base_token_ids(tok, length)
        logits, cache = prefill(model, ids)
        steps = decode_steps(model, cache, logits, n=16)
        med = statistics.median(steps[4:])  # отбрасываем разогрев GPU
        if length == 128:
            decode_tps = 1.0 / med
        print(f"  {length:>9} {med * 1000:>12.2f} ms {1 / med:>10.1f} {16:>7}")
    note("Decode читает все веса на каждый токен, поэтому время шага ~константа.")

    # -------------------------------------------------------- сравнение фаз
    subsection("C. Prefill против decode в терминах tokens/s")
    ids = base_token_ids(tok, 128)
    t_prefill = bench(lambda: prefill(model, ids), repeats=3)
    print(f"  prefill (128 токенов): {128 / t_prefill:>8.0f} tokens/s")
    print(f"  decode  (1 токен/шаг): {decode_tps:>8.0f} tokens/s")
    note("Prefill в разы (десятки раз) быстрее на токен — но за один проход.")

    # ------------------------------------------------------- упор в bandwidth
    subsection("D. Decode упирается в пропускную способность памяти")
    weights = model_nbytes(model)
    print(f"  размер весов модели: {gb(weights)}")
    try:
        from vllm_mlx.optimizations import benchmark_memory_bandwidth, detect_hardware

        hw = detect_hardware()
        print(f"  чип: {hw.chip_name}, bandwidth (спека): {hw.memory_bandwidth_gbs} GB/s")
        print(f"  bandwidth (замер): {benchmark_memory_bandwidth()}")

        theoretical_tps = hw.memory_bandwidth_gbs * 1e9 / weights
        print(f"  теоретический предел decode: {theoretical_tps:>8.0f} tokens/s")
        print(
            f"  фактически: {decode_tps:>8.0f} tokens/s "
            f"({100 * decode_tps / theoretical_tps:.0f}% от предела)"
        )
        note(
            "Каждый decode-шаг обязан прочитать ~весь объём весов из памяти — "
            "именно это, а не арифметика, задаёт потолок."
        )
    except Exception as exc:  # pragma: no cover
        note(f"Не удалось оценить bandwidth: {exc}")

    section("Вывод")
    note("prefill  → compute-bound,  все токены промпта параллельно.")
    note("decode   → memory-bound,   по одному токену, но время шага ~константа.")
    note(
        "Отсюда всё остальное: батчинг (эксп. 3) окупается именно на decode, "
        "а KV-кэш (эксп. 2) экономит именно повторный prefill."
    )


if __name__ == "__main__":
    main()
