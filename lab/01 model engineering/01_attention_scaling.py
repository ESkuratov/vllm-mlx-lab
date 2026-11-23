# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 01: масштабирование внимания и GQA.

Идея
----
Внимание стоит O(S²) по длине последовательности: каждый токен смотрит на всех
предыдущих. Поэтому у длинных промптов prefill на токен дорожает. Смягчают это
GQA (grouped-query attention): ключей/значений в разы меньше, чем query-голов,
и KV-кэш становится компактнее.

Что измеряем:
  1. время одного attention-вызова от длины S (показываем O(S²));
  2. цену GQA (много KV-голов) vs уменьшенного числа KV-голов;
  3. реальный prefill модели на разных длинах.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/01_attention_scaling.py"
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
    load_model,
    note,
    section,
    subsection,
)

HEADS = 32
HEAD_DIM = 128
LENGTHS = (256, 512, 1024, 2048, 4096)


def sdpa_time(seq: int, kv_heads: int) -> float:
    q = mx.random.normal((1, HEADS, seq, HEAD_DIM)).astype(mx.float16)
    k = mx.random.normal((1, kv_heads, seq, HEAD_DIM)).astype(mx.float16)
    v = mx.random.normal((1, kv_heads, seq, HEAD_DIM)).astype(mx.float16)
    mx.eval(q, k, v)

    def run():
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=HEAD_DIM**-0.5)
        mx.eval(out)

    run()  # прогрев
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def main() -> None:
    section("01. Внимание: O(S²) и GQA")
    print(f"  модель: {MODEL_ID}")

    # ------------------------------------------------ изолированный SDPA
    subsection("A. Время attention-вызова vs длина (q-голов: 32, KV: 32)")
    print(f"  {'S':>6}{'время':>12}{'нс/токен':>12}{'×S':>8}")
    base = None
    for seq in LENGTHS:
        t = sdpa_time(seq, HEADS)
        if base is None:
            base = (seq, t)
        print(
            f"  {seq:>6}{t * 1e3:>10.3f} ms{t / seq * 1e9:>11.1f}{t / base[1]:>8.1f}"
        )
    note(
        "Время растёт быстрее длины (столбец ×S — это отношение ко времени при "
        "S=256): при удвоении S время примерно учетверяется — классический O(S²)."
    )

    # ------------------------------------------------------------- GQA
    subsection("B. GQA: во сколько раз меньше KV")
    model, tok = load_model()
    args = model.args
    n_heads = getattr(args, "num_attention_heads", HEADS)
    n_kv = getattr(args, "num_key_value_heads", n_heads)
    print(f"  query-голов: {n_heads}, KV-голов: {n_kv} (в {n_heads / n_kv:.0f}× меньше)")
    print(f"  {'S':>6}{'KV: все головы':>17}{'KV: GQA':>12}{'экономия':>10}")
    for seq in (1024, 4096, 16384):
        per_tok = 2 * HEAD_DIM * 2  # K и V, fp16
        mha = seq * n_heads * per_tok
        gqa = seq * n_kv * per_tok
        print(f"  {seq:>6}{mha / 1024**2:>15.1f} MB{gqa / 1024**2:>10.1f} MB{mha / gqa:>9.0f}x")
    note(
        "Меньше KV-голов → меньше KV-кэш и меньше данных читать на каждом шаге. "
        "Для Qwen2.5 это 2 KV-головы против 16 query → в 8 раз компактнее."
    )

    # ------------------------------------------------ GQA в самом SDPA
    subsection("C. Цена attention при разном числе KV-голов")
    print(f"  {'S':>6}{'KV=32':>12}{'KV=2':>12}")
    for seq in (1024, 4096):
        print(f"  {seq:>6}{sdpa_time(seq, 32) * 1e3:>10.3f} ms{sdpa_time(seq, 2) * 1e3:>10.3f} ms")
    note(
        "Важно: сами FLOPs внимания от числа KV-голов почти не зависят "
        "(считаются по query-головам), поэтому время одинаковое. Выигрыш GQA — "
        "в ПАМЯТИ KV-кэша и в объёме чтения K/V, а не в скорости самого attention."
    )

    # ------------------------------------------------ реальный prefill
    subsection("D. Реальный prefill модели: стоимость на токен растёт")
    print(f"  {'длина':>7}{'время':>12}{'tokens/s':>11}")
    for length in (256, 512, 1024, 2048, 4096):
        ids = base_token_ids(tok, length)
        cache = make_prompt_cache(model)
        t0 = time.perf_counter()
        mx.eval(model(ids[None], cache=cache))
        t = time.perf_counter() - t0
        print(f"  {length:>7}{t * 1000:>10.0f} ms{length / t:>11.0f}")
    note(
        "На коротких промптах доминируют MLP/layernorm, поэтому tokens/s почти "
        "плоский. Вклад O(S²) внимания становится заметен на очень длинных S."
    )

    section("Вывод")
    note("Attention O(S²) → длинный контекст дорожает непропорционально.")
    note("GQA сокращает KV-кэш и чтение K/V в разы.")
    note("Во vllm-mlx: --max-kv-size (окно), KV-квантизация (`inference/05`).")


if __name__ == "__main__":
    main()
