# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 5: квантизация KV-кэша.

Идея
----
KV-кэш растёт линейно с контекстом и при батчинге съедает много памяти.
Его можно хранить не в fp16, а в 8- или 4-битном виде. Работает это так же,
как весовая квантизация: тензор бьётся на группы по `group_size` элементов,
для каждой группы хранится шкала и смещение, а элементы — целыми числами.

Что измеряем:
  1. сколько памяти экономят 8 и 4 бита;
  2. как квантизация влияет на качество (расхождение логитов);
  3. как меняется скорость decode.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/05_kv_quantization.py"
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
    cache_nbytes,
    load_model,
    mb,
    note,
    section,
    subsection,
)

CONTEXT = 512
GROUP = 64


def fresh_prefill(model, ids):
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    return cache, logits


def quantize(cache, bits: int):
    return [layer.to_quantized(GROUP, bits) for layer in cache]


def decode_latency(model, cache, logits, n: int = 16) -> float:
    tok = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    steps = []
    for _ in range(n):
        t0 = time.perf_counter()
        logits = model(tok, cache=cache)
        mx.eval(logits)
        steps.append(time.perf_counter() - t0)
        tok = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    return statistics.median(steps[4:])


def kl_divergence(p_logits: mx.array, q_logits: mx.array) -> float:
    lp = p_logits - mx.logsumexp(p_logits)
    lq = q_logits - mx.logsumexp(q_logits)
    return float(mx.sum(mx.exp(lp) * (lp - lq)))


def main() -> None:
    section("5. Квантизация KV-кэша")
    print(f"  модель: {MODEL_ID}, контекст: {CONTEXT} токенов, group_size={GROUP}")
    model, tok = load_model()
    ids = base_token_ids(tok, CONTEXT)

    cache_fp, logits = fresh_prefill(model, ids)
    cache_q8 = quantize(cache_fp, 8)
    cache_q4 = quantize(cache_fp, 4)

    # ------------------------------------------------------------ память
    subsection("A. Память")
    fp, q8, q4 = (
        cache_nbytes(cache_fp),
        cache_nbytes(cache_q8),
        cache_nbytes(cache_q4),
    )
    print(f"  {'fp16':<6}{mb(fp):>10}  {'100%':>6}")
    print(f"  {'8-bit':<6}{mb(q8):>10}  {q8 / fp:>5.0%}")
    print(f"  {'4-bit':<6}{mb(q4):>10}  {q4 / fp:>5.0%}")
    note(
        "Меньше половины — потому что кроме самих значений хранятся шкалы и "
        "смещения (по одному на каждые group_size элементов)."
    )

    # ------------------------------------------------------------ качество
    subsection("B. Качество: расхождение логитов на следующем токене")
    nid = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    out_fp = model(nid, cache=cache_fp)
    mx.eval(out_fp)
    out_q8 = model(nid, cache=cache_q8)
    mx.eval(out_q8)
    out_q4 = model(nid, cache=cache_q4)
    mx.eval(out_q4)

    ref = out_fp[0, -1]
    ref_top = int(mx.argmax(ref))
    print(f"  {'вариант':<7}{'max|Δlogit|':>13}{'KL(fp‖·)':>11}{'top-1':>8}")
    for name, out in (("8-bit", out_q8), ("4-bit", out_q4)):
        logits_q = out[0, -1]
        max_diff = float(mx.max(mx.abs(logits_q - ref)))
        kl = max(kl_divergence(ref, logits_q), 0.0)
        top_match = "да" if int(mx.argmax(logits_q)) == ref_top else "НЕТ"
        print(f"  {name:<7}{max_diff:>13.4f}{kl:>11.2e}{top_match:>8}")
    note(
        "8 бит обычно почти не меняют ответ; 4 бита заметно шумят — "
        "поэтому vllm-mlx по умолчанию квантизует в 8 бит."
    )

    # ------------------------------------------------------------ скорость
    subsection("C. Скорость decode")
    print(f"  {'вариант':<7}{'мс/шаг':>10}{'tokens/s':>10}")
    results = {}
    for name, bits in (("fp16", None), ("8-bit", 8), ("4-bit", 4)):
        c, lg = fresh_prefill(model, ids)
        if bits is not None:
            c = quantize(c, bits)
        lat = decode_latency(model, c, lg)
        results[name] = lat
        print(f"  {name:<7}{lat * 1000:>10.2f}{1 / lat:>10.1f}")
    note(
        "Внимание: на скорость это почти не влияет. Сам KV-кэш (десятки MB) "
        "в разы меньше весов модели (GB), и decode читает прежде всего веса. "
        "Де-квантизация даже слегка замедляет шаг. Квантизация KV — это "
        "прежде всего экономия ПАМЯТИ (влезает больше параллельных запросов)."
    )

    # --------------------------------------------------------- как включить
    subsection("D. Как это включено во vllm-mlx")
    print("  флаги сервера/бенчмарка:")
    print("    --kv-cache-quantization")
    print("    --kv-cache-quantization-bits {4,8}       (по умолчанию 8)")
    print("    --kv-cache-quantization-group-size 64")
    print("    --kv-cache-min-quantize-tokens 256       (не трогать короткие префиксы)")
    note(
        "В mlx-lm это параметры generate_step: kv_bits, kv_group_size, "
        "quantized_kv_start — квантизовать начинают только после N токенов."
    )

    section("Вывод")
    note("KV-кэш можно хранить в 8/4 битах, экономя >50% памяти.")
    note("8 бит — безопасный компромисс; 4 бита заметно шумят.")
    note("Скорость почти не меняется: выигрыш здесь в памяти, а не в латентности.")


if __name__ == "__main__":
    main()
