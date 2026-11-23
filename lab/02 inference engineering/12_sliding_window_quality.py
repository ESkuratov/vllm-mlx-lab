# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 12: скользящее окно KV и качество на длинном контексте.

Идея
----
Чтобы не держать бесконечный KV-кэш, можно оставить только последние W токенов
(sliding-window attention / max_kv_size). Тогда модель «забывает» начало
контекста. Здесь кладём важный факт в самое начало, заливаем длинным
наполнителем и смотрим, что модель отвечает при разных окнах.

Что измеряем: расхождение логитов с полным кэшем и фактический ответ.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/12_sliding_window_quality.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from common import MODEL_ID, load_model, note, section, subsection

CHUNK = 32
WINDOWS = (None, 1024, 256, 128, 64, 32)


def build_cache(model, ids: mx.array, window):
    cache = (
        make_prompt_cache(model, max_kv_size=window)
        if window
        else make_prompt_cache(model)
    )
    logits = None
    for i in range(0, ids.shape[0], CHUNK):
        logits = model(ids[i : i + CHUNK][None], cache=cache)
        mx.eval(logits)
    return cache, logits


def greedy_text(model, tok, cache, logits, n: int = 6) -> str:
    out = []
    tok_id = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    for _ in range(n):
        out.append(int(tok_id.item()))
        logits = model(tok_id, cache=cache)
        mx.eval(logits)
        tok_id = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    return tok.decode(out)


def kl_divergence(p_logits: mx.array, q_logits: mx.array) -> float:
    lp = p_logits - mx.logsumexp(p_logits)
    lq = q_logits - mx.logsumexp(q_logits)
    return max(0.0, float(mx.sum(mx.exp(lp) * (lp - lq))))


def main() -> None:
    section("12. Скользящее окно KV: память против качества")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()

    fact = tok.encode("The secret number is 31415. Remember it. ")
    filler = tok.encode("The grass is green and the sky is blue. ") * 24
    question = tok.encode("\nQuestion: What is the secret number? Answer:")
    ids = mx.array(fact + filler + question)
    print(f"  длина контекста: {ids.shape[0]} токенов, факт — в начале")

    ref_cache, ref_logits = build_cache(model, ids, None)
    ref = ref_logits[0, -1]
    ref_top = int(mx.argmax(ref))
    print(f"  ответ с полным кэшем: {tok.decode([ref_top])!r}")

    subsection("A. Что происходит при разных окнах")
    print(f"  {'окно':>7}{'offset':>8}{'KL(fp‖·)':>12}{'top-1':>9}  следующий токен")
    for window in WINDOWS:
        cache, logits = build_cache(model, ids, window)
        cur = logits[0, -1]
        kl = kl_divergence(ref, cur)
        match = "да" if int(mx.argmax(cur)) == ref_top else "НЕТ"
        label = "всё" if window is None else str(window)
        print(
            f"  {label:>7}{cache[0].offset:>8}{kl:>12.2e}{match:>9}"
            f"  {tok.decode([int(mx.argmax(cur))])!r}"
        )

    subsection("B. Сгенерированный ответ")
    for window in (None, 256, 32):
        cache, logits = build_cache(model, ids, window)
        text = greedy_text(model, tok, cache, logits, n=8)
        label = "полный кэш" if window is None else f"окно {window}"
        print(f"  {label:<12}: {text!r}")
    note(
        "Маленькое окно видит вопрос, но уже не видит факт в начале — ответ "
        "меняется. Это цена экономии памяти."
    )

    subsection("C. Память")
    full = sum(getattr(c, "nbytes", 0) for c in ref_cache)
    for window in (256, 64):
        cache, _ = build_cache(model, ids, window)
        used = sum(getattr(c, "nbytes", 0) for c in cache)
        print(f"  окно {window:>4}: {used / 1024**2:>6.1f} MB  vs полный {full / 1024**2:.1f} MB")
    note("Во vllm-mlx окно задаётся флагом --max-kv-size.")

    section("Вывод")
    note("Скользящее окно ограничивает память, но теряет дальний контекст.")
    note("Это осознанный trade-off: экономия RAM против «памяти» модели.")


if __name__ == "__main__":
    main()
