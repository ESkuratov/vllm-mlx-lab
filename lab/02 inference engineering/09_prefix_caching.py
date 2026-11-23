# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 9: prefix caching (переиспользование общего префикса).

Идея
----
Сервер часто видит один и тот же длинный префикс: системный промпт, few-shot
примеры, документ. Prefix caching сохраняет KV-кэш для общего префикса и при
следующем запросе прогоняет через модель только «хвост» — TTFT падает.

Здесь используется настоящий LRU-кэш префиксов из mlx-lm (та же идея, что в
``vllm_mlx/prefix_cache.py`` и ``memory_cache.py``).

Что измеряем: время prefill первого запроса vs повторного с общим префиксом.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/09_prefix_caching.py"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.models.cache import LRUPromptCache, make_prompt_cache

from common import (
    MODEL_ID,
    base_token_ids,
    load_model,
    note,
    section,
    subsection,
)


def serve(model, cache_mgr: LRUPromptCache, model_key, tokens: list[int]) -> float:
    """Обработать запрос: взять общий префикс из кэша, досчитать остаток."""
    cache, remaining = cache_mgr.fetch_nearest_cache(model_key, tokens)
    if cache is None:
        cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    if remaining:
        logits = model(mx.array(remaining)[None], cache=cache)
        mx.eval(logits)
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    cache_mgr.insert_cache(model_key, tokens, cache)
    return elapsed


def main() -> None:
    section("9. Prefix caching")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    cache_mgr = LRUPromptCache(max_size=16)
    model_key = (MODEL_ID, None, None)  # ключ модели, как во внутреннем сервере mlx-lm

    system = base_token_ids(tok, 512).tolist()  # длинный общий префикс
    question_a = tok.encode(" Explain what a KV cache is in one sentence.")
    question_b = tok.encode(" Write a haiku about memory.")
    req_a = system + question_a
    req_b = system + question_b

    # прогрев
    serve(model, cache_mgr, model_key, base_token_ids(tok, 32).tolist())

    subsection("A. TTFT: первый запрос vs повторный с тем же префиксом")
    t_a = serve(model, cache_mgr, model_key, req_a)
    t_b = serve(model, cache_mgr, model_key, req_b)

    print(f"  общий системный префикс: {len(system)} токенов")
    print(f"  {'запрос':<22}{'prefill':>11}{'токенов':>10}")
    print(f"  {'A (первый, всё)':<22}{t_a * 1000:>9.0f} ms{len(req_a):>10}")
    print(
        f"  {'B (общий префикс)':<22}{t_b * 1000:>9.0f} ms{len(question_b):>10}"
    )
    print(f"  ускорение prefill: {t_a / t_b:.1f}x")
    note(
        "Запрос B пересчитывает только свои ~10 токенов вопроса: "
        "весь системный промпт берётся из кэша в готовом виде."
    )

    subsection("B. Что это даёт на практике")
    note(
        "В RAG/агентных сценариях общий контекст — часто тысячи токенов; "
        "без кэша каждый запрос платит за их повторный prefill."
    )
    note(
        "Во vllm-mlx: --enable-prefix-cache, --prefix-cache-size, "
        "--cache-memory-mb, --cache-memory-percent."
    )

    section("Вывод")
    note("Prefix caching превращает повторный prefill общего префикса в ~0.")
    note("Ограничения — память под кэш и вытеснение по LRU / бюджету памяти.")


if __name__ == "__main__":
    main()
