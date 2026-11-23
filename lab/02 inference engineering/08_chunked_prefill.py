# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 8: chunked prefill (prefill длинного промпта по частям).

Идея
----
Один большой prefill держит в памяти промежуточные активации всего промпта
(внимание квадратично по длине). Если резать промпт на чанки, пик памяти падает,
а суммарная работа почти не меняется. Во vllm-mlx это позволяет «вклинивать»
обработку длинных промптов между decode-шагами других запросов — так длинный
prefill не блокирует всех.

Что измеряем: время prefill и пиковую память при разных размерах чанка.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/08_chunked_prefill.py"
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

TOTAL = 2048
CHUNKS = (128, 256, 512, 1024, 2048)


def prefill_chunked(model, ids, chunk: int):
    cache = make_prompt_cache(model)
    n = ids.shape[0]
    logits = None
    t0 = time.perf_counter()
    for i in range(0, n, chunk):
        logits = model(ids[i : i + chunk][None], cache=cache)
        mx.eval(logits)
    mx.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    section("8. Chunked prefill")
    print(f"  модель: {MODEL_ID}, промпт: {TOTAL} токенов")
    model, tok = load_model()
    ids = base_token_ids(tok, TOTAL)
    prefill_chunked(model, base_token_ids(tok, 64), 64)  # прогрев

    subsection("A. Время и пиковая память при разных размерах чанка")
    print(f"  {'чанк':>6}{'время':>11}{'tokens/s':>11}{'пик RAM':>12}")
    for chunk in CHUNKS:
        mx.reset_peak_memory()
        t = statistics.median(
            prefill_chunked(model, ids, chunk) for _ in range(3)
        )
        peak = mx.get_peak_memory() / 1024**2
        print(f"  {chunk:>6}{t * 1000:>9.0f} ms{TOTAL / t:>11.0f}{peak:>10.0f} MB")
    note(
        "Суммарное время почти не меняется (работа та же), но чем мельче чанк, "
        "тем ниже пик памяти: промежуточные активации считаются по куску."
    )

    subsection("B. Зачем это нужно")
    note(
        "Один огромный prefill занимает GPU надолго и держит пик памяти. "
        "Разбив его на чанки, планировщик может между кусками обслуживать "
        "decode-шаги других запросов → ниже задержка под нагрузкой."
    )
    note(
        "Во vllm-mlx: --prefill-step-size, --chunked-prefill-tokens, "
        "--prefill-batch-size."
    )

    section("Вывод")
    note("Chunked prefill — trade-off: чуть больше накладных расходов,")
    note("но меньше пик памяти и лучше интерактивность под нагрузкой.")


if __name__ == "__main__":
    main()
