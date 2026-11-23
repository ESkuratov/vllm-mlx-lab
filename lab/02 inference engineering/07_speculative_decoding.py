# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 7: speculative decoding (ускорение через draft-модель).

Идея
----
Decode memory-bound: чтобы сгенерировать один токен, большая модель читает все
свои веса. Speculative decoding использует маленькую быструю draft-модель:
она предлагает k токенов вперёд, а большая модель проверяет их одним проходом
(параллельно, как prefill). Если draft угадала — получили k токенов за цену
одного прохода большой модели.

Что измеряем:
  1. обычная генерация (target) vs speculative;
  2. как число draft-токенов k влияет на ускорение и acceptance rate.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/07_speculative_decoding.py"

Draft-модель задаётся переменной LAB_DRAFT (по умолчанию Qwen2.5-Coder-0.5B).
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.generate import generate_step, speculative_generate_step
from mlx_lm.sample_utils import make_sampler

from common import (
    MODEL_ID,
    base_token_ids,
    load_model,
    note,
    section,
    subsection,
    sync,
)

DRAFT_ID = os.environ.get("LAB_DRAFT", "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit")
PROMPT_TOKENS = 64
MAX_TOKENS = 48


def run_plain(model, ids, sampler) -> tuple[int, float]:
    t0 = time.perf_counter()
    n = 0
    for _ in generate_step(ids, model, max_tokens=MAX_TOKENS, sampler=sampler):
        n += 1
    sync()
    return n, time.perf_counter() - t0


def run_speculative(model, draft, ids, k, sampler) -> tuple[int, int, float]:
    t0 = time.perf_counter()
    n = 0
    accepted = 0
    for token, _logprobs, from_draft in speculative_generate_step(
        ids, model, draft, num_draft_tokens=k, max_tokens=MAX_TOKENS, sampler=sampler
    ):
        n += 1
        accepted += int(from_draft)
    sync()
    return n, accepted, time.perf_counter() - t0


def main() -> None:
    section("7. Speculative decoding")
    print(f"  target: {MODEL_ID}")
    print(f"  draft : {DRAFT_ID}")
    model, tok = load_model()
    draft, _ = load_model(DRAFT_ID)
    ids = base_token_ids(tok, PROMPT_TOKENS)
    sampler = make_sampler(temp=0.0)

    # прогрев обеих моделей
    run_plain(model, base_token_ids(tok, 8), sampler)
    run_speculative(model, draft, base_token_ids(tok, 8), 2, sampler)

    # ------------------------------------------------------------ baseline
    subsection("A. Обычная генерация (только target)")
    n_plain, t_plain = run_plain(model, ids, sampler)
    tps_plain = n_plain / t_plain
    print(f"  {n_plain} токенов за {t_plain:.2f} с → {tps_plain:.0f} tokens/s")

    # ---------------------------------------------------------- speculative
    subsection("B. Speculative decoding: сколько ускоряет")
    print(
        f"  {'k':>4}{'время':>10}{'токенов':>9}{'tokens/s':>11}"
        f"{'ускорение':>11}{'из draft':>10}"
    )
    for k in (1, 2, 4, 8):
        n, accepted, t = run_speculative(model, draft, ids, k, sampler)
        speedup = (n / t) / tps_plain
        print(
            f"  {k:>4}{t:>8.2f} s{n:>9}{n / t:>11.0f}"
            f"{speedup:>10.2f}x{accepted / n:>10.0%}"
        )
    note(
        "Ускорение есть, пока draft чаще угадывает и пока цена проверки "
        "(один параллельный проход target) окупает работу draft."
    )
    note(
        "Если k слишком велик, лишние draft-проходы не окупаются; "
        "если draft плохо совпадает с target — ускорения нет."
    )

    section("Вывод")
    note("Speculative = «предсказал k токенов — проверил одним проходом».")
    note("Выигрыш растёт с acceptance rate и падает с ростом k.")
    note("Во vllm-mlx: --enable-mtp, --mtp-num-draft-tokens, --mllm-draft-model.")


if __name__ == "__main__":
    main()
