# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 14: стоимость детокенизации (detok).

Идея
----
Модель генерирует токены (числа), а пользователю нужен текст. Превращение
токенов в текст — тоже работа. Наивный способ «декодировать весь список
токенов на каждом шаге» квадратичен по длине ответа; инкрементальный
(декодировать только новый токен) — линеен.

Во vllm-mlx этому посвящён отдельный инструмент: ``vllm-mlx bench-detok``.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/14_detokenization.py"
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from common import (
    MODEL_ID,
    base_token_ids,
    load_model,
    note,
    section,
    subsection,
)

MAX_TOKENS = 256


def main() -> None:
    section("14. Детокенизация: наивно vs инкрементально")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    sampler = make_sampler(temp=0.0)

    # --------------------------------------------------- сгенерировать ответ
    prompt_ids = base_token_ids(tok, 32)
    generated = []
    t0 = time.perf_counter()
    for token, _ in generate_step(
        prompt_ids, model, max_tokens=MAX_TOKENS, sampler=sampler
    ):
        generated.append(int(token))
    mx.synchronize()
    t_gen = time.perf_counter() - t0
    n = len(generated)
    print(f"  сгенерировано {n} токенов за {t_gen * 1000:.0f} ms")

    # ------------------------------------------------------- наивный detok
    subsection("A. Наивно: decode(all_tokens) на каждом шаге")
    t0 = time.perf_counter()
    for i in range(1, n + 1):
        _ = tok.decode(generated[:i])
    t_naive = time.perf_counter() - t0
    print(f"  суммарно: {t_naive * 1000:.1f} ms ({t_naive / n * 1e6:.0f} µs/токен)")

    # --------------------------------------------------- инкрементальный
    subsection("B. Инкрементально: decode одного нового токена")
    t0 = time.perf_counter()
    parts = []
    for tok_id in generated:
        parts.append(tok.decode([tok_id]))
    text = "".join(parts)
    t_inc = time.perf_counter() - t0
    print(f"  суммарно: {t_inc * 1000:.1f} ms ({t_inc / n * 1e6:.0f} µs/токен)")

    # ------------------------------------------------------- доля от времени
    subsection("C. Насколько это существенно")
    print(f"  генерация:        {t_gen * 1000:>8.1f} ms")
    print(f"  detok наивный:    {t_naive * 1000:>8.1f} ms  ({t_naive / t_gen:>5.0%} от генерации)")
    print(f"  detok инкремент.: {t_inc * 1000:>8.1f} ms  ({t_inc / t_gen:>5.0%} от генерации)")
    print(f"  ускорение detok:  {t_naive / t_inc:.1f}x")
    note(
        "Наивный detok делает O(n²) работы (каждый раз декодирует весь префикс): "
        "на 256 токенах он в ~15× дороже инкрементального. В этом прогоне detok — "
        "доли процента от генерации, но при быстрой модели, большом батче или "
        "очень длинных ответах его накладные расходы растут квадратично."
    )
    note(
        "Тонкость: инкрементальный detok нельзя делать грубым срезом — "
        "UTF-8-символ может склеиваться из нескольких токенов. Поэтому нужен "
        "state machine: в mlx-lm это TextStateMachine, во vllm-mlx — "
        "--stream-interval и свой detokenizer."
    )

    section("Вывод")
    note("Детокенизация — реальная статья расходов при стриминге.")
    note("Декодируй только новый токен; незавершённые UTF-8 держи в буфере.")
    note("Проверить: vllm-mlx bench-detok (см. команды CLI).")


if __name__ == "__main__":
    main()
