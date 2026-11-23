# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 15: детерминизм генерации при батчинге.

Идея
----
Greedy-генерация должна быть детерминированной: один и тот же промпт даёт один
и тот же ответ. Но батчинг, паддинг, разная позиция в батче и порядок операций
с плавающей запятой теоретически могут менять результат. Проверяем на практике.

Что измеряем: совпадают ли токены для одного и того же промпта
   * при одиночной генерации,
   * в батче с другими запросами,
   * на разных позициях в батче и при разном размере батча.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/15_batch_determinism.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm.generate import batch_generate, generate_step
from mlx_lm.sample_utils import make_sampler

from common import (
    MODEL_ID,
    base_token_ids,
    load_model,
    note,
    section,
    subsection,
)

MAX_TOKENS = 24


def gen_alone(model, prompt, sampler):
    ids = []
    for token, _ in generate_step(
        prompt, model, max_tokens=MAX_TOKENS, sampler=sampler
    ):
        ids.append(int(token))
    return ids


def gen_batch(model, tok, prompts, b, sampler):
    r = batch_generate(
        model, tok, prompts, max_tokens=MAX_TOKENS,
        completion_batch_size=b, prefill_batch_size=b,
        sampler=sampler, return_token_ids=True,
    )
    return r.token_ids


def main() -> None:
    section("15. Детерминизм генерации при батчинге")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    sampler = make_sampler(temp=0.0)  # greedy

    prompt_a = base_token_ids(tok, 48)
    prompt_b = (base_token_ids(tok, 48) + 1000).tolist()  # другой промпт
    prompt_a_list = prompt_a.tolist()

    ref = gen_alone(model, prompt_a, sampler)
    print(f"  эталон (одиночная генерация): {len(ref)} токенов")

    subsection("A. Тот же промпт в разных батчах")
    cases = [
        ("[A]", [prompt_a_list], 1, 0),
        ("[A, B]", [prompt_a_list, prompt_b], 2, 0),
        ("[B, A]", [prompt_b, prompt_a_list], 2, 1),
        ("[A] × 4", [prompt_a_list] * 4, 4, 0),
        ("[A, B, B, B]", [prompt_a_list, prompt_b, prompt_b, prompt_b], 4, 0),
    ]
    print(f"  {'батч':<16}{'позиция':>8}{'совпало':>10}{'общих токенов':>15}")
    all_ok = True
    for label, prompts, b, pos in cases:
        ids = gen_batch(model, tok, prompts, b, sampler)
        got = ids[pos]
        same = got == ref
        all_ok &= same
        shared = sum(1 for x, y in zip(got, ref) if x == y)
        print(f"  {label:<16}{pos:>8}{'да' if same else 'НЕТ':>10}{shared:>15}")

    subsection("B. Разный размер батча (тот же A на позиции 0)")
    for b in (1, 2, 4, 8):
        prompts = [prompt_a_list] + [prompt_b] * (b - 1)
        got = gen_batch(model, tok, prompts, b, sampler)[0]
        print(f"  батч {b:>2}: {'совпало' if got == ref else 'НЕТ'}")

    subsection("C. Кросс-проверка: A в одиночку ещё раз")
    again = gen_alone(model, prompt_a, sampler)
    print(f"  повтор одиночной генерации: {'совпало' if again == ref else 'НЕТ'}")

    section("Вывод")
    if all_ok:
        note("На этой конфигурации greedy-генерация детерминирована и в батче.")
    else:
        note("Обнаружены расхождения — батчинг/порядок операций влияют на результат.")
    note(
        "Практический вывод: полагаться на побитовое совпадение между разными "
        "движками/батчами нельзя; для воспроизводимости фиксируй seed и "
        "используй greedy/одинаковый батч."
    )


if __name__ == "__main__":
    main()
