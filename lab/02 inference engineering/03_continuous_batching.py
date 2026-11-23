# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 3: зачем нужен continuous batching.

Идея
----
Decode упирается в память: один шаг читает все веса ради одного токена.
Если обслуживать несколько последовательностей в одном батче, веса читаются
один раз на всех → суммарный tokens/s растёт почти линейно по батчу.

* **static batching** — набираем батч и ждём, пока завершится самая длинная
  последовательность; короткие всё это время «простаивают» (пустые слоты).
* **continuous batching** — как только последовательность закончилась, её слот
  сразу занимает следующий запрос из очереди. Слоты почти не простаивают.

Что измеряем:
  1. последовательная генерация vs один общий батч;
  2. масштабирование пропускной способности от размера батча;
  3. симуляция static vs continuous на неравномерной нагрузке.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/03_continuous_batching.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm.generate import batch_generate, generate_step
from mlx_lm.sample_utils import make_sampler
import mlx.core as mx

from common import (
    MODEL_ID,
    base_token_ids,
    load_model,
    note,
    section,
    subsection,
    sync,
)
import time


MAX_TOKENS = 32


def generate_one(model, prompt_ids, max_tokens, sampler) -> int:
    """Последовательная генерация одного промпта; вернуть число токенов."""
    count = 0
    for _ in generate_step(
        mx.array(prompt_ids), model, max_tokens=max_tokens, sampler=sampler
    ):
        count += 1
    return count


def simulate(lengths, slots: int, continuous: bool) -> int:
    """Сколько decode-шагов займёт workload при заданном числе слотов."""
    if not continuous:
        steps = 0
        for i in range(0, len(lengths), slots):
            group = lengths[i : i + slots]
            steps += max(group)  # ждём самую длинную
        return steps

    queue = list(lengths)
    running: list[int] = []
    steps = 0
    while queue or running:
        while queue and len(running) < slots:
            running.append(queue.pop(0))
        steps += 1
        running = [r - 1 for r in running if r - 1 > 0]
    return steps


def main() -> None:
    section("3. Continuous batching")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    sampler = make_sampler(temp=0.0)  # greedy для детерминизма

    # ---------------------------------------------- последовательно vs батч
    subsection("A. Последовательно vs один батч (8 запросов)")
    n = 8
    prompts = [base_token_ids(tok, 64).tolist() for _ in range(n)]

    t0 = time.perf_counter()
    seq_tokens = sum(generate_one(model, p, MAX_TOKENS, sampler) for p in prompts)
    sync()
    seq_time = time.perf_counter() - t0

    resp = batch_generate(
        model,
        tok,
        prompts,
        max_tokens=MAX_TOKENS,
        completion_batch_size=n,
        prefill_batch_size=n,
        sampler=sampler,
    )
    batch_tokens = resp.stats.generation_tokens

    print(f"  {'режим':<14}{'время':>10}{'токенов':>10}{'tokens/s':>12}")
    print(f"  {'последоват.':<14}{seq_time:>8.2f} s{seq_tokens:>10}{seq_tokens / seq_time:>12.0f}")
    print(f"  {'батч':<14}{resp.stats.wall_time:>8.2f} s{batch_tokens:>10}{resp.stats.generation_tps:>12.0f}")
    print(f"  ускорение: {seq_time / resp.stats.wall_time:.1f}x")
    note(
        "Батч читает веса один раз на всех → пропускная способность во много раз "
        "выше, при этом латентность отдельного запроса почти не растёт."
    )

    # ------------------------------------------------ масштабирование
    subsection("B. Пропускная способность растёт с размером батча")
    n = 16
    prompts = [base_token_ids(tok, 64).tolist() for _ in range(n)]
    print(
        f"  {'батч':>5}{'время':>9}{'tokens/s':>11}"
        f"{'мс/шаг':>9}{'ток./шаг':>10}{'prompt, с':>11}"
    )
    for b in (1, 2, 4, 8, 16):
        r = batch_generate(
            model,
            tok,
            prompts,
            max_tokens=MAX_TOKENS,
            completion_batch_size=b,
            prefill_batch_size=b,
            sampler=sampler,
        )
        s = r.stats
        ms_per_step = 1000 * s.decode_time / max(s.generation_steps, 1)
        tok_per_step = s.generation_tokens / max(s.generation_steps, 1)
        print(
            f"  {b:>5}{s.wall_time:>8.2f} s{s.generation_tps:>11.0f}"
            f"{ms_per_step:>9.2f}{tok_per_step:>10.1f}{s.prompt_time:>11.3f}"
        )
    note(
        "Время decode-шага растёт, но заметно медленнее числа токенов: пока "
        "упор в память — throughput растёт почти линейно, на больших батчах "
        "начинает ограничивать уже compute (шаг 1→16 стал в 4.2× дороже, "
        "а токенов на шаг — в 16×)."
    )

    # ------------------------------------------- static vs continuous
    subsection("C. Static vs continuous на неравномерной нагрузке")
    lengths = [4, 6, 8, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128,
               4, 6, 8, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96, 128]
    slots = 8
    total = sum(lengths)
    steps_static = simulate(lengths, slots, continuous=False)
    steps_cont = simulate(lengths, slots, continuous=True)
    print(f"  запросов: {len(lengths)}, слотов: {slots}, суммарно токенов: {total}")
    print(f"  {'режим':<12}{'шагов':>8}{'простой слотов':>18}{'утилизация':>13}")
    for name, steps in (("static", steps_static), ("continuous", steps_cont)):
        util = total / (steps * slots)
        print(f"  {name:<12}{steps:>8}{steps * slots - total:>18}{util:>12.0%}")
    note(
        "Static простаивает, пока не закончится самая длинная в группе. "
        "Continuous сразу занимает освободившийся слот новой работой."
    )
    note(
        "Именно этим управляют во vllm-mlx: --continuous-batching, "
        "--max-num-seqs, --prefill-batch-size, --completion-batch-size."
    )

    section("Вывод")
    note("Decode memory-bound → батчинг амортизирует чтение весов.")
    note("Continuous batching выжимает простой слотов из-за разной длины ответов.")
    note("Плата — KV-кэш на каждый активный запрос (эксп. 2, 4, 5).")


if __name__ == "__main__":
    main()
