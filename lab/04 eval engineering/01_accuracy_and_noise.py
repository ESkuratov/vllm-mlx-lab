# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 1 (eval): точность — это оценка, а не число.

Идея
----
«Модель стала лучше на 2 пункта» — почти всегда шум. Accuracy считается по
конечной выборке, значит у неё есть стандартная ошибка: sqrt(p(1-p)/n). На
12 примерах при p≈0.8 это ±8 пунктов, и любой «прирост» меньше этого числа
неотличим от подбрасывания монетки.

Здесь мы смотрим на три источника неопределённости, которые обычно
смешивают в один:

* **выборка** — примеров мало, ответы разных примеров не независимы;
* **схема прогона** — greedy детерминирован, sampling — нет; сравнивать
  конфигурации можно только при одной схеме;
* **сравнение** — разница двух accuracy имеет свою ошибку, и она в sqrt(2)
  раз больше, чем ошибка одного замера.

Что измеряем:
  0. accuracy greedy-прогона на фиксированном наборе с проверяемым ответом;
  1. детерминизм: совпадают ли ответы трёх greedy-прогонов;
  2. шум sampling: разброс accuracy и согласие по каждому примеру;
  3. доверительный интервал (Wilson) и минимально различимая дельта;
  4. сколько примеров нужно, чтобы увидеть дельту в 5 пунктов.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/04 eval engineering/01_accuracy_and_noise.py"

Модель по умолчанию слабовата для арифметики; для более чистой картинки:
    LAB_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit \
        python "lab/04 eval engineering/01_accuracy_and_noise.py"
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import generate
from mlx_lm.sample_utils import make_sampler

from common import MODEL_ID, load_model, note, section, subsection

# (вопрос, ожидаемый ответ) — ответы проверяются программно, не «на глаз»
TASKS: list[tuple[str, str]] = [
    ("17 * 3 = ?", "51"),
    ("84 / 7 = ?", "12"),
    ("100 - 37 = ?", "63"),
    ("13 + 29 = ?", "42"),
    ("7 * 8 = ?", "56"),
    ("9 * 9 = ?", "81"),
    ("1000 - 1 = ?", "999"),
    ("2 to the power of 10 = ?", "1024"),
    ("3 to the power of 4 = ?", "81"),
    ("the square root of 144 = ?", "12"),
    ("half of 90 = ?", "45"),
    ("128 / 4 = ?", "16"),
    ("8471 + 2965 = ?", "11436"),
    ("the cube root of 27 = ?", "3"),
    # Эти два — на границе возможностей модели: один стабильно неверный,
    # второй плывёт от seed к seed. Без них шум sampling не виден.
    ("487 * 23 = ?", "11201"),
    ("the 7th prime number = ?", "17"),
]

SYSTEM = "You are a calculator. Reply with the single number and nothing else."
MAX_TOKENS = 12
GREEDY_RUNS = 3
SAMPLED_RUNS = 3
SAMPLED_TEMP = 0.8
SEED = 0

INT_RE = re.compile(r"-?\d[\d,]*")


def parse_int(text: str) -> int | None:
    """Первое целое в ответе модели (запятые-разделители убираем)."""
    m = INT_RE.search(text)
    if not m:
        return None
    try:
        return int(m.group(0).replace(",", ""))
    except ValueError:
        return None


def ask(model, tok, question: str, *, temp: float, seed: int | None) -> str:
    """Один ответ. seed=None — greedy; иначе фиксируем ГПСЧ MLX перед прогоном."""
    if seed is not None:
        mx.random.seed(seed)
    prompt = tok.apply_chat_template(
        [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        add_generation_prompt=True,
    )
    return generate(
        model,
        tok,
        prompt=prompt,
        max_tokens=MAX_TOKENS,
        sampler=make_sampler(temp=temp),
        verbose=False,
    )


def accuracy(model, tok, *, temp: float, seed: int | None) -> list[bool]:
    out = []
    for prompt, expected in TASKS:
        got = parse_int(ask(model, tok, prompt, temp=temp, seed=seed))
        out.append(got == int(expected))
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Доверительный интервал для доли: Wilson, а не ±z·se (он врёт у краёв)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mde(p: float, n: int, z: float = 1.96) -> float:
    """Минимально различимая дельта для двух независимых выборок по n."""
    return z * math.sqrt(2 * p * (1 - p) / n)


def n_for_delta(p: float, delta: float, z: float = 1.96) -> int:
    """Сколько примеров нужно, чтобы дельта была на грани значимости."""
    return math.ceil(2 * p * (1 - p) * (z / delta) ** 2)


def show(label: str, results: list[bool]) -> float:
    n = len(results)
    k = sum(results)
    lo, hi = wilson(k, n)
    print(
        f"  {label:<18} {k}/{n} = {k / n:>5.1%}   "
        f"95% CI [{lo:>5.1%}, {hi:>5.1%}]"
    )
    return k / n


def main() -> None:
    section("01. Eval: точность, шум и доверительный интервал")
    print(f"  модель: {MODEL_ID}")
    print(f"  примеров: {len(TASKS)}, greedy-прогонов: {GREEDY_RUNS}, "
          f"sampling: {SAMPLED_RUNS} × T={SAMPLED_TEMP}, seed={SEED}")
    model, tok = load_model()

    subsection("A. Greedy: базовый замер")
    greedy = accuracy(model, tok, temp=0.0, seed=None)
    for (question, expected), ok in zip(TASKS, greedy):
        mark = "OK " if ok else "FAIL"
        print(f"  {mark} {question:<30} → {expected:>5}")
    p_greedy = show("greedy", greedy)

    subsection("B. Детерминизм: три greedy-прогона")
    runs = [accuracy(model, tok, temp=0.0, seed=None) for _ in range(GREEDY_RUNS - 1)]
    runs = [greedy, *runs]
    mismatches = sum(len({run[i] for run in runs}) > 1 for i in range(len(TASKS)))
    print(f"  примеров с расхождением между прогонами: {mismatches}/{len(TASKS)}")
    note(
        "Ноль расхождений — не гарантия: детерминизм ломается на батчинге и "
        "на смене порядка операций (см. inference/15)."
    )

    subsection("C. Sampling: сколько шума добавляет схема прогона")
    sampled_runs = [
        accuracy(model, tok, temp=SAMPLED_TEMP, seed=SEED + i)
        for i in range(SAMPLED_RUNS)
    ]
    for i, res in enumerate(sampled_runs):
        show(f"T={SAMPLED_TEMP} seed={SEED + i}", res)
    per_item = [
        sum(run[i] for run in sampled_runs) / len(sampled_runs)
        for i in range(len(TASKS))
    ]
    unstable = [i for i, v in enumerate(per_item) if 0 < v < 1]
    print(f"  нестабильных примеров (иногда верно, иногда нет): "
          f"{len(unstable)}/{len(TASKS)}")
    if unstable:
        for i in unstable:
            print(f"    пример {i}: {per_item[i]:.0%} верных из "
                  f"{len(sampled_runs)} прогонов — «{TASKS[i][0]}»")

    rates = [sum(r) / len(r) for r in sampled_runs]
    spread = max(rates) - min(rates)
    print(f"  размах accuracy по прогонам: {spread:.1%} "
          f"(от {min(rates):.0%} до {max(rates):.0%})")

    subsection("D. Что может различить такой набор")
    n = len(TASKS)
    lo, hi = wilson(sum(greedy), n)
    print(f"  greedy accuracy: {p_greedy:.1%}, 95% CI [{lo:.1%}, {hi:.1%}] "
          f"— ширина {(hi - lo) * 100:.0f} п.п.")
    print(f"  минимально различимая дельта при n={n}, p={p_greedy:.2f}: "
          f"±{mde(p_greedy, n):.1%}")
    note(
        "Дельта меньше этой величины на таком наборе неотличима от шума — "
        "ни в плюс, ни в минус."
    )

    subsection("E. Сколько примеров нужно")
    print(f"  {'дельта':>8}{'примеров':>10}")
    for delta in (0.02, 0.05, 0.10):
        print(f"  {delta:>8.0%}{n_for_delta(p_greedy, delta):>10}")
    note(
        "Оценка для одной выборки; сравнение двух конфигураций на тех же "
        "примерах требует ещё и поправки на парность (одни и те же вопросы)."
    )

    section("Вывод")
    note("Accuracy — оценка с интервалом; без интервала это не результат.")
    note("Смена схемы прогона (greedy → sampling) даёт сдвиг больше, "
         "чем многие «улучшения» модели.")
    note("Перед сравнением: фиксированный набор, фиксированный seed, "
         "парные прогоны, дельта против MDE.")


if __name__ == "__main__":
    main()
