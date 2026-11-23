# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 03: MoE (mixture-of-experts) — роутинг и экономика.

Идея
----
Dense-модель на каждый токен считает все параметры. MoE заменяет один большой
FFN на несколько «экспертов», и для каждого токена роутер выбирает top-k из них.
Итого: параметров — много (ёмкость), а вычислений на токен — мало (только k
экспертов). Это называют «разреженной» моделью: sparse по активациям.

Тонкость — балансировка: без неё роутер сваливается на нескольких экспертов,
остальные простаивают.

Здесь строим маленький MoE из mx и измеряем реальную механику роутинга;
реальная модель не нужна (и её архитектура — тоже MoE, но размеры другие).

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/03_moe.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn

from common import note, section, subsection

HIDDEN = 256
FFN = 512
EXPERTS = 8


def expert_params(hidden=HIDDEN, ffn=FFN) -> int:
    return 2 * hidden * ffn  # два Linear без bias


class MoE(nn.Module):
    def __init__(self, hidden=HIDDEN, ffn=FFN, experts=EXPERTS):
        super().__init__()
        self.router = nn.Linear(hidden, experts, bias=False)
        self.experts = [
            (nn.Linear(hidden, ffn, bias=False), nn.Linear(ffn, hidden, bias=False))
            for _ in range(experts)
        ]
        self.top_k = 2

    def __call__(self, x):
        logits = self.router(x)  # (N, E)
        probs = mx.softmax(logits, axis=-1)
        top = mx.argsort(-probs, axis=-1)[:, : self.top_k]
        gates = mx.take_along_axis(probs, top, axis=-1)
        gates = gates / mx.sum(gates, axis=-1, keepdims=True)

        out = mx.zeros_like(x)
        for e, (up, down) in enumerate(self.experts):
            mask = (top == e).any(axis=-1)  # (N,) — попал ли токен к эксперту e
            if not bool(mx.any(mask)):
                continue
            h = mx.maximum(up(x), 0.0)
            y = down(h)
            # гейт для тех, у кого эксперт e в топ-k
            g = mx.sum(mx.where(top == e, gates, 0.0), axis=-1)
            out = out + y * g[:, None] * mask[:, None]
        return out, probs, top


def utilization(top) -> list[int]:
    counts = [0] * EXPERTS
    for row in top.tolist():
        for e in row:
            counts[e] += 1
    return counts


def entropy(p: mx.array) -> float:
    p = p.astype(mx.float32)
    return max(0.0, float(-mx.sum(p * mx.log(mx.maximum(p, 1e-9)))))


def main() -> None:
    section("03. MoE: роутинг, активные параметры, балансировка")

    # --------------------------------------------------- экономика
    subsection("A. Параметры: всего много, активных мало")
    dense_ffn = FFN * EXPERTS  # dense с тем же «бюджетом параметров»
    total = EXPERTS * expert_params()
    active = 2 * expert_params()  # top-2
    router = HIDDEN * EXPERTS
    print(f"  экспертов: {EXPERTS}, top-k: 2, hidden={HIDDEN}, ffn={FFN}")
    print(f"  параметров MoE всего:    {total + router:>10,}")
    print(f"  активных на токен:       {active + router:>10,}  "
          f"({(active + router) / (total + router):.0%} от всех)")
    print(f"  dense с тем же бюджетом: {2 * HIDDEN * dense_ffn:>10,} активных на токен")
    print(f"  → на каждый токен MoE считает в ~{EXPERTS / 2:.0f}× меньше FFN-параметров")
    note(
        "Именно поэтому у реальных MoE такие цифры: Mixtral 8×7B — ~47B всего, "
        "~13B активных; Qwen3-30B-A3B — 30B всего, ~3B активных."
    )

    # --------------------------------------------------- роутинг
    subsection("B. Роутинг: как эксперты распределяются")
    mx.random.seed(0)
    moe = MoE()
    x = mx.random.normal((256, HIDDEN))
    out, probs, top = moe(x)
    mx.eval(out, probs, top)

    counts = utilization(top)
    print(f"  распределение токенов по экспертам (из {len(x)} токенов, по 2 выбора):")
    for e, c in enumerate(counts):
        bar = "█" * (c // 4)
        print(f"    эксперт {e}: {c:>4} {bar}")
    print(f"  энтропия среднего распределения роутера: {entropy(mx.mean(probs, axis=0)):.3f} "
          f"(максимум ln {EXPERTS} = {float(mx.log(mx.array(EXPERTS))):.3f})")

    # --------------------------------------------------- коллапс
    subsection("C. Коллапс роутера и балансирующий лосс")
    collapsed = mx.zeros((len(x), EXPERTS))
    collapsed = collapsed.at[:, 0].add(6.0)  # сильный перекос на эксперта 0
    cprobs = mx.softmax(collapsed, axis=-1)
    ctop = mx.argsort(-cprobs, axis=-1)[:, :2]
    ccounts = utilization(ctop)
    print(f"  перекошенный роутер: {ccounts}")
    print(f"  энтропия: {entropy(mx.mean(cprobs, axis=0)):.3f}")

    def aux_loss(prob: mx.array) -> float:
        """Стандартный load-balancing loss: E · Σ f_e · P_e."""
        import numpy as np

        p = np.array(prob)
        f = (p > 0).mean(axis=0)  # доля токенов, выбравших эксперта
        P = p.mean(axis=0)        # средняя вероятность эксперта
        return float(EXPERTS * (f * P).sum())

    def onehot(top) -> mx.array:
        import numpy as np

        t = np.array(top)
        m = np.zeros((len(t), EXPERTS), dtype=np.float32)
        for i, row in enumerate(t):
            for e in row:
                m[i, e] = 1.0
        m /= m.sum(axis=1, keepdims=True)
        return mx.array(m)

    print(f"  aux loss (сбалансированный): {aux_loss(onehot(top)):.3f}")
    print(f"  aux loss (коллапс):          {aux_loss(onehot(ctop)):.3f}")
    note(
        "Балансирующий лосс штрафует перекос; без него роутер переиспользует "
        "пару экспертов, а ёмкость остальных пропадает."
    )

    # --------------------------------------------------- стоимость памяти
    subsection("D. Что это значит для инференса")
    note("Активных вычислений мало → decode быстрый, как у маленькой модели.")
    note("Но все эксперты надо держать в памяти → модель занимает как большая.")
    note("Итог: скорость «3B», а знания — «30B» (примерно).")
    note("Во vllm-mlx MoE-модели поддерживаются как обычные MLX-чекпойнты.")

    section("Вывод")
    note("MoE ≠ быстрее по памяти, но быстрее по вычислениям на токен.")
    note("Ключ к качеству — роутинг и балансировка экспертов.")


if __name__ == "__main__":
    main()
