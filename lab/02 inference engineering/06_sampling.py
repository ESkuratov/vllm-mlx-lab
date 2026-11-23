# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 6 (бонус): как работают сэмплинг и температура.

Продолжение lab/logits.py: модель выдаёт «сырые» логиты для следующего токена,
а как из них выбрать токен — отдельный слой (logits processors):

* температура — делит логиты на T (T<1 делает распределение «острее»,
  T>1 — «площе»);
* top-k — оставляем только k самых вероятных токенов;
* top-p (nucleus) — оставляем минимальный набор с накопленной вероятностью ≥ p;
* greedy — всегда берём argmax.

Что показываем: как эти фильтры меняют распределение, энтропию и разнообразие
сгенерированных токенов.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/06_sampling.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import MODEL_ID, load_model, note, section, subsection

PROMPT = "The capital of France is"


def entropy(probs: mx.array) -> float:
    # считаем во float32: во float16 log(1e-12) превращается в -inf и даёт NaN
    p = probs.astype(mx.float32)
    return max(0.0, float(-mx.sum(p * mx.log(mx.maximum(p, 1e-9)))))


def topk(probs: mx.array, k: int) -> mx.array:
    idx = mx.argsort(-probs)[:k]
    masked = mx.zeros_like(probs)
    masked[idx] = probs[idx]
    return masked / mx.sum(masked)


def topp(probs: mx.array, p: float):
    order = mx.argsort(-probs)
    sorted_p = probs[order]
    csum = mx.cumsum(sorted_p)
    ge = csum >= p
    if bool(mx.any(ge)):
        keep = int(mx.argmax(ge)) + 1  # минимальный набор c суммой ≥ p
    else:
        keep = sorted_p.shape[0]
    core = sorted_p[:keep] / mx.sum(sorted_p[:keep])
    return core, keep


def show_top(tok, probs: mx.array, n: int = 8) -> None:
    idx = mx.argsort(-probs)[:n].tolist()
    cum = 0.0
    for rank, t in enumerate(idx, 1):
        prob = float(probs[t])
        cum += prob
        print(f"    {rank:>2}. {tok.decode([t])!r:<16} p={prob:>7.4f}  ∑={cum:>6.3f}")


def main() -> None:
    section("6. Сэмплинг: температура, top-k, top-p")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    ids = mx.array(tok.encode(PROMPT))
    logits = model(ids[None])[0, -1]
    mx.eval(logits)

    # -------------------------------------------------------- сырое
    subsection(f"A. Сырое распределение («{PROMPT}»)")
    base = mx.softmax(logits)
    show_top(tok, base)
    print(f"    энтропия: {entropy(base):.3f} нат")

    # -------------------------------------------------------- температура
    subsection("B. Температура")
    print(f"  {'T':>5}{'энтропия':>11}{'топ-1 p':>10}{'топ-5 p':>10}")
    for T in (0.5, 0.8, 1.0, 1.5, 2.0):
        p = mx.softmax(logits / T)
        top1 = float(mx.max(p))
        top5 = float(mx.sum(mx.sort(p)[::-1][:5]))
        print(f"  {T:>5}{entropy(p):>11.3f}{top1:>10.3f}{top5:>10.3f}")
    note(
        "T<1 заостряет распределение (модель увереннее, меньше разнообразия), "
        "T>1 размывает — растёт энтропия. vllm-mlx: --default-temperature."
    )

    # ------------------------------------------------------------- top-k
    subsection("C. Top-k")
    print(f"  {'k':>5}{'масса ядра':>13}{'энтропия':>11}")
    for k in (1, 5, 20, 100):
        kept_mass = float(mx.sum(mx.sort(base)[::-1][:k]))
        p = topk(base, k)
        print(f"  {k:>5}{kept_mass:>12.1%}{entropy(p):>11.3f}")
    note("Top-k жёстко режет хвост: при малом k теряется вариативность.")

    # ------------------------------------------------------------- top-p
    subsection("D. Top-p (nucleus)")
    print(f"  {'p':>5}{'токенов в ядре':>17}{'энтропия':>11}")
    for p_cut in (0.5, 0.9, 0.95, 0.99):
        core, kept = topp(base, p_cut)
        print(f"  {p_cut:>5}{kept:>17}{entropy(core):>11.3f}")
    note(
        "Nucleus сам подбирает размер: при уверенной модели ядро крошечное, "
        "при неопределённой — широкое."
    )

    # --------------------------------------------------- greedy vs sample
    subsection("E. Greedy vs случайная выборка (200 токенов, T=0.8)")
    greedy = int(mx.argmax(logits))
    print(f"  greedy: {tok.decode([greedy])!r}")
    p = mx.softmax(logits / 0.8)
    mx.random.seed(0)
    samples = mx.random.categorical(mx.log(p + 1e-12), shape=(200,))
    mx.eval(samples)
    counts = {}
    for t in samples.tolist():
        counts[t] = counts.get(t, 0) + 1
    for t, c in sorted(counts.items(), key=lambda kv: -kv[1])[:8]:
        print(f"    {tok.decode([t])!r:<16} {c:>4} раз ({c / 200:>5.1%})")
    print(f"  уникальных токенов в 200 выборках: {len(counts)}")
    note(
        "Даже при уверенной модели случайная выборка иногда даёт другие токены — "
        "именно это даёт «творчество» и нестабильность. Greedy всегда один и тот же."
    )

    section("Вывод")
    note("Логиты → (температура) → (top-k/top-p) → softmax → выбор токена.")
    note("Низкая T + greedy = детерминизм и код; высокая T = разнообразие.")
    note("Во vllm-mlx параметры по умолчанию: --default-temperature, --default-top-p, --default-top-k, --default-min-p.")


if __name__ == "__main__":
    main()
