# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 1 (observability): TTFT, TPOT и то, что видит сервер.

Идея
----
Один запрос к LLM описывается двумя числами, а не «латентностью»:

* **TTFT** (time to first token) — время до первого токена. Это в основном
  prefill: он растёт с длиной промпта и почти не зависит от длины ответа.
* **TPOT** (time per output token) — время между последующими токенами. Это
  decode-шаг: он почти константен, пока контекст не вырос.

Из них собирается throughput = 1/TPOT на поток, и именно их vllm-mlx
публикует в ``/metrics`` (``vllm_mlx/metrics.py``: ``InferenceTracker``,
``MetricsCollector.observe_ttft``, ``observe_inference``).

Ловушка измерения: MLX считает лениво и пайплайнит шаги (``mx.async_eval``),
поэтому «время до yield» без ``mx.synchronize()`` занижает TTFT, а последний
токен ещё не посчитан. Здесь мы синхронизируемся на каждом токене и смотрим,
на сколько это меняет цифры.

Что измеряем:
  1. TTFT при разной длине промпта (32 / 128 / 512 токенов);
  2. TPOT на 64 токенах и сравнение со счётчиком ``generation_tps`` из mlx-lm;
  3. разброс по 8 повторам: p50 / p95 / max для TTFT и TPOT;
  4. какие из этих чисел вообще отдаёт vllm-mlx и под какими именами.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/05 observability engineering/01_ttft_tpot.py"
"""
from __future__ import annotations

import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from common import MODEL_ID, base_token_ids, load_model, note, section, subsection

PROMPT_LENS = (32, 128, 512)
GEN_TOKENS = 64
REPEATS = 8


def percentile(values: list[float], q: float) -> float:
    """Перцентиль с линейной интерполяцией (q в [0, 1])."""
    if not values:
        return float("nan")
    data = sorted(values)
    if len(data) == 1:
        return data[0]
    pos = q * (len(data) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(data) - 1)
    return data[lo] + (data[hi] - data[lo]) * (pos - lo)


def measure(model, tok, *, prompt_tokens: int, max_tokens: int) -> dict:
    """Один стриминговый запрос: TTFT, TPOT и счётчики самого mlx-lm."""
    prompt = base_token_ids(tok, prompt_tokens)
    sampler = make_sampler(temp=0.0)
    t0 = time.perf_counter()
    stamps: list[float] = []
    last = None
    for resp in stream_generate(
        model, tok, prompt, max_tokens=max_tokens, sampler=sampler
    ):
        mx.synchronize()  # цена честной синхронизации — тоже часть замера
        stamps.append(time.perf_counter())
        last = resp
    ttft = stamps[0] - t0 if stamps else float("nan")
    tpot = (
        (stamps[-1] - stamps[0]) / (len(stamps) - 1)
        if len(stamps) > 1
        else float("nan")
    )
    return {
        "ttft": ttft,
        "tpot": tpot,
        "tokens": len(stamps),
        "reported_tps": getattr(last, "generation_tps", float("nan")),
        "prompt_tps": getattr(last, "prompt_tps", float("nan")),
        "peak_gb": getattr(last, "peak_memory", float("nan")),
    }


def show_spread(label: str, values: list[float], scale: float = 1000.0,
                unit: str = "мс") -> None:
    med = statistics.median(values)
    print(
        f"  {label:<12} p50 {med * scale:>7.2f} {unit}"
        f"   p95 {percentile(values, 0.95) * scale:>7.2f} {unit}"
        f"   max {max(values) * scale:>7.2f} {unit}"
    )


def main() -> None:
    section("01. Observability: TTFT, TPOT и разброс")
    print(f"  модель: {MODEL_ID}, повторов: {REPEATS}")
    model, tok = load_model()
    # Первый запрос процесса — «холодный»: ядра Metal компилируются здесь.
    cold = measure(model, tok, prompt_tokens=PROMPT_LENS[1], max_tokens=8)

    subsection("A. TTFT растёт с промптом, TPOT — почти нет")
    print(f"  {'промпт':>7}{'TTFT, мс':>10}{'TPOT, мс':>10}"
          f"{'tokens/s':>10}{'prompt t/s':>12}")
    first = None
    for n_prompt in PROMPT_LENS:
        r = measure(model, tok, prompt_tokens=n_prompt, max_tokens=GEN_TOKENS)
        if first is None:
            first = r
        print(
            f"  {n_prompt:>7}{r['ttft'] * 1000:>10.1f}"
            f"{r['tpot'] * 1000:>10.2f}"
            f"{1 / r['tpot']:>10.0f}{r['prompt_tps']:>12.0f}"
        )
    note(
        "TTFT вырос в разы, TPOT — в пределах шума: prefill масштабируется "
        "с длиной входа, decode-шаг — с размером модели."
    )

    subsection("B. Два способа посчитать TPOT")
    r = measure(model, tok, prompt_tokens=PROMPT_LENS[0], max_tokens=GEN_TOKENS)
    print(f"  из стрима (с mx.synchronize на токен): {r['tpot'] * 1000:>7.2f} мс"
          f"  → {1 / r['tpot']:>5.0f} tokens/s")
    print(f"  счётчик mlx-lm (generation_tps):       "
          f"{1 / r['reported_tps'] * 1000:>7.2f} мс"
          f"  → {r['reported_tps']:>5.0f} tokens/s")
    note(
        "Стрим включает detokenization и Python-оверхед на каждый токен, "
        "поэтому он не быстрее счётчика модели. Разница — цена обвязки, "
        "а не цена forward."
    )

    subsection("C. Разброс: почему нужен p95, а не среднее")
    runs = [
        measure(model, tok, prompt_tokens=PROMPT_LENS[1], max_tokens=GEN_TOKENS)
        for _ in range(REPEATS)
    ]
    rest = [x["ttft"] for x in runs]
    med = statistics.median(rest)
    print(f"  холодный TTFT {cold['ttft'] * 1000:>7.2f} мс, "
          f"медиана прогретых {med * 1000:>7.2f} мс "
          f"({cold['ttft'] / med:.2f}×)")
    show_spread("TTFT", rest)
    show_spread("TPOT", [x["tpot"] for x in runs])
    print(f"  среднее TTFT {statistics.mean(rest) * 1000:>7.2f} мс, "
          f"ст.отклонение {statistics.stdev(rest) * 1000:>6.2f} мс")
    note(
        "На простое разброс маленький, и p95 ≈ p50: хвост — свойство не "
        "железа, а нагрузки (очередь и батч — в следующих лабах). Но "
        "холодный прогон в SLO не должен попадать: прогреваем перед замером."
    )

    subsection("D. Что из этого отдаёт vllm-mlx")
    names, methods = metrics_names()
    for name in names:
        print(f"  · {name}")
    for line in methods:
        print(f"  · {line}")
    note(
        "TTFT и TPOT считаются на стороне сервера из тех же двух отметок "
        "времени; /metrics отдаёт их гистограммами, а не одним числом."
    )

    section("Вывод")
    note("Латентность — это пара (TTFT, TPOT), плюс отдельно throughput.")
    note("TTFT = префилл (длина входа), TPOT = decode-шаг (размер модели).")
    note("Любой замер без p95 и без прогрева — оптимистичная оценка.")

def metrics_names() -> tuple[list[str], list[str]]:
    """Имена метрик и методов, которые реально публикует установленный vllm-mlx."""
    try:
        from vllm_mlx import metrics
    except Exception as exc:  # noqa: BLE001 — пакета может не быть
        return ([f"vllm_mlx.metrics недоступен: {exc}"], [])

    names: list[str] = []
    src = Path(metrics.__file__).read_text(encoding="utf-8", errors="ignore")
    for m in re.finditer(r"\"(vllm[_a-z:]*)\"", src):
        if m.group(1) not in names:
            names.append(m.group(1))
    methods = []
    for cls in ("InferenceTracker", "MetricsCollector"):
        obj = getattr(metrics, cls, None)
        if obj is None:
            continue
        hooks = [
            n for n in dir(obj)
            if n.startswith(("observe_", "track_", "render_"))
        ]
        methods.append(f"{cls}: " + ", ".join(hooks))
    return (names[:10], methods)


if __name__ == "__main__":
    main()
