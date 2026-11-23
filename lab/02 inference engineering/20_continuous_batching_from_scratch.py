# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 20: continuous batching и chunked prefill с нуля.

Идея
----
Continuous batching — это не «взять batch_generate из библиотеки», а
конкретная политика над слотами и кэшами:

* есть N decode-слотов;
* один prefill-слот режет длинный промпт на чанки;
* когда prefill закончился, запрос переезжает в свободный decode-слот;
* когда запрос завершился, слот сразу занимает следующий;
* prefill и decode чередуются, чтобы длинный промпт не блокировал всех.

В этой лаборатории мы пишем дискретный симулятор такого планировщика.
Модель не загружается: стоимость шага задаётся CostModel. По умолчанию
это разумные константы; `LAB_REAL=1` калибрует их по реальной модели.

Симулятор не делает настоящий batched forward — он показывает **политику**
и метрики: TTFT, TPOT, completion latency, utilization, throughput.
Реальный continuous batching в vllm-mlx/mlx-lm устроен так же по смыслу,
но с настоящими кэшами и ядрами.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/20_continuous_batching_from_scratch.py"

Калибровка по реальной модели (медленнее):
    LAB_REAL=1 python "lab/02 inference engineering/20_continuous_batching_from_scratch.py"
"""
from __future__ import annotations

import math
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import base_token_ids, load_model, note, section, subsection


# --------------------------------------------------------------------------- #
# Модель стоимости
# --------------------------------------------------------------------------- #


@dataclass
class CostModel:
    """Упрощённая стоимость шага в миллисекундах.

    prefill = overhead + per_token * tokens   (compute-bound)
    decode  = base + per_slot * active_slots  (веса читаются один раз на батч)
    """

    prefill_overhead_ms: float = 0.30
    prefill_per_token_ms: float = 0.080
    decode_base_ms: float = 2.00
    decode_per_slot_ms: float = 0.12

    def prefill(self, tokens: int) -> float:
        return (
            self.prefill_overhead_ms + self.prefill_per_token_ms * max(0, tokens)
        ) / 1000.0

    def decode(self, active_slots: int) -> float:
        if active_slots <= 0:
            return 0.0
        return (
            self.decode_base_ms + self.decode_per_slot_ms * active_slots
        ) / 1000.0


def calibrate_with_real_model() -> CostModel:
    """Измерить prefill/decode на реальной модели и подставить в CostModel."""
    from mlx_lm.models.cache import make_prompt_cache

    model, tok = load_model()
    print("  калибровка по реальной модели...")
    ids = base_token_ids(tok, 128)

    def eval_prefill():
        logits = model(ids[None], cache=make_prompt_cache(model))
        mx.eval(logits)

    eval_prefill()
    t_prefill = _median_time(eval_prefill, repeats=3)

    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    token = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    steps = []
    for _ in range(8):
        t0 = time.perf_counter()
        logits = model(token, cache=cache)
        mx.eval(logits)
        steps.append(time.perf_counter() - t0)
        token = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    t_decode = statistics.median(steps[2:])

    # Предположение: в батче доминирует чтение весов, а на слот приходится
    # небольшая добавка. Это именно модель, а не измерение batch-шага.
    return CostModel(
        prefill_overhead_ms=0.30,
        prefill_per_token_ms=t_prefill * 1000.0 / 128.0,
        decode_base_ms=t_decode * 1000.0,
        decode_per_slot_ms=0.10 * t_decode * 1000.0,
    )


def _median_time(fn, repeats: int = 3) -> float:
    fn()
    mx.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


# --------------------------------------------------------------------------- #
# Запросы и результаты
# --------------------------------------------------------------------------- #


@dataclass
class Request:
    rid: int
    arrival: float
    prompt_len: int
    output_len: int
    priority: int = 0
    first_token: float | None = None
    generated: int = 0
    token_times: list[float] = field(default_factory=list)


@dataclass
class Slot:
    request: Request
    generated: int = 1


@dataclass
class PrefillState:
    request: Request
    processed: int = 0


@dataclass
class Result:
    request: Request
    first_token: float
    finish: float

    @property
    def ttft(self) -> float:
        return self.first_token - self.request.arrival

    @property
    def latency(self) -> float:
        return self.finish - self.request.arrival

    @property
    def tpot(self) -> float:
        times = self.request.token_times
        if len(times) < 2:
            return (self.finish - self.first_token) / max(
                1, self.request.output_len - 1
            )
        return statistics.mean(b - a for a, b in zip(times, times[1:]))

    @property
    def gap_p95(self) -> float:
        times = self.request.token_times
        if len(times) < 2:
            return 0.0
        return _p95([b - a for a, b in zip(times, times[1:])])

    @property
    def gap_max(self) -> float:
        times = self.request.token_times
        if len(times) < 2:
            return 0.0
        return max(b - a for a, b in zip(times, times[1:]))


@dataclass
class RunStats:
    results: list[Result] = field(default_factory=list)
    makespan: float = 0.0
    decode_slot_time: float = 0.0
    prefill_time: float = 0.0
    max_slots: int = 1

    def summary(self) -> dict[str, float]:
        if not self.results:
            return {}
        ttfts = [r.ttft for r in self.results]
        latencies = [r.latency for r in self.results]
        tpots = [r.tpot for r in self.results]
        gaps = [r.gap_p95 for r in self.results]
        max_gaps = [r.gap_max for r in self.results]
        tokens = sum(r.request.output_len for r in self.results)
        return {
            "makespan": self.makespan,
            "tokens": tokens,
            "tokens_per_s": tokens / self.makespan if self.makespan > 0 else 0.0,
            "ttft_mean": statistics.mean(ttfts),
            "ttft_p95": _p95(ttfts),
            "latency_mean": statistics.mean(latencies),
            "latency_p95": _p95(latencies),
            "latency_max": max(latencies),
            "tpot_mean": statistics.mean(tpots),
            "gap_p95": _p95(gaps),
            "gap_max": max(max_gaps),
            "decode_util": (
                self.decode_slot_time / (self.max_slots * self.makespan)
                if self.makespan > 0
                else 0.0
            ),
            "prefill_busy": (
                self.prefill_time / self.makespan if self.makespan > 0 else 0.0
            ),
        }


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


# --------------------------------------------------------------------------- #
# Планировщик
# --------------------------------------------------------------------------- #


def _pick(waiting: list[Request], policy: str) -> Request:
    if policy == "fcfs":
        return min(waiting, key=lambda r: (r.arrival, r.rid))
    if policy == "sjf":
        return min(waiting, key=lambda r: (r.prompt_len + r.output_len, r.arrival))
    if policy == "priority":
        return min(waiting, key=lambda r: (-r.priority, r.arrival))
    raise ValueError(f"неизвестная политика: {policy}")


def _finish(request: Request, now: float, stats: RunStats) -> None:
    if request.first_token is None:
        raise RuntimeError("запрос завершился до первого токена")
    stats.results.append(Result(request, request.first_token, now))
    if not request.token_times or request.token_times[-1] != now:
        request.token_times.append(now)


def run_continuous(
    requests: list[Request],
    cost: CostModel,
    *,
    max_slots: int = 4,
    prefill_chunk: int = 128,
    policy: str = "fcfs",
) -> RunStats:
    """Continuous batching: один prefill + N decode-слотов."""
    if max_slots <= 0 or prefill_chunk <= 0:
        raise ValueError("max_slots и prefill_chunk должны быть положительными")

    pending = sorted(requests, key=lambda r: (r.arrival, r.rid))
    waiting: list[Request] = []
    ready: list[Request] = []
    running: list[Slot] = []
    prefill: PrefillState | None = None
    stats = RunStats(max_slots=max_slots)
    now = 0.0
    index = 0

    while index < len(pending) or waiting or ready or running or prefill is not None:
        # 1. Принять всё, что уже пришло.
        while index < len(pending) and pending[index].arrival <= now + 1e-12:
            waiting.append(pending[index])
            index += 1

        progressed = False

        # 2. Освободившиеся слоты занимаем готовыми после prefill запросами.
        while ready and len(running) < max_slots:
            request = ready.pop(0)
            running.append(Slot(request, request.generated))
            progressed = True

        # 3. Начать prefill можно, только если нет готового запроса,
        #    ожидающего слот. Это моделирует один pending-prefill слот.
        if prefill is None and waiting and not ready:
            request = _pick(waiting, policy)
            waiting.remove(request)
            prefill = PrefillState(request)
            progressed = True

        # 4. Один чанк prefill.
        if prefill is not None:
            request = prefill.request
            take = min(prefill_chunk, max(0, request.prompt_len - prefill.processed))
            chunk_cost = cost.prefill(take)
            now += chunk_cost
            stats.prefill_time += chunk_cost
            prefill.processed += take
            progressed = True

            if prefill.processed >= request.prompt_len:
                request.first_token = now
                request.generated = 1
                request.token_times.append(now)
                prefill = None
                if request.output_len <= 1:
                    _finish(request, now, stats)
                else:
                    ready.append(request)

        # 5. После завершения prefill слот мог освободиться.
        while ready and len(running) < max_slots:
            request = ready.pop(0)
            running.append(Slot(request, request.generated))
            progressed = True

        # 6. Один общий decode-шаг для всех активных слотов.
        if running:
            step_cost = cost.decode(len(running))
            now += step_cost
            stats.decode_slot_time += step_cost * len(running)
            progressed = True
            finished: list[Slot] = []
            for slot in running:
                slot.generated += 1
                slot.request.token_times.append(now)
                if slot.generated >= slot.request.output_len:
                    _finish(slot.request, now, stats)
                    finished.append(slot)
            for slot in finished:
                running.remove(slot)

        # 7. Если работы нет, прыгаем к следующему приходу.
        if not progressed:
            if index < len(pending):
                now = max(now, pending[index].arrival)
            elif not (waiting or ready or running or prefill is not None):
                break
            else:
                raise RuntimeError("симулятор застрял")

    stats.makespan = max((r.finish for r in stats.results), default=0.0) - min(
        (r.request.arrival for r in stats.results), default=0.0
    )
    return stats


def run_static(
    requests: list[Request],
    cost: CostModel,
    *,
    max_slots: int = 4,
) -> RunStats:
    """Static batching: группа целиком prefill, затем decode до конца."""
    pending = sorted(requests, key=lambda r: (r.arrival, r.rid))
    stats = RunStats(max_slots=max_slots)
    now = 0.0

    for start in range(0, len(pending), max_slots):
        group = pending[start : start + max_slots]
        # Группа стартует, когда пришёл последний её участник.
        now = max(now, max(r.arrival for r in group))

        # Prefill всей группы одним батчем (в реальности — batched prefill).
        total_prompt = sum(r.prompt_len for r in group)
        prefill_cost = cost.prefill(total_prompt)
        now += prefill_cost
        stats.prefill_time += prefill_cost
        for request in group:
            request.first_token = now
            request.generated = 1
            request.token_times.append(now)

        active = [r for r in group if r.output_len > 1]
        for request in group:
            if request.output_len <= 1:
                _finish(request, request.first_token, stats)

        while active:
            step_cost = cost.decode(len(active))
            now += step_cost
            stats.decode_slot_time += step_cost * len(active)
            finished = []
            for request in active:
                request.generated += 1
                request.token_times.append(now)
                if request.generated >= request.output_len:
                    _finish(request, now, stats)
                    finished.append(request)
            for request in finished:
                active.remove(request)

    stats.makespan = max((r.finish for r in stats.results), default=0.0) - min(
        (r.request.arrival for r in stats.results), default=0.0
    )
    return stats


# --------------------------------------------------------------------------- #
# Workload и печать
# --------------------------------------------------------------------------- #


def make_workload() -> list[Request]:
    """Детерминированный неравномерный workload."""
    prompt_lens = [128, 512, 256, 1024, 128, 768, 256, 512, 384, 128, 896, 256]
    output_lens = [64, 32, 128, 64, 32, 96, 128, 64, 32, 96, 64, 128]
    arrivals = [0.00, 0.00, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.35, 0.45, 0.60, 0.70]
    priorities = [1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0]
    return [
        Request(i, arrivals[i], prompt_lens[i], output_lens[i], priorities[i])
        for i in range(len(prompt_lens))
    ]


def reset_workload(requests: list[Request]) -> list[Request]:
    return [
        Request(r.rid, r.arrival, r.prompt_len, r.output_len, r.priority)
        for r in requests
    ]


def _row(name: str, s: dict[str, float]) -> None:
    print(
        f"  {name:<18} {s['tokens_per_s']:>8.1f} {s['ttft_p95'] * 1e3:>9.1f} "
        f"{s['latency_p95'] * 1e3:>10.1f} {s['tpot_mean'] * 1e3:>8.1f} "
        f"{s['gap_p95'] * 1e3:>8.1f} {s['decode_util']:>8.1%} "
        f"{s['prefill_busy']:>8.1%}"
    )


def _header() -> None:
    print(
        f"  {'политика':<18} {'tok/s':>8} {'TTFT p95':>9} {'lat p95':>10} "
        f"{'TPOT':>8} {'gap p95':>8} {'decode':>8} {'prefill':>8}"
    )
    print(
        f"  {'':<18} {'':>8} {'ms':>9} {'ms':>10} {'ms':>8} "
        f"{'ms':>8} {'util':>8} {'busy':>8}"
    )


def compare_static_continuous(workload: list[Request], cost: CostModel) -> None:
    subsection("A. Static vs continuous batching")
    static = run_static(reset_workload(workload), cost, max_slots=4)
    continuous = run_continuous(reset_workload(workload), cost, max_slots=4)
    _header()
    _row("static", static.summary())
    _row("continuous", continuous.summary())
    note(
        "Static ждёт самую длинную последовательность в группе: короткие "
        "запросы держат слот, но ничего не делают. Continuous отдаёт слот "
        "следующему сразу."
    )
    note(
        "В continuous prefill одного запроса чередуется с decode-шагами "
        "других, поэтому длинный промпт не блокирует весь батч."
    )


def compare_chunks(workload: list[Request], cost: CostModel) -> None:
    subsection("B. Chunked prefill: размер чанка")
    _header()
    for chunk in (32, 128, 512, 1024):
        stats = run_continuous(
            reset_workload(workload), cost, max_slots=4, prefill_chunk=chunk
        )
        _row(f"chunk={chunk}", stats.summary())
    note(
        "Мелкий чанк уменьшает gap между decode-шагами других запросов: "
        "между кусками prefill они успевают сгенерировать токен. Плата — "
        "больше dispatch-переходов и позже первый токен у самого "
        "prefill-запроса. Крупный чанк быстрее завершает prefill, но "
        "дольше блокирует decode соседей."
    )
    note(
        "В этой модели стоимость чанка = overhead + per_token × tokens; "
        "поэтому виден компромисс, а не бесплатный выигрыш."
    )


def compare_policies(workload: list[Request], cost: CostModel) -> None:
    subsection("C. Политика выбора из очереди")
    _header()
    rows = []
    for policy in ("fcfs", "sjf", "priority"):
        stats = run_continuous(
            reset_workload(workload), cost, max_slots=4, prefill_chunk=128, policy=policy
        )
        _row(policy, stats.summary())
        high = _p95(
            [r.ttft for r in stats.results if r.request.priority == 1]
        )
        low = _p95(
            [r.ttft for r in stats.results if r.request.priority == 0]
        )
        rows.append((policy, high, low, stats.summary()["latency_mean"]))
    for policy, high, low, mean_lat in rows:
        print(
            f"      {policy:<9} TTFT p95: priority=1 {high * 1e3:6.1f} ms, "
            f"priority=0 {low * 1e3:6.1f} ms; mean latency {mean_lat * 1e3:6.1f} ms"
        )
    note(
        "SJF уменьшает среднюю латентность, но ухудшает p95: длинные запросы "
        "могут голодать. Priority улучшает TTFT выбранного класса ценой "
        "остальных. FCFS справедлив по приходу, но не по стоимости."
    )


def show_workload(workload: list[Request]) -> None:
    subsection("Workload")
    print(
        f"  запросов: {len(workload)}, суммарный prompt: "
        f"{sum(r.prompt_len for r in workload)} токенов, "
        f"output: {sum(r.output_len for r in workload)} токенов"
    )
    print(
        "  длины prompt: "
        + ", ".join(str(r.prompt_len) for r in workload)
    )
    print(
        "  длины output: "
        + ", ".join(str(r.output_len) for r in workload)
    )


def main() -> None:
    section("20. Continuous batching и chunked prefill с нуля")
    real = os.environ.get("LAB_REAL") == "1"
    if real:
        cost = calibrate_with_real_model()
    else:
        cost = CostModel()
    print(
        f"  cost model: prefill={cost.prefill_overhead_ms:.2f}+"
        f"{cost.prefill_per_token_ms:.3f}/токен ms, "
        f"decode={cost.decode_base_ms:.2f}+"
        f"{cost.decode_per_slot_ms:.3f}/слот ms"
    )
    if not real:
        note("Это синтетическая модель. Для калибровки: LAB_REAL=1.")

    workload = make_workload()
    show_workload(workload)
    compare_static_continuous(workload, cost)
    compare_chunks(workload, cost)
    compare_policies(workload, cost)

    section("Вывод")
    note("Continuous batching = слоты + очередь + prefill/decode interleaving.")
    note("Chunked prefill — компромисс: latency очереди vs dispatch overhead.")
    note("Политика очереди меняет TTFT, среднюю латентность и fairness.")
    note("Дальше: заменить CostModel настоящим batched forward и кэшами.")


if __name__ == "__main__":
    main()
