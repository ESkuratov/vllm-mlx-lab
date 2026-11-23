# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 18: планировщик — политики, справедливость, starvation.

Идея
----
Continuous batching (эксп. 3) решает «как считать», а планировщик решает
«кого считать следующим». Когда запросов больше, чем слотов, политика выбора
определяет латентность, справедливость и голодание (starvation).

Здесь строим событийный симулятор сервера и сравниваем политики:
  * FCFS — кто пришёл раньше;
  * SJF — сначала короткие (низкая средняя латентность, но длинные голодают);
  * Priority — по приоритету (интерактив важнее батча);
  * Fair — round-robin по классам (честно, но не оптимально).

В конце — реальный `BatchGenerator` из mlx-lm, чтобы свериться с симуляцией.
Во vllm-mlx политиками заведует `scheduler.py`.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/18_scheduler.py"
"""
from __future__ import annotations

import random
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import MODEL_ID, base_token_ids, load_model, note, section, subsection


@dataclass
class Req:
    rid: int
    arrival: int
    prompt: int
    output: int
    priority: int
    cls: str


def pick(queue: list[Req], policy: str, t: int, rr_state: dict) -> Req:
    if policy == "fcfs":
        key = lambda r: r.arrival
    elif policy == "sjf":
        key = lambda r: r.prompt + r.output
    elif policy == "priority":
        key = lambda r: (-r.priority, r.arrival)
    elif policy == "fair":
        classes = sorted({r.cls for r in queue})
        n = len(classes)
        start = rr_state.get("idx", 0)
        for off in range(n):
            cls = classes[(start + off) % n]
            cands = [r for r in queue if r.cls == cls]
            if cands:
                rr_state["idx"] = (start + off + 1) % n
                chosen = min(cands, key=lambda r: r.arrival)
                queue.remove(chosen)
                return chosen
    chosen = min(queue, key=key)
    queue.remove(chosen)
    return chosen


def simulate(requests: list[Req], slots: int, policy: str, prefill_speed: int = 128):
    """Событийный симулятор: слот занят prefill, затем decode по 1 токену/тик."""
    requests = sorted(requests, key=lambda r: r.arrival)
    t = 0
    idx = 0
    queue: list[Req] = []
    active = []  # dict: r, rem_prefill, rem_out, ttft
    done = []
    rr_state: dict = {}

    while (idx < len(requests) or queue or active) and t < 100000:
        while idx < len(requests) and requests[idx].arrival <= t:
            queue.append(requests[idx])
            idx += 1

        while len(active) < slots and queue:
            r = pick(queue, policy, t, rr_state)
            active.append({"r": r, "prefill": r.prompt, "out": r.output, "ttft": None})

        for a in active:
            if a["prefill"] > 0:
                a["prefill"] -= prefill_speed
                if a["prefill"] <= 0:
                    a["prefill"] = 0
                    a["ttft"] = t + 1
            else:
                a["out"] -= 1
        t += 1

        still = []
        for a in active:
            if a["prefill"] == 0:
                if a["out"] <= 0:
                    done.append(a)
                else:
                    still.append(a)
            else:
                still.append(a)
        active = still

    return done, t


def jain(counts: list[int]) -> float:
    s = sum(counts)
    if s == 0:
        return 0.0
    return (s * s) / (len(counts) * sum(c * c for c in counts))


def report(name: str, done, horizon: int) -> None:
    ttfts = [a["ttft"] for a in done]
    lat = [a["ttft"] + a["r"].output - a["r"].arrival for a in done]
    lat_sorted = sorted(lat)
    p95 = lat_sorted[int(0.95 * (len(lat_sorted) - 1))]
    by_cls: dict[str, list[int]] = {}
    for a in done:
        by_cls.setdefault(a["r"].cls, []).append(a["ttft"] + a["r"].output - a["r"].arrival)
    cls_lat = {k: statistics.mean(v) for k, v in by_cls.items()}
    # честность по КАЧЕСТВУ: ровная латентность между классами = высокая
    inv = [1.0 / max(1e-6, cls_lat[k]) for k in by_cls]
    scale = min(inv)
    fair_idx = jain([int(round(x / scale * 100)) for x in inv])
    print(
        f"  {name:<9}{statistics.mean(ttfts):>6.1f}{statistics.mean(lat):>7.1f}"
        f"{p95:>7.1f}{cls_lat.get('interactive', float('nan')):>12.1f}"
        f"{cls_lat.get('batch', float('nan')):>8.1f}{fair_idx:>7.2f}"
    )


def make_workload() -> list[Req]:
    rng = random.Random(0)
    reqs = []
    rid = 0
    for i in range(24):
        cls = "interactive" if i % 2 == 0 else "batch"
        reqs.append(
            Req(
                rid=rid,
                arrival=i * 3 + rng.randint(0, 2),
                prompt=rng.randint(64, 256),
                output=rng.randint(20, 60) if cls == "interactive" else rng.randint(120, 240),
                priority=2 if cls == "interactive" else 0,
                cls=cls,
            )
        )
        rid += 1
    return reqs


def main() -> None:
    section("18. Планировщик: политики и справедливость")
    workload = make_workload()
    print(f"  запросов: {len(workload)} (interactive: короткие, приоритет 2; batch: длинные)")
    print(f"  слотов: 4, prefill_speed=128 токенов/тик")

    subsection("A. Сравнение политик (те же запросы, разный порядок)")
    print(
        f"  {'политика':<9}{'TTFT':>6}{'лат.':>7}{'p95':>7}"
        f"{'интерактив':>12}{'батч':>8}{'Jain':>7}"
    )
    for policy in ("fcfs", "sjf", "priority", "fair"):
        done, horizon = simulate(workload, slots=4, policy=policy)
        report(policy, done, horizon)
    note(
        "FCFS: длинные batch-задачи блокируют интерактив → он ждёт.\n"
        "  SJF: минимальная средняя латентность, но batch может голодать.\n"
        "  Priority: интерактив быстрый, batch — платит.\n"
        "  Fair: классы обслуживаются по кругу — компромисс."
    )

    subsection("B. Наглядно: сколько ждёт интерактив")
    for policy in ("fcfs", "priority", "sjf"):
        done, _ = simulate(workload, slots=4, policy=policy)
        inter = [a["ttft"] for a in done if a["r"].cls == "interactive"]
        batch = [a["ttft"] for a in done if a["r"].cls == "batch"]
        print(
            f"  {policy:<9} TTFT интерактив {statistics.mean(inter):>6.1f}  |  "
            f"TTFT batch {statistics.mean(batch):>6.1f}"
        )
    note("Приоритеты меняют не пропускную способность, а распределение задержек.")

    subsection("C. Сверка с реальным BatchGenerator (mlx-lm)")
    model, tok = load_model()
    from mlx_lm.generate import batch_generate
    from mlx_lm.sample_utils import make_sampler

    sampler = make_sampler(temp=0.0)
    prompts = [base_token_ids(tok, 64).tolist() for _ in range(16)]
    print(f"  {'conc':>5}{'шагов':>8}{'время':>9}{'tokens/s':>11}{'prompt_time':>13}")
    for conc in (2, 4, 8, 16):
        r = batch_generate(
            model, tok, prompts, max_tokens=24,
            completion_batch_size=conc, prefill_batch_size=conc, sampler=sampler,
        )
        s = r.stats
        print(
            f"  {conc:>5}{s.generation_steps:>8}{s.wall_time:>8.2f}s"
            f"{s.generation_tps:>11.0f}{s.prompt_time:>12.2f}s"
        )
    note(
        "BatchGenerator сам подхватывает работу по мере освобождения слотов — это "
        "и есть continuous batching. Порядок в нём близок к FCFS; во vllm-mlx "
        "scheduler.py добавляет приоритеты и ограничения по памяти."
    )

    section("Вывод")
    note("Пропускную способность задаёт батчинг; политика задаёт КОМУ какие задержки.")
    note("SJF минимизирует среднее время, priority — latency важных, fair — перекос.")
    note("Практика: приоритеты + aging против starvation + бюджет памяти на KV.")


if __name__ == "__main__":
    main()
