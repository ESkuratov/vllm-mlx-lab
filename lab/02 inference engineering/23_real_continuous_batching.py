# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 23: настоящий continuous batching на реальной модели.

Идея
----
В inference/20 мы строили политику планировщика на синтетической CostModel.
Здесь те же слоты и та же логика, но с настоящими KV-кэшами и настоящим
batched forward модели Qwen2.5.

Механика:

* у каждого запроса свой prefill-кэш (``make_prompt_cache``);
* prefill идёт чанками и может чередоваться с decode-шагами;
* готовый запрос «переезжает» в общий batch через ``_merge_caches``;
* при завершении строка удаляется через ``BatchKVCache.filter``;
* освободившийся слот сразу занимает следующий запрос.

Это те же операции, что делает ``mlx_lm.generate.BatchGenerator``, но
собранные руками и с явными метриками. Используются приватные
``_merge_caches`` / ``_extend_cache`` из mlx-lm — они версионно-зависимы.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/23_real_continuous_batching.py"

Модель задаётся через LAB_MODEL. На 3B greedy-результат совпадает с
последовательным; на 0.5B bf16-батчинг может разойтись на поздних токенах.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache

from common import MODEL_ID, note, section, subsection

MAX_SLOTS = 2
PREFILL_CHUNK = 4
MAX_TOKENS = 12

PROMPTS = [
    "Explain what a KV cache is.",
    "Write a haiku about memory.",
    "What is paged attention?",
    "Why is decode memory bound?",
    "Define continuous batching.",
    "What does a scheduler do?",
]


# --------------------------------------------------------------------------- #
# Запросы
# --------------------------------------------------------------------------- #


@dataclass
class Request:
    rid: int
    ids: list[int]
    cache: list | None = None
    out: list[int] = field(default_factory=list)
    first: float | None = None
    finish: float | None = None
    processed: int = 0
    last: int | None = None
    done: bool = False

    @property
    def prefill_total(self) -> int:
        return max(0, len(self.ids) - 1)


# --------------------------------------------------------------------------- #
# Последовательный baseline
# --------------------------------------------------------------------------- #


def _prefill(model, cache, ids: list[int], chunk: int = PREFILL_CHUNK) -> None:
    """Скормить ids чанками в переданный кэш."""
    for start in range(0, len(ids), chunk):
        part = mx.array(ids[start : start + chunk])[None]
        model(part, cache=cache)
        mx.eval([c.state for c in cache])


def _step(model, cache, token: int) -> int:
    logits = model(mx.array([[token]]), cache=cache)
    mx.eval(logits)
    return int(mx.argmax(logits[0, -1]))


def sequential_generate(model, requests: list[Request], max_tokens: int, eos: int | None):
    """Каждый запрос отдельно: prefill по чанкам + greedy decode."""
    t0 = time.perf_counter()
    for req in requests:
        req.cache = make_prompt_cache(model)
        _prefill(model, req.cache, req.ids[:-1])
        token = req.ids[-1]
        while len(req.out) < max_tokens:
            token = _step(model, req.cache, token)
            req.out.append(token)
            if token == eos:
                break
    mx.synchronize()
    return time.perf_counter() - t0


# --------------------------------------------------------------------------- #
# Continuous batching
# --------------------------------------------------------------------------- #


def continuous_generate(
    model,
    requests: list[Request],
    max_tokens: int,
    eos: int | None,
    max_slots: int = MAX_SLOTS,
    prefill_chunk: int = PREFILL_CHUNK,
):
    """Один prefill-слот + N decode-слотов на реальных BatchKVCache."""
    from mlx_lm.generate import _extend_cache, _merge_caches

    waiting = list(requests)
    active: list[Request] = []
    batch = []
    inputs: list[int] = []
    prefill: Request | None = None
    decode_slot_time = 0.0
    prefill_time = 0.0
    events: list[str] = []
    step = 0
    t0 = time.perf_counter()

    while waiting or active or prefill is not None:
        # 1. Начать prefill следующего запроса, если есть свободный слот.
        if prefill is None and waiting and len(active) < max_slots:
            req = waiting.pop(0)
            req.cache = make_prompt_cache(model)
            req.processed = 0
            prefill = req

        # 2. Один чанк prefill.
        if prefill is not None:
            take = min(prefill_chunk, prefill.prefill_total - prefill.processed)
            if take > 0:
                part = mx.array(
                    prefill.ids[prefill.processed : prefill.processed + take]
                )[None]
                t_pref = time.perf_counter()
                model(part, cache=prefill.cache)
                mx.eval([c.state for c in prefill.cache])
                prefill_time += time.perf_counter() - t_pref
                prefill.processed += take

            if prefill.processed >= prefill.prefill_total:
                req = prefill
                req.last = req.ids[-1]
                new_batch = _merge_caches([req.cache])
                if active:
                    _extend_cache(batch, new_batch)
                else:
                    batch = new_batch
                active.append(req)
                inputs.append(req.last)
                events.append(f"add #{req.rid}")
                prefill = None

        # 3. Один общий decode-шаг.
        if active:
            t_step = time.perf_counter()
            logits = model(mx.array(inputs)[:, None], cache=batch)
            mx.eval(logits)
            step_dt = time.perf_counter() - t_step
            decode_slot_time += step_dt * len(active)

            tokens = [int(mx.argmax(logits[i, -1])) for i in range(len(active))]
            for req, token in zip(active, tokens):
                req.out.append(token)
                if req.first is None:
                    req.first = time.perf_counter() - t0
                if len(req.out) >= max_tokens or token == eos:
                    req.done = True
                    req.finish = time.perf_counter() - t0

            keep = [i for i, req in enumerate(active) if not req.done]
            if len(keep) < len(active):
                if keep:
                    keep_idx = mx.array(keep, dtype=mx.int32)
                    for cache in batch:
                        cache.filter(keep_idx)
                    for i, req in enumerate(active):
                        if req.done:
                            events.append(f"done #{req.rid}")
                    active = [active[i] for i in keep]
                    inputs = [tokens[i] for i in keep]
                else:
                    for req in active:
                        events.append(f"done #{req.rid}")
                    active = []
                    inputs = []
                    batch = []
            else:
                inputs = tokens
            step += 1

    mx.synchronize()
    elapsed = time.perf_counter() - t0
    stats = {
        "elapsed": elapsed,
        "decode_slot_time": decode_slot_time,
        "prefill_time": prefill_time,
        "events": events,
        "steps": step,
    }
    return stats


# --------------------------------------------------------------------------- #
# Метрики
# --------------------------------------------------------------------------- #


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]


def report(requests: list[Request], stats: dict, max_slots: int) -> None:
    elapsed = stats["elapsed"]
    ttfts = [r.first for r in requests if r.first is not None]
    tpots = []
    latencies = []
    for req in requests:
        if req.first is None or req.finish is None:
            continue
        steps = max(1, len(req.out) - 1)
        tpots.append((req.finish - req.first) / steps)
        latencies.append(req.finish)
    tokens = sum(len(r.out) for r in requests)
    decode_util = stats["decode_slot_time"] / (max_slots * elapsed) if elapsed else 0
    print(f"  времени: {elapsed:.3f} с, шагов decode: {stats['steps']}")
    print(f"  токенов: {tokens}, throughput: {tokens / elapsed:.1f} tok/s")
    print(
        f"  TTFT mean/p95: {statistics.mean(ttfts) * 1e3:.1f} / "
        f"{_p95(ttfts) * 1e3:.1f} ms"
    )
    print(
        f"  TPOT mean: {statistics.mean(tpots) * 1e3:.1f} ms, "
        f"latency p95: {_p95(latencies) * 1e3:.1f} ms"
    )
    print(
        f"  decode slot util: {decode_util:.1%}, "
        f"prefill busy: {stats['prefill_time'] / elapsed:.1%}"
    )
    print("  события: " + ", ".join(stats["events"][:14]))


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    section("23. Настоящий continuous batching")
    model_id = os.environ.get("LAB_MODEL", MODEL_ID)
    print(f"  модель: {model_id}")
    model, tok = load(model_id)
    eos = tok.eos_token_id

    requests = []
    for i, prompt in enumerate(PROMPTS):
        ids = tok.encode(prompt, add_special_tokens=False)
        requests.append(Request(i, ids))
    print(
        "  запросы: "
        + ", ".join(f"#{r.rid}:{len(r.ids)}" for r in requests)
        + f"; max_slots={MAX_SLOTS}, prefill_chunk={PREFILL_CHUNK}"
    )

    # Прогрев.
    warm = Request(-1, tok.encode("Hello", add_special_tokens=False))
    sequential_generate(model, [warm], 2, eos)

    subsection("A. Последовательно, без батчинга")
    seq_requests = [Request(r.rid, list(r.ids)) for r in requests]
    t_seq = sequential_generate(model, seq_requests, MAX_TOKENS, eos)
    seq_tokens = sum(len(r.out) for r in seq_requests)
    print(
        f"  времени: {t_seq:.3f} с, токенов: {seq_tokens}, "
        f"throughput: {seq_tokens / t_seq:.1f} tok/s"
    )

    subsection("B. Continuous batching на BatchKVCache")
    cont_requests = [Request(r.rid, list(r.ids)) for r in requests]
    stats = continuous_generate(model, cont_requests, MAX_TOKENS, eos)
    report(cont_requests, stats, MAX_SLOTS)
    print(f"  ускорение vs sequential: {t_seq / stats['elapsed']:.2f}x")

    subsection("C. Корректность greedy")
    mismatches = []
    for seq, cont in zip(seq_requests, cont_requests):
        if seq.out != cont.out:
            mismatches.append(cont.rid)
    if not mismatches:
        print("  ✓ все запросы выдали те же токены, что и последовательный greedy")
    else:
        print(
            f"  ⚠ расхождения у запросов {mismatches}: bf16-батчинг и "
            "left padding могут менять argmax на поздних токенах."
        )
        note(
            "На 3B в этой конфигурации расхождений нет. На 0.5B они возможны — "
            "это численная особенность, а не ошибка планировщика."
        )

    note(
        "Ключевые операции: make_prompt_cache → prefill чанками → "
        "_merge_caches/_extend_cache → model(inputs[:, None]) → "
        "BatchKVCache.filter при завершении."
    )
    note(
        "Это настоящий batched forward, а не CostModel: TTFT/TPOT и "
        "утилизация слотов измерены на реальной модели."
    )
    note(
        "_merge_caches/_extend_cache — приватные API mlx-lm. При обновлении "
        "библиотеки проверьте их сигнатуры или замените на BatchKVCache.prepare/extend."
    )

    section("Вывод")
    note("Continuous batching = общий batch-кэш + слоты + prefill/decode.")
    note("Новый запрос добавляется в batch через extend, старый — через filter.")
    note("Chunked prefill позволяет decode-шагам идти между кусками промпта.")
    note("На реальной модели видно цену left padding и bf16-нестабильности.")


if __name__ == "__main__":
    main()
