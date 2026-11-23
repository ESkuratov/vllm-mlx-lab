# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 17: MoE — роутинг и grouped expert matmul.

Идея
----
MoE заменяет один dense FFN на E экспертов и top-k роутер:

* router выдаёт логиты по экспертам;
* выбираются top-k, их softmax-веса нормируются;
* для каждого токена считаются только выбранные эксперты;
* grouped expert matmul собирает токены по экспертам и считает их одним
  `mx.gather_qmm`, без цикла «токен × эксперт» по Python.

Это архитектура Qwen3-MoE (30B-A3B): много параметров, но мало активных
вычислений на токен.

Здесь веса синтетические: 30B-чекпоинт не нужен, чтобы увидеть механику.
Сравниваем с медленным эталоном, который деquantize-ит каждого эксперта.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/17_qwen3_moe.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn

from common import note, section, subsection

GROUP = 64
BITS = 4


# --------------------------------------------------------------------------- #
# Quantized expert weights
# --------------------------------------------------------------------------- #


@dataclass
class QuantizedExperts:
    weight: mx.array
    scales: mx.array
    biases: mx.array
    group_size: int
    bits: int


def quantize_experts(w: mx.array, group_size=GROUP, bits=BITS) -> QuantizedExperts:
    packed, scales, biases = mx.quantize(w, group_size=group_size, bits=bits)
    return QuantizedExperts(packed, scales, biases, group_size, bits)


def grouped_expert_linear(
    x: mx.array, w: QuantizedExperts, expert_ids: mx.array
) -> mx.array:
    """x [N, D], expert_ids [N] -> [N, out]."""
    *leading, d = x.shape
    flat_x = x.reshape(-1, d)
    flat_ids = expert_ids.reshape(-1)
    sort_idx = mx.argsort(flat_ids)
    inv_sort_idx = mx.argsort(sort_idx)
    grouped_x = flat_x[sort_idx]
    grouped_ids = flat_ids[sort_idx]
    out = mx.gather_qmm(
        mx.expand_dims(grouped_x, -2),
        w.weight,
        w.scales,
        w.biases,
        lhs_indices=mx.arange(grouped_x.shape[0]),
        rhs_indices=grouped_ids,
        transpose=True,
        group_size=w.group_size,
        bits=w.bits,
        sorted_indices=True,
    ).squeeze(-2)
    out_dim = w.weight.shape[-2]
    return out[inv_sort_idx].reshape(*leading, out_dim)


def route_topk(
    x: mx.array, router: mx.array, top_k: int, norm_topk_prob: bool = False
):
    logits = x @ router.T
    probs = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True)
    ids = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(probs, ids, axis=-1)
    if norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)
    return probs, ids, scores.astype(x.dtype)


# --------------------------------------------------------------------------- #
# MoE block
# --------------------------------------------------------------------------- #


class Moe:
    def __init__(self, router, gate, up, down, top_k, norm_topk_prob=False):
        self.router = router
        self.gate = gate
        self.up = up
        self.down = down
        self.top_k = top_k
        self.norm_topk_prob = norm_topk_prob

    def __call__(self, x: mx.array) -> mx.array:
        b, length, d = x.shape
        _, ids, scores = route_topk(
            x, self.router, self.top_k, self.norm_topk_prob
        )
        expanded = mx.broadcast_to(
            mx.expand_dims(x, -2), (b, length, self.top_k, d)
        ).reshape(-1, d)
        flat_ids = ids.reshape(-1)
        gate = grouped_expert_linear(expanded, self.gate, flat_ids)
        up = grouped_expert_linear(expanded, self.up, flat_ids)
        hidden = nn.silu(gate) * up
        out = grouped_expert_linear(hidden, self.down, flat_ids)
        out = out.reshape(b, length, self.top_k, d)
        return mx.sum(out * mx.expand_dims(scores, -1), axis=-2)


def dequant_expert(w: QuantizedExperts, e: int) -> mx.array:
    return mx.dequantize(
        w.weight[e], w.scales[e], w.biases[e],
        group_size=w.group_size, bits=w.bits,
    )


def reference_moe(x, router, gate, up, down, top_k, norm_topk_prob=False):
    """Медленный эталон: цикл по токенам и выбранным экспертам."""
    b, length, d = x.shape
    flat = x.reshape(-1, d)
    logits = flat @ router.T
    probs = mx.softmax(logits.astype(mx.float32), axis=-1, precise=True)
    ids = mx.argpartition(-probs, kth=top_k - 1, axis=-1)[..., :top_k]
    scores = mx.take_along_axis(probs, ids, axis=-1)
    if norm_topk_prob:
        scores = scores / mx.sum(scores, axis=-1, keepdims=True)

    out = mx.zeros_like(flat)
    for t in range(flat.shape[0]):
        for j in range(top_k):
            e = int(ids[t, j])
            wg = dequant_expert(gate, e)
            wu = dequant_expert(up, e)
            wd = dequant_expert(down, e)
            h = nn.silu(flat[t : t + 1] @ wg.T) * (flat[t : t + 1] @ wu.T)
            out[t : t + 1] += (h @ wd.T) * scores[t, j]
    return out.reshape(b, length, d)


# --------------------------------------------------------------------------- #
# Проверки и замеры
# --------------------------------------------------------------------------- #


def _assert_close(name, actual, expected):
    ok = mx.allclose(
        actual.astype(mx.float32),
        expected.astype(mx.float32),
        rtol=5e-2,
        atol=5e-2,
    )
    if not bool(ok):
        diff = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
        raise AssertionError(f"{name}: max diff {float(diff):.4f}")
    print(f"  ✓ {name}")


def _bench(fn, repeats=3):
    def run():
        mx.eval(fn())

    run()
    mx.synchronize()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        run()
        mx.synchronize()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def build_case(seed=1, experts=8, d=256, hidden=512, tokens=32, top_k=2):
    mx.random.seed(seed)
    router = (mx.random.normal((experts, d)) * 0.02).astype(mx.bfloat16)
    wg = (mx.random.normal((experts, hidden, d)) * 0.02).astype(mx.bfloat16)
    wu = (mx.random.normal((experts, hidden, d)) * 0.02).astype(mx.bfloat16)
    wd = (mx.random.normal((experts, d, hidden)) * 0.02).astype(mx.bfloat16)
    gate = quantize_experts(wg)
    up = quantize_experts(wu)
    down = quantize_experts(wd)
    x = (mx.random.normal((1, tokens, d)) * 0.5).astype(mx.bfloat16)
    return x, router, gate, up, down, top_k


def correctness():
    subsection("A. Групповой matmul vs цикл по экспертам")
    x, router, gate, up, down, top_k = build_case()
    moe = Moe(router, gate, up, down, top_k, norm_topk_prob=True)
    actual = moe(x)
    expected = reference_moe(
        x, router, gate, up, down, top_k, norm_topk_prob=True
    )
    _assert_close("MoE grouped vs reference", actual, expected)


def economics():
    subsection("B. Экономика параметров и баланс роутера")
    x, router, gate, up, down, top_k = build_case()
    experts = gate.weight.shape[0]
    d = x.shape[-1]
    hidden = gate.weight.shape[1]
    total = experts * 3 * hidden * d
    active = top_k * 3 * hidden * d
    print(f"  экспертов: {experts}, top_k={top_k}")
    print(f"  параметров всего: {total / 1e6:.1f} M")
    print(f"  активных на токен: {active / 1e6:.1f} M ({active / total:.1%})")

    _, ids, _ = route_topk(x, router, top_k, True)
    counts = [0] * experts
    for e in ids.reshape(-1).tolist():
        counts[e] += 1
    print(f"  загрузка экспертов: {counts}")
    note(
        "MoE даёт ёмкость большой модели при вычислениях маленькой. "
        "Без балансировки роутер может свалить всё на несколько экспертов."
    )


def benchmark():
    subsection("C. Grouped gather_qmm vs Python-цикл")
    x, router, gate, up, down, top_k = build_case(tokens=32)
    moe = Moe(router, gate, up, down, top_k, True)
    t_fast = _bench(lambda: moe(x), repeats=3)
    t_ref = _bench(
        lambda: reference_moe(x, router, gate, up, down, top_k, True),
        repeats=2,
    )
    print(f"  grouped: {t_fast * 1e3:7.2f} ms")
    print(f"  reference: {t_ref * 1e3:7.2f} ms")
    print(f"  ускорение: {t_ref / t_fast:.1f}x")
    note(
        "Grouped-путь сортирует токены по экспертам и делает один "
        "quantized matmul на эксперта, а не Python-цикл по (token, expert)."
    )


def main():
    section("17. MoE: роутинг и grouped expert matmul")
    correctness()
    economics()
    benchmark()

    note(
        "В Qwen3-MoE используется тот же принцип: router + top-k + "
        "SwitchGLU/grouped matmul. Веса здесь синтетические, чтобы "
        "лаборатория работала без 30B-чекпоинта."
    )

    section("Вывод")
    note("Router выбирает top-k экспертов и нормирует их веса.")
    note("Grouped matmul собирает токены по экспертам и переиспользует веса.")
    note("Активных параметров в top_k/E раз меньше, чем всего.")
    note("Баланс экспертов — отдельная проблема, без него MoE теряет смысл.")


if __name__ == "__main__":
    main()
