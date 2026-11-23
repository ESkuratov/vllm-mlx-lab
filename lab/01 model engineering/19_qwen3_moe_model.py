# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 19: уменьшенная Qwen3-MoE модель.

Идея
----
Lab/45 показал grouped expert matmul отдельно. Здесь мы собираем
**полный decoder-only MoE-блок**:

* GQA attention с RoPE и QK-RMSNorm;
* sparse MoE FFN: router + top-k + grouped experts;
* residual stream, pre-norm, lm_head;
* сравнение grouped-модели с эталоном, который считает экспертов циклом.

Веса синтетические: это уменьшенная копия архитектуры Qwen3-MoE, а не
30B-чекпоинт. Так видна вся проводка без загрузки 30B.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/19_qwen3_moe_model.py"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn

from common import note, section, subsection
from metal_ops import rms_norm_ref, rope_ref
from _loader import load_lab

lab17 = load_lab("17_qwen3_moe", "lab17_moe")
Moe = lab17.Moe
quantize_experts = lab17.quantize_experts
reference_moe = lab17.reference_moe
route_topk = lab17.route_topk


@dataclass
class Config:
    vocab: int = 512
    hidden: int = 256
    layers: int = 2
    heads: int = 4
    kv_heads: int = 2
    head_dim: int = 64
    experts: int = 8
    top_k: int = 2
    moe_hidden: int = 512
    rope_theta: float = 10000.0
    eps: float = 1e-6


# --------------------------------------------------------------------------- #
# Слои
# --------------------------------------------------------------------------- #


class Attention:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        d = cfg.hidden
        q_dim = cfg.heads * cfg.head_dim
        kv_dim = cfg.kv_heads * cfg.head_dim
        self.wq = mx.random.normal((q_dim, d)) * 0.02
        self.wk = mx.random.normal((kv_dim, d)) * 0.02
        self.wv = mx.random.normal((kv_dim, d)) * 0.02
        self.wo = mx.random.normal((d, q_dim)) * 0.02
        self.q_norm = mx.ones((cfg.head_dim,))
        self.k_norm = mx.ones((cfg.head_dim,))
        for name in ("wq", "wk", "wv", "wo", "q_norm", "k_norm"):
            setattr(self, name, getattr(self, name).astype(mx.bfloat16))

    def __call__(self, x: mx.array) -> mx.array:
        b, length, _ = x.shape
        cfg = self.cfg
        q = (x @ self.wq.T).reshape(b, length, cfg.heads, cfg.head_dim)
        k = (x @ self.wk.T).reshape(b, length, cfg.kv_heads, cfg.head_dim)
        v = (x @ self.wv.T).reshape(b, length, cfg.kv_heads, cfg.head_dim)
        q = rms_norm_ref(q, self.q_norm, cfg.eps)
        k = rms_norm_ref(k, self.k_norm, cfg.eps)
        offsets = mx.zeros((b,), dtype=mx.int32)
        q = rope_ref(q, offsets, cfg.rope_theta).transpose(0, 2, 1, 3)
        k = rope_ref(k, offsets, cfg.rope_theta).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        repeats = cfg.heads // cfg.kv_heads
        if repeats > 1:
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)
        scores = mx.matmul(q, k.swapaxes(-2, -1)) * (cfg.head_dim**-0.5)
        mask = mx.tril(mx.ones((length, length)), 0)
        scores = scores + mx.where(mask, 0.0, -mx.inf)[None, None]
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = mx.matmul(probs, v).transpose(0, 2, 1, 3).reshape(b, length, -1)
        return out @ self.wo.T


class Block:
    def __init__(self, cfg: Config, grouped: bool = True):
        self.cfg = cfg
        self.input_norm = mx.ones((cfg.hidden,)).astype(mx.bfloat16)
        self.post_norm = mx.ones((cfg.hidden,)).astype(mx.bfloat16)
        self.attn = Attention(cfg)
        self.router = (mx.random.normal((cfg.experts, cfg.hidden)) * 0.02).astype(
            mx.bfloat16
        )
        wg = (mx.random.normal(
            (cfg.experts, cfg.moe_hidden, cfg.hidden)) * 0.02
        ).astype(mx.bfloat16)
        wu = (mx.random.normal(
            (cfg.experts, cfg.moe_hidden, cfg.hidden)) * 0.02
        ).astype(mx.bfloat16)
        wd = (mx.random.normal(
            (cfg.experts, cfg.hidden, cfg.moe_hidden)) * 0.02
        ).astype(mx.bfloat16)
        self.gate = quantize_experts(wg)
        self.up = quantize_experts(wu)
        self.down = quantize_experts(wd)
        self.grouped = grouped
        self.moe = Moe(
            self.router, self.gate, self.up, self.down,
            cfg.top_k, norm_topk_prob=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        cfg = self.cfg
        x = x + self.attn(rms_norm_ref(x, self.input_norm, cfg.eps))
        h = rms_norm_ref(x, self.post_norm, cfg.eps)
        if self.grouped:
            y = self.moe(h)
        else:
            y = reference_moe(
                h, self.router, self.gate, self.up, self.down,
                cfg.top_k, norm_topk_prob=True,
            )
        return x + y


class TinyMoeModel:
    def __init__(self, cfg: Config, grouped: bool = True):
        self.cfg = cfg
        self.embed = (mx.random.normal((cfg.vocab, cfg.hidden)) * 0.02).astype(
            mx.bfloat16
        )
        self.layers = [Block(cfg, grouped) for _ in range(cfg.layers)]
        self.norm = mx.ones((cfg.hidden,)).astype(mx.bfloat16)
        self.lm_head = (mx.random.normal((cfg.vocab, cfg.hidden)) * 0.02).astype(
            mx.bfloat16
        )

    def __call__(self, ids: mx.array) -> mx.array:
        h = self.embed[ids]
        for layer in self.layers:
            h = layer(h)
        h = rms_norm_ref(h, self.norm, self.cfg.eps)
        return h @ self.lm_head.T


def _assert_close(name, actual, expected):
    ok = mx.allclose(
        actual.astype(mx.float32), expected.astype(mx.float32),
        rtol=5e-2, atol=5e-2,
    )
    if not bool(ok):
        diff = mx.max(mx.abs(actual.astype(mx.float32) - expected.astype(mx.float32)))
        raise AssertionError(f"{name}: max diff {float(diff):.4f}")
    print(f"  ✓ {name}")


def main():
    section("19. Уменьшенная Qwen3-MoE модель")
    cfg = Config()
    mx.random.seed(11)
    grouped = TinyMoeModel(cfg, grouped=True)
    reference = TinyMoeModel(cfg, grouped=False)
    # Одинаковые веса: копируем параметры grouped -> reference.
    reference.embed = grouped.embed
    reference.norm = grouped.norm
    reference.lm_head = grouped.lm_head
    for g, r in zip(grouped.layers, reference.layers):
        r.input_norm = g.input_norm
        r.post_norm = g.post_norm
        r.attn = g.attn
        r.router = g.router
        r.gate = g.gate
        r.up = g.up
        r.down = g.down

    ids = mx.array([[1, 5, 9, 13, 17, 21, 25, 29]])
    print(
        f"  config: hidden={cfg.hidden}, layers={cfg.layers}, "
        f"H={cfg.heads}/{cfg.kv_heads}, experts={cfg.experts}, "
        f"top_k={cfg.top_k}"
    )

    subsection("A. Grouped MoE vs reference")
    out = grouped(ids)
    ref = reference(ids)
    mx.eval(out, ref)
    _assert_close("full MoE model logits", out, ref)

    subsection("B. Параметры и активность")
    expert_params = cfg.experts * 3 * cfg.moe_hidden * cfg.hidden
    active = cfg.top_k * 3 * cfg.moe_hidden * cfg.hidden
    attn_params = 4 * cfg.hidden * cfg.hidden
    print(f"  experts: {expert_params / 1e6:.1f} M")
    print(f"  active experts/token: {active / 1e6:.1f} M")
    print(f"  attention/block: {attn_params / 1e6:.3f} M")
    print(
        f"  active fraction (MoE): {active / expert_params:.1%} "
        f"при top_k={cfg.top_k}/{cfg.experts}"
    )

    # Загрузка роутера на последнем слое.
    h = grouped.embed[ids]
    for layer in grouped.layers[:-1]:
        h = layer(h)
    last = grouped.layers[-1]
    h = rms_norm_ref(h, last.input_norm, cfg.eps)
    _, ids_top, _ = route_topk(h, last.router, cfg.top_k, True)
    counts = [0] * cfg.experts
    for e in ids_top.reshape(-1).tolist():
        counts[e] += 1
    print(f"  router load (last layer): {counts}")

    note(
        "Grouped-путь и reference используют одни веса; отличие только "
        "в способе вычисления экспертов."
    )
    note(
        "Это уменьшенная копия Qwen3-MoE: та же схема router + top-k + "
        "grouped matmul, но без 30B-чекпоинта."
    )

    section("Вывод")
    note("MoE встраивается в обычный decoder block вместо dense FFN.")
    note("Grouped experts дают тот же результат, что цикл, но быстрее.")
    note("Активных параметров в top_k/E раз меньше, чем всего.")
    note("Для реального Qwen3-MoE остаётся загрузить чекпоинт и его конфиг.")


if __name__ == "__main__":
    main()
