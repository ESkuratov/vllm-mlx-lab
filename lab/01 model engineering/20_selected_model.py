# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 20: единая Week 2 `selected`-модель с checkpoint-флагами.

Идея
----
В tiny-llm Week 2 механизмы включаются постепенно:

``kv-cache → capacity-cache → quantized-matvec → simd-matmul →
rmsnorm → rope → swiglu → tiled-prefill → selected``.

Здесь один класс ``SelectedModel`` собирает реальный Qwen-чекпоинт и
переключает признаки по имени checkpoint. ``selected`` включает всё:
fused RMSNorm/RoPE/SwiGLU и tiled dense prefill attention из model/16.

``kv-cache``/``capacity-cache`` — про decode; они покрыты inference/24 и inference/23.
В этой лаборатории они отмечены в таблице, но prefill-модель их не
использует.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/20_selected_model.py"
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import load

from common import MODEL_ID, note, section, subsection
from metal_ops import (
    HAS_METAL,
    metal_rms_norm,
    metal_rope,
    metal_swiglu,
    rms_norm_ref,
    rope_ref,
    swiglu_ref,
)
from _loader import load_lab

lab13 = load_lab("13_load_qwen3", "lab13_qwen")
lab16 = load_lab("16_tiled_dense_prefill_attention", "lab16_attn")
QLinear = lab13.QLinear
tiled_attention = lab16.tiled_attention

CHECKPOINTS = (
    "kv-cache",
    "capacity-cache",
    "quantized-matvec",
    "simd-matmul",
    "rmsnorm",
    "rope",
    "swiglu",
    "tiled-prefill",
    "selected",
)


def features(checkpoint: str) -> dict[str, bool]:
    index = CHECKPOINTS.index(checkpoint)
    return {
        "capacity_cache": index >= 1,
        "quantized_weights": index >= 2,
        "simd_matmul": index >= 3,
        "fast_rmsnorm": index >= 4,
        "fast_rope": index >= 5,
        "fast_swiglu": index >= 6,
        "tiled_prefill": index >= 7,
        "selected": index >= 8,
    }


class Norm:
    def __init__(self, weight, eps, fast):
        self.weight = weight
        self.eps = eps
        self.fast = fast and HAS_METAL

    def __call__(self, x):
        if self.fast:
            return metal_rms_norm(x, self.weight, self.eps)
        return rms_norm_ref(x, self.weight, self.eps)


class MLP:
    def __init__(self, layer, fast_swiglu):
        self.gate = QLinear(layer.gate_proj)
        self.up = QLinear(layer.up_proj)
        self.down = QLinear(layer.down_proj)
        self.fast = fast_swiglu and HAS_METAL

    def __call__(self, x):
        gate, up = self.gate(x), self.up(x)
        hidden = metal_swiglu(gate, up) if self.fast else swiglu_ref(gate, up)
        return self.down(hidden)


class Attention:
    def __init__(self, layer, args, flags):
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = getattr(
            args, "head_dim", args.hidden_size // self.n_heads
        )
        self.scale = self.head_dim**-0.5
        self.rope_theta = args.rope_theta
        self.fast_rope = flags["fast_rope"] and HAS_METAL
        self.tiled = flags["tiled_prefill"] and HAS_METAL
        self.q_proj = QLinear(layer.q_proj)
        self.k_proj = QLinear(layer.k_proj)
        self.v_proj = QLinear(layer.v_proj)
        self.o_proj = QLinear(layer.o_proj)
        self.q_norm = (
            Norm(layer.q_norm.weight, args.rms_norm_eps, flags["fast_rmsnorm"])
            if hasattr(layer, "q_norm")
            else None
        )
        self.k_norm = (
            Norm(layer.k_norm.weight, args.rms_norm_eps, flags["fast_rmsnorm"])
            if hasattr(layer, "k_norm")
            else None
        )

    def _rope(self, x, offsets):
        if self.fast_rope:
            return metal_rope(x, offsets, self.rope_theta)
        return rope_ref(x, offsets, self.rope_theta)

    def __call__(self, x):
        b, length, _ = x.shape
        q = self.q_proj(x).reshape(b, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, length, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)
        offsets = mx.zeros((b,), dtype=mx.int32)
        q = self._rope(q, offsets).transpose(0, 2, 1, 3)
        k = self._rope(k, offsets).transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        repeats = self.n_heads // self.n_kv_heads
        if repeats > 1:
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        if self.tiled and length >= 8 and self.head_dim == 128:
            out = tiled_attention(q, k, v, self.scale, causal=True)
        else:
            scores = mx.matmul(q, k.swapaxes(-2, -1)) * self.scale
            mask = mx.tril(mx.ones((length, length)), 0)
            scores = scores + mx.where(mask, 0.0, -mx.inf)[None, None]
            probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
            out = mx.matmul(probs, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(out)


class Block:
    def __init__(self, layer, args, flags):
        self.input_norm = Norm(
            layer.input_layernorm.weight, args.rms_norm_eps, flags["fast_rmsnorm"]
        )
        self.attn = Attention(layer.self_attn, args, flags)
        self.post_norm = Norm(
            layer.post_attention_layernorm.weight,
            args.rms_norm_eps,
            flags["fast_rmsnorm"],
        )
        self.mlp = MLP(layer.mlp, flags["fast_swiglu"])

    def __call__(self, x):
        x = x + self.attn(self.input_norm(x))
        return x + self.mlp(self.post_norm(x))


class SelectedModel:
    def __init__(self, mlx_model, checkpoint: str):
        if checkpoint not in CHECKPOINTS:
            raise ValueError(f"unknown checkpoint {checkpoint}")
        self.checkpoint = checkpoint
        self.flags = features(checkpoint)
        self.args = mlx_model.args
        self.embed = mlx_model.model.embed_tokens
        self.layers = [
            Block(layer, self.args, self.flags)
            for layer in mlx_model.model.layers
        ]
        self.norm = Norm(
            mlx_model.model.norm.weight,
            self.args.rms_norm_eps,
            self.flags["fast_rmsnorm"],
        )
        if getattr(self.args, "tie_word_embeddings", True):
            self.lm_head = self.embed
        else:
            self.lm_head = QLinear(mlx_model.lm_head)

    def __call__(self, ids):
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        if isinstance(self.lm_head, QLinear):
            return self.lm_head(h)
        return self.lm_head.as_linear(h)


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


def main():
    section("20. Week 2 selected-модель")
    model_id = os.environ.get("LAB_MODEL", MODEL_ID)
    print(f"  модель: {model_id}")
    mlx_model, tok = load(model_id)
    ids = mx.array(tok.encode(
        "Explain what a KV cache is in one sentence.",
        add_special_tokens=False,
    ))[None]
    print(f"  prompt: {ids.shape[1]} токенов")

    subsection("A. Checkpoint-флаги")
    print(
        f"  {'checkpoint':<17} {'cap':>4} {'W4':>4} {'simd':>5} "
        f"{'rms':>4} {'rope':>5} {'swiglu':>7} {'tiled':>6}"
    )
    for name in CHECKPOINTS:
        f = features(name)
        print(
            f"  {name:<17} {int(f['capacity_cache']):>4} "
            f"{int(f['quantized_weights']):>4} {int(f['simd_matmul']):>5} "
            f"{int(f['fast_rmsnorm']):>4} {int(f['fast_rope']):>5} "
            f"{int(f['fast_swiglu']):>7} {int(f['tiled_prefill']):>6}"
        )

    subsection("B. Logits selected vs mlx_lm")
    model = SelectedModel(mlx_model, "selected")
    ref = mlx_model(ids)
    out = model(ids)
    mx.eval(ref, out)
    ref_top5 = [int(x) for x in mx.argsort(-ref[0, -1])[:5]]
    out_top5 = [int(x) for x in mx.argsort(-out[0, -1])[:5]]
    if ref_top5 != out_top5:
        raise SystemExit(f"top-5 не совпал: {out_top5} != {ref_top5}")
    diff = float(mx.max(mx.abs(out.astype(mx.float32) - ref.astype(mx.float32))))
    print(f"  ✓ top-5: {out_top5}")
    print(f"  max |diff| = {diff:.4f}")

    subsection("C. Стоимость checkpoint'ов")
    print(f"  {'checkpoint':<17} {'prefill ms':>11} {'vs mlx':>8}")
    t_mlx = _bench(lambda: mlx_model(ids))
    for name in ("kv-cache", "quantized-matvec", "rmsnorm", "swiglu", "tiled-prefill", "selected"):
        m = SelectedModel(mlx_model, name)
        t = _bench(lambda m=m: m(ids), repeats=2)
        print(f"  {name:<17} {t * 1e3:>11.2f} {t_mlx / t:>7.2f}x")
    note(
        "kv-cache/capacity-cache — decode-признаки; в prefill-модели они "
        "только отмечены. Реальный decode-путь — inference/24 и inference/23."
    )
    note(
        "selected включает fused RMSNorm/RoPE/SwiGLU и tiled prefill "
        "attention. На коротком prompt tiled-путь может быть не быстрее "
        "MLX SDPA, но это тот же оператор, что в model/16."
    )

    section("Вывод")
    note("Один класс SelectedModel переключает признаки по checkpoint.")
    note("selected — это cumulative feature set, а не отдельная копия кода.")
    note("Top-5 logits совпадают с mlx_lm в пределах float16-допуска.")
    note("Осталось только прогнать decode-часть через capacity-cache.")


if __name__ == "__main__":
    main()
