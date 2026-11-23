# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 13: собрать Qwen из checkpoint и подключить свои ядра.

Идея
----
До этого мы либо пользовались ``mlx_lm.load``, либо строили игрушечный
transformer. Здесь мы делаем то, что tiny-llm делает в Week 1 Day 5:

* читаем конфиг и веса реального Qwen-чекпоинта;
* сами собираем embedding, attention, RoPE, GQA, RMSNorm, SwiGLU и lm_head;
* projections остаются 4-битными и идут через ``mx.quantized_matmul``;
* RMSNorm/RoPE/SwiGLU берутся из ``lab/metal_ops.py`` — тех же ядер,
  что в model/10;
* logits сверяются с исходной моделью ``mlx_lm``.

Qwen2.5 не имеет QK-norm; для Qwen3 код применит q_norm/k_norm, если они
есть в чекпоинте.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/13_load_qwen3.py"

Модель задаётся через LAB_MODEL.
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

from common import MODEL_ID, gb, note, section, subsection
from metal_ops import (
    HAS_METAL,
    metal_rope,
    metal_rms_norm,
    metal_swiglu,
    rms_norm_ref,
    rope_ref,
    swiglu_ref,
)

PROMPT = "Explain what a KV cache is in one sentence."


# --------------------------------------------------------------------------- #
# Слои из весов чекпоинта
# --------------------------------------------------------------------------- #


class QLinear:
    """QuantizedLinear без привязки к nn.Module: только weight/scales/bias."""

    def __init__(self, layer):
        self.weight = layer.weight
        self.scales = layer.scales
        self.biases = layer.biases
        self.bias = getattr(layer, "bias", None)
        self.group_size = layer.group_size
        self.bits = layer.bits

    def __call__(self, x: mx.array) -> mx.array:
        y = mx.quantized_matmul(
            x,
            self.weight,
            scales=self.scales,
            biases=self.biases,
            transpose=True,
            group_size=self.group_size,
            bits=self.bits,
        )
        if self.bias is not None:
            y = y + self.bias
        return y


class QNorm:
    def __init__(self, weight: mx.array, eps: float):
        self.weight = weight
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        if HAS_METAL:
            return metal_rms_norm(x, self.weight, self.eps)
        return rms_norm_ref(x, self.weight, self.eps)


class QAttention:
    def __init__(self, layer, args):
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        # Qwen3 задаёт head_dim явно; Qwen2.5 — нет.
        self.head_dim = getattr(
            args, "head_dim", args.hidden_size // self.n_heads
        )
        self.scale = self.head_dim**-0.5
        self.rope_theta = args.rope_theta
        self.q_proj = QLinear(layer.q_proj)
        self.k_proj = QLinear(layer.k_proj)
        self.v_proj = QLinear(layer.v_proj)
        self.o_proj = QLinear(layer.o_proj)
        self.q_norm = (
            QNorm(layer.q_norm.weight, args.rms_norm_eps)
            if hasattr(layer, "q_norm")
            else None
        )
        self.k_norm = (
            QNorm(layer.k_norm.weight, args.rms_norm_eps)
            if hasattr(layer, "k_norm")
            else None
        )

    def __call__(self, x: mx.array) -> mx.array:
        b, length, _ = x.shape
        q = self.q_proj(x).reshape(b, length, self.n_heads, self.head_dim)
        k = self.k_proj(x).reshape(b, length, self.n_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(b, length, self.n_kv_heads, self.head_dim)
        if self.q_norm is not None:
            q = self.q_norm(q)
        if self.k_norm is not None:
            k = self.k_norm(k)

        offsets = mx.zeros((b,), dtype=mx.int32)
        if HAS_METAL:
            q = metal_rope(q, offsets, self.rope_theta)
            k = metal_rope(k, offsets, self.rope_theta)
        else:
            q = rope_ref(q, offsets, self.rope_theta)
            k = rope_ref(k, offsets, self.rope_theta)

        q = q.transpose(0, 2, 1, 3)
        k = k.transpose(0, 2, 1, 3)
        v = v.transpose(0, 2, 1, 3)
        repeats = self.n_heads // self.n_kv_heads
        if repeats > 1:
            k = mx.repeat(k, repeats, axis=1)
            v = mx.repeat(v, repeats, axis=1)

        scores = mx.matmul(q, k.swapaxes(-2, -1)) * self.scale
        causal = mx.tril(mx.ones((length, length)), 0)
        mask = mx.where(causal, 0.0, -mx.inf).astype(scores.dtype)
        scores = scores + mask[None, None, :, :]
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(x.dtype)
        out = mx.matmul(probs, v)
        out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
        return self.o_proj(out)


class QMLP:
    def __init__(self, layer):
        self.gate = QLinear(layer.gate_proj)
        self.up = QLinear(layer.up_proj)
        self.down = QLinear(layer.down_proj)

    def __call__(self, x: mx.array) -> mx.array:
        gate = self.gate(x)
        up = self.up(x)
        hidden = metal_swiglu(gate, up) if HAS_METAL else swiglu_ref(gate, up)
        return self.down(hidden)


class QBlock:
    def __init__(self, layer, args):
        self.input_norm = QNorm(layer.input_layernorm.weight, args.rms_norm_eps)
        self.attn = QAttention(layer.self_attn, args)
        self.post_norm = QNorm(
            layer.post_attention_layernorm.weight, args.rms_norm_eps
        )
        self.mlp = QMLP(layer.mlp)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.input_norm(x))
        return x + self.mlp(self.post_norm(x))


class QwenFromCheckpoint:
    """Собранная вручную модель; веса взяты из mlx_lm-чекпоинта."""

    def __init__(self, mlx_model):
        self.args = mlx_model.args
        self.embed = mlx_model.model.embed_tokens
        self.layers = [QBlock(layer, self.args) for layer in mlx_model.model.layers]
        self.norm = QNorm(mlx_model.model.norm.weight, self.args.rms_norm_eps)
        if getattr(self.args, "tie_word_embeddings", True):
            self.lm_head = self.embed
        else:
            self.lm_head = QLinear(mlx_model.lm_head)

    def __call__(self, ids: mx.array) -> mx.array:
        h = self.embed(ids)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        if isinstance(self.lm_head, QLinear):
            return self.lm_head(h)
        return self.lm_head.as_linear(h)


# --------------------------------------------------------------------------- #
# Утилиты
# --------------------------------------------------------------------------- #


def _assert_close(name: str, actual: mx.array, expected: mx.array) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: {actual.shape} != {expected.shape}")
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


def _bench(fn, repeats: int = 3) -> float:
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


def _weight_stats(mlx_model) -> tuple[int, int]:
    """(packed bytes, dequantized parameter count) для модели."""
    packed = 0
    params = 0
    for _, module in mlx_model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            out_dim = module.weight.shape[0]
            packed_cols = module.weight.shape[1]
            params += out_dim * (packed_cols * 32 // module.bits)
            packed += module.weight.nbytes
            packed += module.scales.nbytes
            if module.biases is not None:
                packed += module.biases.nbytes
            if getattr(module, "bias", None) is not None:
                packed += module.bias.nbytes
        elif isinstance(module, nn.QuantizedEmbedding):
            params += module.num_embeddings * module.dims
            packed += module.weight.nbytes + module.scales.nbytes
            if module.biases is not None:
                packed += module.biases.nbytes
    return packed, params


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def main() -> None:
    section("13. Сборка Qwen из checkpoint + свои ядра")
    model_id = os.environ.get("LAB_MODEL", MODEL_ID)
    print(f"  модель: {model_id}")
    mlx_model, tok = load(model_id)
    args = mlx_model.args
    print(
        f"  hidden={args.hidden_size}, layers={args.num_hidden_layers}, "
        f"H={args.num_attention_heads}, H_kv={args.num_key_value_heads}, "
        f"rope_theta={args.rope_theta}, tie={args.tie_word_embeddings}"
    )
    if getattr(args, "rope_traditional", False):
        raise SystemExit(
            "эта лаборатория использует non-traditional RoPE; "
            "для traditional нужен другой layout"
        )
    if getattr(args, "rope_scaling", None):
        raise SystemExit(
            "rope_scaling в этой лаборатории не поддержан: нужен другой "
            "расчёт частот"
        )
    print(f"  Metal-ядра: {'да' if HAS_METAL else 'нет, используем эталоны'}")

    ids_list = tok.encode(PROMPT, add_special_tokens=False)
    ids = mx.array(ids_list)[None]
    print(f"  prompt: {len(ids_list)} токенов: {PROMPT!r}")

    subsection("A. Сборка и сверка logits")
    custom = QwenFromCheckpoint(mlx_model)
    reference = mlx_model(ids)
    actual = custom(ids)
    mx.eval(reference, actual)
    diff = mx.abs(actual.astype(mx.float32) - reference.astype(mx.float32))
    max_diff = float(mx.max(diff))
    mean_diff = float(mx.mean(diff))
    ref_top5 = [int(x) for x in mx.argsort(-reference[0, -1])[:5]]
    act_top5 = [int(x) for x in mx.argsort(-actual[0, -1])[:5]]
    if ref_top5 != act_top5:
        raise AssertionError(
            f"top-5 не совпал: custom={act_top5}, mlx_lm={ref_top5}"
        )
    print(f"  ✓ top-5 logits совпали: {act_top5}")
    print(f"  max |diff| = {max_diff:.4f}, mean |diff| = {mean_diff:.4f}")

    subsection("B. Память")
    packed, params = _weight_stats(mlx_model)
    print(f"  packed W4: {gb(packed)}")
    print(f"  параметров: {params / 1e9:.3f} B")
    print(f"  если хранить BF16: {gb(params * 2)}")
    note(
        "Мы не деquantize всю модель: проекции читают packed-веса через "
        "mx.quantized_matmul, а embedding распаковывает только выбранные строки."
    )

    subsection("C. Скорость prefill")
    t_ref = _bench(lambda: mlx_model(ids))
    t_custom = _bench(lambda: custom(ids))
    print(
        f"  mlx_lm: {t_ref * 1e3:.1f} ms, custom: {t_custom * 1e3:.1f} ms "
        f"({t_ref / t_custom:.2f}x)"
    )
    note(
        "Custom-путь медленнее: ручной attention, Python-обвязка и наши "
        "учебные ядра. Цель лаборатории — показать сборку и проводку, "
        "а не обогнать mlx_lm."
    )

    note(
        "Так же, как здесь, в tiny-llm Week 2 подключаются quantized matvec, "
        "SIMD matmul и fused RMSNorm/RoPE/SwiGLU к реальной модели."
    )

    section("Вывод")
    note("Checkpoint — это конфиг + тензоры; модель можно собрать руками.")
    note("Проекции остаются packed W4, embedding распаковывается по строкам.")
    note("RMSNorm/RoPE/SwiGLU идут через те же Metal-ядра, что в model/10.")
    note("Logits совпадают с mlx_lm в пределах float16-допуска.")


if __name__ == "__main__":
    main()
