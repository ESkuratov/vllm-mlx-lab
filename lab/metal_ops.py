# SPDX-License-Identifier: Apache-2.0
"""Общие Metal-ядра для лабораторий: RMSNorm, RoPE, SwiGLU.

Ядра компилируются через ``mx.fast.metal_kernel``: отдельная сборка
расширения не нужна. Лаборатория 33 сравнивает их с MLX, лаборатория 38
использует их в собранной модели Qwen.
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

ROPE_BASE = 10000.0
RMS_EPS = 1.0e-5

_RMSNORM_SOURCE = r"""
    const int lane = thread_position_in_grid.x;
    const int row = thread_position_in_grid.y;

    float ss = 0.0f;
    for (int d = lane; d < D; d += 32) {
        float v = static_cast<float>(x[row * D + d]);
        ss += v * v;
    }
    ss = simd_sum(ss);
    const float eps = eps_in;
    const float rms = metal::rsqrt(ss / static_cast<float>(D) + eps);

    for (int d = lane; d < D; d += 32) {
        out[row * D + d] = static_cast<T>(
            static_cast<float>(x[row * D + d]) * rms *
            static_cast<float>(weight[d]));
    }
"""

_ROPE_SOURCE = r"""
    const int i = thread_position_in_grid.x;
    const int h = thread_position_in_grid.y;
    const int t = thread_position_in_grid.z;
    const int b = t / S;
    const int s = t - b * S;
    const int half_d = D / 2;

    const float freq = static_cast<float>(inv_freq[i]);
    const float angle = static_cast<float>(offsets[b] + s) * freq;
    const float c = metal::cos(angle);
    const float sn = metal::sin(angle);

    const int xbase = (t * H + h) * D;
    const float x1 = static_cast<float>(x[xbase + i]);
    const float x2 = static_cast<float>(x[xbase + i + half_d]);
    out[xbase + i] = static_cast<T>(x1 * c - x2 * sn);
    out[xbase + i + half_d] = static_cast<T>(x2 * c + x1 * sn);
"""

_SWIGLU_SOURCE = r"""
    uint i = thread_position_in_grid.x;
    const float g = static_cast<float>(gate[i]);
    const float u = static_cast<float>(up[i]);
    const float z = metal::exp(-metal::abs(g));
    const float sig = g < 0.0f ? z / (1.0f + z) : 1.0f / (1.0f + z);
    out[i] = static_cast<T>(g * sig * u);
"""


def _make_kernels():
    if not mx.metal.is_available():
        return None, None, None
    rmsnorm = mx.fast.metal_kernel(
        name="shared_rmsnorm",
        input_names=["x", "weight", "eps_in"],
        output_names=["out"],
        source=_RMSNORM_SOURCE,
    )
    rope = mx.fast.metal_kernel(
        name="shared_rope",
        input_names=["x", "offsets", "inv_freq"],
        output_names=["out"],
        source=_ROPE_SOURCE,
    )
    swiglu = mx.fast.metal_kernel(
        name="shared_swiglu",
        input_names=["gate", "up"],
        output_names=["out"],
        source=_SWIGLU_SOURCE,
    )
    return rmsnorm, rope, swiglu


_RMSNORM, _ROPE, _SWIGLU = _make_kernels()
HAS_METAL = _RMSNORM is not None


# --------------------------------------------------------------------------- #
# Metal-путь
# --------------------------------------------------------------------------- #


def metal_rms_norm(x: mx.array, weight: mx.array, eps: float = RMS_EPS) -> mx.array:
    if not HAS_METAL:
        raise RuntimeError("Metal недоступен")
    if x.shape[-1] != weight.shape[0]:
        raise ValueError("weight должен совпадать с последней осью x")
    original = x.shape
    rows = x.size // x.shape[-1]
    d = x.shape[-1]
    return _RMSNORM(
        inputs=[
            mx.contiguous(x).reshape(rows, d),
            mx.contiguous(weight.astype(x.dtype)),
            mx.array(eps, dtype=mx.float32),
        ],
        template=[("T", x.dtype), ("D", d)],
        grid=(32, rows, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(rows, d)],
        output_dtypes=[x.dtype],
    )[0].reshape(original)


def metal_rope(
    x: mx.array,
    offsets: mx.array,
    base: float = ROPE_BASE,
) -> mx.array:
    if not HAS_METAL:
        raise RuntimeError("Metal недоступен")
    if x.ndim != 4:
        raise ValueError("ожидается x [B, S, H, D]")
    b, s, h, d = x.shape
    if d % 2 != 0:
        raise ValueError("D должно быть чётным")
    if offsets.shape != (b,):
        raise ValueError("offsets должен иметь форму [B]")
    half = d // 2
    i = mx.arange(half, dtype=mx.float32)
    inv_freq = (base ** (-2.0 * i / d)).astype(mx.float32)
    return _ROPE(
        inputs=[mx.contiguous(x), offsets.astype(mx.int32), inv_freq],
        template=[("T", x.dtype), ("S", s), ("H", h), ("D", d)],
        grid=(half, h, b * s),
        threadgroup=(min(half, 32), 1, 1),
        output_shapes=[x.shape],
        output_dtypes=[x.dtype],
    )[0]


def metal_swiglu(gate: mx.array, up: mx.array) -> mx.array:
    if not HAS_METAL:
        raise RuntimeError("Metal недоступен")
    if gate.shape != up.shape:
        raise ValueError("gate и up должны совпадать по форме")
    return _SWIGLU(
        inputs=[mx.contiguous(gate), mx.contiguous(up)],
        template=[("T", gate.dtype)],
        grid=(gate.size, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[gate.shape],
        output_dtypes=[gate.dtype],
    )[0]


# --------------------------------------------------------------------------- #
# Эталоны
# --------------------------------------------------------------------------- #


def rms_norm_ref(x: mx.array, weight: mx.array, eps: float = RMS_EPS) -> mx.array:
    xf = x.astype(mx.float32)
    y = xf * mx.rsqrt(mx.mean(xf * xf, axis=-1, keepdims=True) + eps)
    return (y.astype(x.dtype) * weight).astype(x.dtype)


def rope_ref(x: mx.array, offsets: mx.array, base: float = ROPE_BASE) -> mx.array:
    b, s, _, d = x.shape
    half = d // 2
    i = mx.arange(half, dtype=mx.float32)
    inv_freq = (base ** (-2.0 * i / d)).astype(mx.float32)
    pos = offsets[:, None] + mx.arange(s)[None, :]
    angle = pos[:, :, None, None] * inv_freq[None, None, None, :]
    cos = mx.cos(angle).astype(x.dtype)
    sin = mx.sin(angle).astype(x.dtype)
    x1 = x[..., :half]
    x2 = x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def swiglu_ref(gate: mx.array, up: mx.array) -> mx.array:
    return nn.silu(gate) * up
