# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 16: tiled dense prefill attention с simdgroup_matrix.

Идея
----
В 37 мы ходили по страницам одним threadgroup на query. Здесь —
dense prefill attention, нарезанный на плитки:

* один threadgroup считает 8 query-строк;
* QK^T считается через `simdgroup_matrix` (8×8×8 MMA);
* scores проходят causal mask и online-softmax;
* P·V аккумулируется по key-плиткам;
* короткие/частичные плитки добиваются нулями.

P·V здесь скалярный: так проще увидеть online-softmax и rescale.
Production-ядро tiny-llm делает и P·V через MMA, но принцип тот же.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/16_tiled_dense_prefill_attention.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import note, section, subsection

_SOURCE = r"""
    const int lane = thread_index_in_threadgroup;
    const int tile_idx = threadgroup_position_in_grid.y;
    const int tiles_per_head = (L + 7) / 8;
    const int b = tile_idx / (HQ * tiles_per_head);
    const int rem = tile_idx - b * (HQ * tiles_per_head);
    const int h = rem / tiles_per_head;
    const int qm0 = (rem - h * tiles_per_head) * 8;
    const int kvh = h / (HQ / HKV);

    threadgroup T q_tile[8 * D];
    threadgroup T kt_tile[D * 8];
    threadgroup T v_tile[8 * D];
    threadgroup float s_tile[64];
    threadgroup float acc[8 * D];
    threadgroup float row_max[8];
    threadgroup float row_sum[8];
    threadgroup float alpha_row[8];
    threadgroup float beta_row[8];

    // Q tile: [8, D], zero-padded за L.
    for (int idx = lane; idx < 8 * D; idx += 32) {
        const int r = idx / D;
        const int d = idx - r * D;
        const int m = qm0 + r;
        q_tile[idx] = m < L
            ? q[((b * HQ + h) * L + m) * D + d]
            : T(0);
        acc[idx] = 0.0f;
    }
    if (lane < 8) {
        row_max[lane] = -1.0e30f;
        row_sum[lane] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k0 = 0; k0 < S; k0 += 8) {
        // K^T tile: [D, 8].
        for (int idx = lane; idx < D * 8; idx += 32) {
            const int d = idx / 8;
            const int kk = idx - d * 8;
            const int kg = k0 + kk;
            kt_tile[idx] = kg < S
                ? k[((b * HKV + kvh) * S + kg) * D + d]
                : T(0);
        }
        // V tile: [8, D].
        for (int idx = lane; idx < 8 * D; idx += 32) {
            const int kk = idx / D;
            const int d = idx - kk * D;
            const int kg = k0 + kk;
            v_tile[idx] = kg < S
                ? v[((b * HKV + kvh) * S + kg) * D + d]
                : T(0);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // S = Q @ K^T, 8×8, через MMA.
        simdgroup_matrix<float, 8, 8> sacc;
        sacc.thread_elements()[0] = 0.0f;
        sacc.thread_elements()[1] = 0.0f;
        for (int d0 = 0; d0 < D; d0 += 8) {
            simdgroup_matrix<T, 8, 8> qm;
            simdgroup_matrix<T, 8, 8> km;
            simdgroup_load(qm, q_tile + d0, D);
            simdgroup_load(km, kt_tile + d0 * 8, 8);
            simdgroup_multiply_accumulate(sacc, qm, km, sacc);
        }
        simdgroup_store(sacc, s_tile, 8);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Causal mask + online softmax по строкам.
        if (lane < 8) {
            const int r = lane;
            const int qpos = qm0 + r;
            int visible = S;
            if (CAUSAL) {
                visible = S - L + qpos + 1;
                visible = clamp(visible, 0, S);
            }
            if (qpos >= L) visible = 0;

            float tile_max = -1.0e30f;
            for (int c = 0; c < 8; c++) {
                const int kg = k0 + c;
                if (kg < visible) {
                    tile_max = max(tile_max, s_tile[r * 8 + c] * scale);
                }
            }
            float tile_sum = 0.0f;
            for (int c = 0; c < 8; c++) {
                const int kg = k0 + c;
                if (kg < visible) {
                    const float p = metal::exp(s_tile[r * 8 + c] * scale - tile_max);
                    s_tile[r * 8 + c] = p;
                    tile_sum += p;
                } else {
                    s_tile[r * 8 + c] = 0.0f;
                }
            }

            const float new_max = max(row_max[r], tile_max);
            float alpha = 1.0f;
            float beta = 0.0f;
            if (tile_sum > 0.0f) {
                alpha = row_max[r] <= -1.0e29f
                    ? 0.0f
                    : metal::exp(row_max[r] - new_max);
                beta = metal::exp(tile_max - new_max);
            }
            row_sum[r] = row_sum[r] * alpha + tile_sum * beta;
            row_max[r] = tile_sum > 0.0f ? new_max : row_max[r];
            alpha_row[r] = alpha;
            beta_row[r] = beta;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // Rescale старого accumulator и добавить P·V.
        for (int idx = lane; idx < 8 * D; idx += 32) {
            const int r = idx / D;
            const int d = idx - r * D;
            float add = 0.0f;
            for (int c = 0; c < 8; c++) {
                add += s_tile[r * 8 + c] * beta_row[r] *
                    static_cast<float>(v_tile[c * D + d]);
            }
            acc[idx] = acc[idx] * alpha_row[r] + add;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int idx = lane; idx < 8 * D; idx += 32) {
        const int r = idx / D;
        const int d = idx - r * D;
        const int m = qm0 + r;
        if (m < L) {
            out[((b * HQ + h) * L + m) * D + d] = static_cast<T>(
                row_sum[r] == 0.0f ? 0.0f : acc[idx] / row_sum[r]);
        }
    }
"""


def _make_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="tiled_dense_prefill_attention",
        input_names=["q", "k", "v", "scale"],
        output_names=["out"],
        source=_SOURCE,
        header="#include <metal_simdgroup_matrix>\n",
    )


_KERNEL = _make_kernel()


def tiled_attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float,
    causal: bool = True,
) -> mx.array:
    if _KERNEL is None:
        raise RuntimeError("Metal недоступен")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("ожидается q/k/v [B,H,L,D]")
    b, hq, length, d = q.shape
    kb, hkv, s, kd = k.shape
    if kb != b or k.shape != v.shape or kd != d:
        raise ValueError("несовместимые формы k/v")
    if hq % hkv != 0:
        raise ValueError("H_q должно делиться на H_kv")
    out = _KERNEL(
        inputs=[mx.contiguous(q), mx.contiguous(k), mx.contiguous(v),
                mx.array(scale, dtype=mx.float32)],
        template=[
            ("T", q.dtype),
            ("HQ", hq),
            ("HKV", hkv),
            ("L", length),
            ("S", s),
            ("D", d),
            ("CAUSAL", 1 if causal else 0),
        ],
        grid=(32, b * hq * ((length + 7) // 8), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]
    return out


def dense_reference(q, k, v, scale, causal=True):
    b, hq, length, d = q.shape
    hkv = k.shape[1]
    repeats = hq // hkv
    k = mx.repeat(k, repeats, axis=1)
    v = mx.repeat(v, repeats, axis=1)
    scores = mx.matmul(q, k.swapaxes(-2, -1)) * scale
    if causal:
        rows = mx.arange(length)[:, None]
        cols = mx.arange(k.shape[2])[None, :]
        mask = mx.where(
            cols <= (k.shape[2] - length + rows), 0.0, -mx.inf
        ).astype(scores.dtype)
        scores = scores + mask[None, None]
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    return mx.matmul(probs, v)


def _assert_close(name, actual, expected):
    ok = mx.allclose(
        actual.astype(mx.float32),
        expected.astype(mx.float32),
        rtol=3e-2,
        atol=3e-2,
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


def correctness():
    subsection("A. Kernel vs dense-эталон")
    cases = [
        # L, S, Hq, Hkv, D, causal
        (8, 16, 4, 2, 64, True),
        (16, 35, 4, 2, 64, True),
        (9, 17, 8, 2, 128, True),
        (8, 16, 4, 2, 64, False),
    ]
    for i, (length, s, hq, hkv, d, causal) in enumerate(cases):
        mx.random.seed(i + 1)
        q = mx.random.normal((1, hq, length, d)).astype(mx.bfloat16)
        k = mx.random.normal((1, hkv, s, d)).astype(mx.bfloat16)
        v = mx.random.normal((1, hkv, s, d)).astype(mx.bfloat16)
        out = tiled_attention(q, k, v, d**-0.5, causal)
        ref = dense_reference(q, k, v, d**-0.5, causal)
        _assert_close(
            f"L={length}, S={s}, H={hq}/{hkv}, D={d}, causal={causal}",
            out,
            ref,
        )


def benchmark():
    subsection("B. Prefill-плитки")
    for length, s, hq, hkv, d in ((64, 64, 16, 2, 128), (128, 128, 16, 2, 128)):
        mx.random.seed(7)
        q = mx.random.normal((1, hq, length, d)).astype(mx.bfloat16)
        k = mx.random.normal((1, hkv, s, d)).astype(mx.bfloat16)
        v = mx.random.normal((1, hkv, s, d)).astype(mx.bfloat16)
        t_kernel = _bench(lambda: tiled_attention(q, k, v, d**-0.5, True))
        t_ref = _bench(lambda: dense_reference(q, k, v, d**-0.5, True), repeats=2)
        print(
            f"  L={length:>3}, S={s:>3}, H={hq}/{hkv}, D={d}: "
            f"kernel {t_kernel * 1e3:7.2f} ms, dense {t_ref * 1e3:7.2f} ms"
        )
    note(
        "8 query-строк на threadgroup — это tile по Q. QK^T идёт через MMA, "
        "P·V — скалярно; online-softmax переносит max/sum/acc между key-плитками."
    )
    note(
        "Production-ядро tiny-llm делает 32×32 плитки, несколько SIMD-групп "
        "и P·V через MMA. Здесь важна механика, а не последний процент."
    )


def main():
    section("16. Tiled dense prefill attention")
    if _KERNEL is None:
        note("Metal недоступен: лаборатория пропущена.")
        return
    print(f"  устройство: {mx.device_info().get('device_name', 'unknown')}")
    correctness()
    benchmark()

    section("Вывод")
    note("Query tile 8×D, key tile 8×D; QK^T считается simdgroup_matrix.")
    note("Causal mask и online-softmax обновляют max/sum/acc по плиткам.")
    note("P·V можно тоже перевести на MMA — это следующий шаг.")
    note("Такой оператор — основа tiled-prefill/selected в Week 2.")


if __name__ == "__main__":
    main()
