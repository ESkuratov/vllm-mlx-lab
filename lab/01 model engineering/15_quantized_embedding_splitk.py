# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 15: quantized embedding и split-K matmul.

Идея
----
Два оставшихся куска W4-пути tiny-llm:

1. **Quantized embedding** — не разворачивать всю таблицу [V, K], а
   распаковать только выбранные строки. Один поток на (token, dim).
2. **Split-K** — если output-плиток мало (маленький M), разбить reduction
   по K на несколько threadgroup и сложить частичные суммы. Это повышает
   occupancy, но добавляет reduction.

Оба ядра — через ``mx.fast.metal_kernel``. Split-K использует тот же
tile 8×8 и ``simdgroup_matrix``, что model/11.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/15_quantized_embedding_splitk.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import note, section, subsection

GROUP_SIZE = 128
BITS = 4

# --------------------------------------------------------------------------- #
# Metal sources
# --------------------------------------------------------------------------- #

_EMBED_SOURCE = r"""
    const int dim = thread_position_in_grid.x;
    const int token = thread_position_in_grid.y;
    const int row = indices[token];
    const int packed = w[row * (K / 8) + dim / 8];
    const int shift = (dim % 8) * 4;
    const float q = static_cast<float>((packed >> shift) & 0xFu);
    const int g = dim / GROUP_SIZE;
    const float scale = static_cast<float>(
        scales[row * (K / GROUP_SIZE) + g]);
    const float bias = static_cast<float>(
        biases[row * (K / GROUP_SIZE) + g]);
    out[token * K + dim] = static_cast<T>(q * scale + bias);
"""

_SPLITK_SOURCE = r"""
    const int lane = thread_index_in_threadgroup;
    const int z = threadgroup_position_in_grid.z;
    const int split = z % SPLIT;
    const int m0 = (z / SPLIT) * 8;
    const int n0 = threadgroup_position_in_grid.y * 8;

    threadgroup T a_tile[64];
    threadgroup T b_tile[64];
    threadgroup float c_tile[64];

    simdgroup_matrix<float, 8, 8> acc;
    acc.thread_elements()[0] = 0.0f;
    acc.thread_elements()[1] = 0.0f;

    for (int k0 = K_START; k0 < K_END; k0 += 8) {
        for (int idx = lane; idx < 64; idx += 32) {
            const int mm = idx / 8;
            const int kk = idx - mm * 8;
            const int m = m0 + mm;
            const int k = k0 + kk;
            a_tile[idx] = (m < M && k < K_END)
                ? a[m * K + k]
                : T(0);
        }
        for (int idx = lane; idx < 64; idx += 32) {
            const int kk = idx / 8;
            const int nn = idx - kk * 8;
            const int k = k0 + kk;
            const int n = n0 + nn;
            if (k < K_END && n < N) {
                const uint32_t packed = w[n * (K / 8) + k / 8];
                const int shift = (k % 8) * 4;
                const float q = static_cast<float>((packed >> shift) & 0xFu);
                const int g = k / GROUP_SIZE;
                const float scale = static_cast<float>(
                    scales[n * (K / GROUP_SIZE) + g]);
                const float bias = static_cast<float>(
                    biases[n * (K / GROUP_SIZE) + g]);
                b_tile[idx] = static_cast<T>(q * scale + bias);
            } else {
                b_tile[idx] = T(0);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_matrix<T, 8, 8> am;
        simdgroup_matrix<T, 8, 8> bm;
        simdgroup_load(am, a_tile, 8);
        simdgroup_load(bm, b_tile, 8);
        simdgroup_multiply_accumulate(acc, am, bm, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    simdgroup_store(acc, c_tile, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = lane; idx < 64; idx += 32) {
        const int mm = idx / 8;
        const int nn = idx - mm * 8;
        const int m = m0 + mm;
        const int n = n0 + nn;
        if (m < M && n < N) {
            partial[(split * M + m) * N + n] = static_cast<float>(c_tile[idx]);
        }
    }
"""


def _make_kernels():
    if not mx.metal.is_available():
        return None, None
    embed = mx.fast.metal_kernel(
        name="w4_embedding",
        input_names=["indices", "w", "scales", "biases"],
        output_names=["out"],
        source=_EMBED_SOURCE,
    )
    splitk = mx.fast.metal_kernel(
        name="w4_splitk_mma",
        input_names=["a", "w", "scales", "biases"],
        output_names=["partial"],
        source=_SPLITK_SOURCE,
        header="#include <metal_simdgroup_matrix>\n",
    )
    return embed, splitk


_EMBED, _SPLITK = _make_kernels()


# --------------------------------------------------------------------------- #
# Операторы
# --------------------------------------------------------------------------- #


def quantized_embedding(indices, wq, scales, biases, group_size=GROUP_SIZE):
    if indices.ndim != 1 or indices.dtype != mx.int32:
        raise ValueError("indices должен быть 1D int32")
    vocab, packed_cols = wq.shape
    k = packed_cols * 8
    if scales.shape != (vocab, k // group_size):
        raise ValueError("форма scales не совпадает")
    if biases.shape != scales.shape:
        raise ValueError("форма biases не совпадает")
    out = _EMBED(
        inputs=[mx.contiguous(indices), mx.contiguous(wq),
                mx.contiguous(scales), mx.contiguous(biases)],
        template=[
            ("T", scales.dtype),
            ("K", k),
            ("GROUP_SIZE", group_size),
        ],
        grid=(k, indices.size, 1),
        threadgroup=(min(k, 256), 1, 1),
        output_shapes=[(indices.size, k)],
        output_dtypes=[scales.dtype],
    )[0]
    return out


def _check(x, wq, scales, biases, group_size=GROUP_SIZE):
    m, k = x.shape
    n, packed_cols = wq.shape
    if packed_cols != k // 8:
        raise ValueError("упакованная ширина не совпадает")
    if scales.shape != (n, k // group_size):
        raise ValueError("форма scales не совпадает")
    if biases.shape != scales.shape:
        raise ValueError("форма biases не совпадает")
    return m, n, k


def w4_splitk(x, wq, scales, biases, split=1, group_size=GROUP_SIZE):
    """Вернуть [M, N]: сумма partial-плиток по split-K."""
    m, n, k = _check(x, wq, scales, biases, group_size)
    if split <= 0:
        raise ValueError("split must be positive")
    chunk = (k + split - 1) // split
    chunk = ((chunk + 7) // 8) * 8
    partials = []
    for s in range(split):
        start = s * chunk
        end = min(k, start + chunk)
        if start >= k:
            break
        partial = _SPLITK(
            inputs=[x, wq, scales, biases],
            template=[
                ("T", x.dtype),
                ("M", m),
                ("N", n),
                ("K", k),
                ("K_START", start),
                ("K_END", end),
                ("SPLIT", split),
                ("GROUP_SIZE", group_size),
            ],
            grid=(32, (n + 7) // 8, ((m + 7) // 8) * split),
            threadgroup=(32, 1, 1),
            output_shapes=[(split * m * n,)],
            output_dtypes=[mx.float32],
        )[0]
        partials.append(partial.reshape(split, m, n)[s])
    return mx.sum(mx.stack(partials, axis=0), axis=0).astype(x.dtype)


def mlx_reference(x, wq, scales, biases, group_size=GROUP_SIZE):
    return mx.quantized_matmul(
        x, wq, scales=scales, biases=biases, transpose=True,
        group_size=group_size, bits=BITS,
    )


# --------------------------------------------------------------------------- #
# Проверки и замеры
# --------------------------------------------------------------------------- #


def _assert_close(name, actual, expected):
    ok = mx.allclose(
        actual.astype(mx.float32),
        expected.astype(mx.float32),
        rtol=2e-2,
        atol=2e-2,
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


def embedding_check():
    subsection("A. Quantized embedding: только выбранные строки")
    mx.random.seed(1)
    vocab, k = 1024, 2048
    w = (mx.random.normal((vocab, k)) * 0.02).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    ids = mx.array([0, 7, 42, 1000], dtype=mx.int32)
    out = quantized_embedding(ids, wq, scales, biases)
    ref = mx.dequantize(
        wq[ids], scales[ids], biases[ids],
        group_size=GROUP_SIZE, bits=BITS,
    )
    _assert_close("embedding vs dequantize rows", out, ref)
    print(
        f"  таблица: {w.nbytes / 1024**2:.1f} MB BF16; "
        f"4 строки распакованы без полного dequantize"
    )


def splitk_check():
    subsection("B. Split-K: корректность")
    mx.random.seed(2)
    m, n, k = 8, 2048, 2048
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
    ref = mlx_reference(x, wq, scales, biases)
    for split in (1, 2, 4, 8):
        _assert_close(f"split={split}", w4_splitk(x, wq, scales, biases, split), ref)


def splitk_benchmark(n=2048, k=2048, label="заполненный grid"):
    subsection(f"C. Split-K: M-sweep, {label}")
    mx.random.seed(3)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(wq, scales, biases)
    print(
        f"  N={n}, K={k}; время в мс; "
        f"packed-вес {wq.nbytes / 1024**2:.1f} MB"
    )
    print(
        f"  {'M':>4} {'s=1':>9} {'s=2':>9} {'s=4':>9} {'s=8':>9} "
        f"{'MLX':>9}"
    )
    for m in (1, 8, 32, 128):
        x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
        times = []
        for split in (1, 2, 4, 8):
            times.append(_bench(lambda s=split: w4_splitk(x, wq, scales, biases, s), 2))
        t_mlx = _bench(lambda: mlx_reference(x, wq, scales, biases), 3)
        print(
            f"  {m:>4} " + " ".join(f"{t * 1e3:>9.3f}" for t in times)
            + f" {t_mlx * 1e3:>9.3f}"
        )

    note(
        "На N=2048 grid уже заполнен: split-K только добавляет reduction. "
        "На N=256 output-плиток меньше, и split=2 иногда чуть лучше, но "
        "выигрыш нестабилен: каждый split — отдельный launch плюс reduction."
    )
    note(
        "Это тот же компромисс, что в tiny-llm: split_k включают по форме "
        "после замера, а не всегда."
    )


def main():
    section("15. Quantized embedding и split-K")
    if _EMBED is None or _SPLITK is None:
        note("Metal недоступен: лаборатория пропущена.")
        return
    print(f"  устройство: {mx.device_info().get('device_name', 'unknown')}")
    embedding_check()
    splitk_check()
    splitk_benchmark(2048, 2048, "N=2048: grid заполнен")
    splitk_benchmark(256, 2048, "N=256: grid недозагружен")

    section("Вывод")
    note("Embedding распаковывает только выбранные строки таблицы.")
    note("Split-K делит reduction по K и складывает partial-плитки.")
    note("Split-K — компромисс occupancy vs reduction.")
    note("Вместе с model/11 это полный W4-набор: matvec, tiled matmul, split-K.")


if __name__ == "__main__":
    main()
