# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 11: tiled W4 matmul с simdgroup_matrix.

Идея
----
В model/09 matvec читал строку весов на каждый выход. Для prefill (M велик)
это плохо: одна и та же строка весов нужна многим строкам активаций.
Нужен tile reuse:

* threadgroup считает плитку 8×8 выходных элементов;
* активации и деквантованные веса грузятся в threadgroup memory;
* ``simdgroup_matrix`` делает 8×8×8 MMA с FP32-аккумулятором;
* частичные плитки (M или N не кратны 8) добиваются нулями.

Это учебный шаг к kernel'у tiny-llm Week 2 Day 3: там плитка 32×32,
несколько SIMD-групп и более сложная раскладка. Здесь мы оставляем один
SIMD-group на плитку, чтобы код читался.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/11_tiled_w4_matmul.py"
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

_MMA_SOURCE = r"""
    const int lane = thread_index_in_threadgroup;
    const int m0 = threadgroup_position_in_grid.z * 8;
    const int n0 = threadgroup_position_in_grid.y * 8;

    threadgroup T a_tile[64];
    threadgroup T b_tile[64];
    threadgroup float c_tile[64];

    simdgroup_matrix<float, 8, 8> acc;
    acc.thread_elements()[0] = 0.0f;
    acc.thread_elements()[1] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += 8) {
        // A: 8 строк m × 8 столбцов k, row-major.
        for (int idx = lane; idx < 64; idx += 32) {
            const int mm = idx / 8;
            const int kk = idx - mm * 8;
            const int m = m0 + mm;
            const int k = k0 + kk;
            a_tile[idx] = (m < M && k < K) ? a[m * K + k] : T(0);
        }

        // B: 8 строк k × 8 столбцов n, row-major.
        for (int idx = lane; idx < 64; idx += 32) {
            const int kk = idx / 8;
            const int nn = idx - kk * 8;
            const int k = k0 + kk;
            const int n = n0 + nn;
            if (k < K && n < N) {
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
            y[m * N + n] = static_cast<T>(c_tile[idx]);
        }
    }
"""

_NAIVE_SOURCE = r"""
    const int n = thread_position_in_grid.x;
    const int m = thread_position_in_grid.y;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        const uint32_t packed = w[n * (K / 8) + k / 8];
        const int shift = (k % 8) * 4;
        const float q = static_cast<float>((packed >> shift) & 0xFu);
        const int g = k / GROUP_SIZE;
        const float scale = static_cast<float>(
            scales[n * (K / GROUP_SIZE) + g]);
        const float bias = static_cast<float>(
            biases[n * (K / GROUP_SIZE) + g]);
        acc += (q * scale + bias) * static_cast<float>(a[m * K + k]);
    }
    y[m * N + n] = static_cast<T>(acc);
"""


def _make_kernels():
    if not mx.metal.is_available():
        return None, None
    mma = mx.fast.metal_kernel(
        name="w4_tiled_mma",
        input_names=["a", "w", "scales", "biases"],
        output_names=["y"],
        source=_MMA_SOURCE,
        header="#include <metal_simdgroup_matrix>\n",
    )
    naive = mx.fast.metal_kernel(
        name="w4_naive_matmul",
        input_names=["a", "w", "scales", "biases"],
        output_names=["y"],
        source=_NAIVE_SOURCE,
    )
    return mma, naive


_MMA, _NAIVE = _make_kernels()


# --------------------------------------------------------------------------- #
# Операторы
# --------------------------------------------------------------------------- #


def _check(x, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS):
    if bits != 4:
        raise ValueError("лаборатория поддерживает только 4 бита")
    m, k = x.shape
    n, packed_cols = wq.shape
    if packed_cols != k // 8:
        raise ValueError("упакованная ширина не совпадает с K")
    if scales.shape != (n, k // group_size):
        raise ValueError("форма scales не совпадает")
    if biases.shape != scales.shape:
        raise ValueError("форма biases не совпадает")
    return m, n, k


def w4_tiled(x, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS):
    m, n, k = _check(x, wq, scales, biases, group_size, bits)
    return _MMA(
        inputs=[x, wq, scales, biases],
        template=[
            ("T", x.dtype),
            ("M", m),
            ("N", n),
            ("K", k),
            ("GROUP_SIZE", group_size),
        ],
        grid=(32, (n + 7) // 8, (m + 7) // 8),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


def w4_naive(x, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS):
    m, n, k = _check(x, wq, scales, biases, group_size, bits)
    return _NAIVE(
        inputs=[x, wq, scales, biases],
        template=[
            ("T", x.dtype),
            ("M", m),
            ("N", n),
            ("K", k),
            ("GROUP_SIZE", group_size),
        ],
        grid=(n, m, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]


def mlx_reference(x, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS):
    return mx.quantized_matmul(
        x,
        wq,
        scales=scales,
        biases=biases,
        transpose=True,
        group_size=group_size,
        bits=bits,
    )


# --------------------------------------------------------------------------- #
# Проверки и замеры
# --------------------------------------------------------------------------- #


def _assert_close(name, actual, expected):
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: {actual.shape} != {expected.shape}")
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


def correctness():
    subsection("A. Корректность, включая частичные плитки")
    mx.random.seed(123)
    for m, n, k in ((4, 2048, 2048), (13, 2050, 2048), (8, 2048, 1920)):
        w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
        wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
        x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
        ref = mlx_reference(x, wq, scales, biases)
        _assert_close(
            f"tiled M={m}, N={n}, K={k}",
            w4_tiled(x, wq, scales, biases),
            ref,
        )
        _assert_close(
            f"naive M={m}, N={n}, K={k}",
            w4_naive(x, wq, scales, biases),
            ref,
        )


def benchmark(n=2048, k=2048):
    subsection("B. M-sweep: naive vs tiled MMA vs MLX")
    mx.random.seed(5)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(wq, scales, biases)

    print(
        f"  N={n}, K={k}; время в мс, медиана; "
        f"packed-вес {wq.nbytes / 1024**2:.1f} MB"
    )
    print(f"  {'M':>4} {'naive':>11} {'tiled':>11} {'MLX qmm':>11} {'tiled/naive':>12}")
    for m in (1, 4, 8, 32, 128):
        x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
        t_naive = _bench(lambda: w4_naive(x, wq, scales, biases), repeats=2)
        t_tiled = _bench(lambda: w4_tiled(x, wq, scales, biases), repeats=3)
        t_mlx = _bench(lambda: mlx_reference(x, wq, scales, biases), repeats=3)
        print(
            f"  {m:>4} {t_naive * 1e3:>11.3f} {t_tiled * 1e3:>11.3f} "
            f"{t_mlx * 1e3:>11.3f} {t_naive / t_tiled:>11.2f}x"
        )

    note(
        "Tiled MMA переиспользует веса: каждая 8×8 плитка B читается один раз "
        "на 8 строк A. Наивный matmul читает строку весов на каждый выход."
    )
    note(
        "Один SIMD-group на плитку — учебный компромисс. Production-ядро "
        "берёт плитку 32×32, несколько SIMD-групп и double-buffering."
    )
    note(
        "MLX qmm всё ещё быстрее: у него тюнинг, векторизация и лучшая "
        "раскладка threadgroup memory."
    )


def main():
    section("11. Tiled W4 matmul с simdgroup_matrix")
    if _MMA is None or _NAIVE is None:
        note("Metal недоступен: лаборатория пропущена.")
        return

    print(f"  устройство: {mx.device_info().get('device_name', 'unknown')}")
    correctness()
    benchmark()

    section("Вывод")
    note("Tile reuse — главный шаг от matvec к prefill matmul.")
    note("simdgroup_matrix делает 8×8×8 MMA, аккумулятор — FP32.")
    note("Частичные M/N/K закрываются нулями и guard'ами при записи.")
    note("Дальше — плитка 32×32, несколько SIMD-групп и split-K.")


if __name__ == "__main__":
    main()
