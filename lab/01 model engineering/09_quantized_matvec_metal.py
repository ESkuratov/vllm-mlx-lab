# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 09: W4 quantized matvec на Metal.

Идея
----
4-битные веса хранят по 8 значений в одном uint32. На каждый токен decode
модель читает все веса, поэтому распаковывать их в отдельный BF16-тензор
(как делает наивный `mx.dequantize`) — значит прочитать и записать лишние
гигабайты. Правильный путь: распаковать вес прямо в регистре и сразу
умножить на активацию.

Здесь мы пишем два Metal-ядра через `mx.fast.metal_kernel`:

* ``w4_matvec`` — одна SIMD-группа (32 lane) на выходной элемент,
  K делится между lane, в конце ``simd_sum``. Это decode-путь для M ≤ 8.
* ``w4_matmul_naive`` — один поток на выходной элемент. Просто, но каждый
  поток заново читает строку весов; видно, почему для больших M нужен
  tiled matmul с переиспользованием весов.

Оба результата сверяются с `mx.quantized_matmul` (эталон MLX) и с
декомпрессией через `mx.dequantize`.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/09_quantized_matvec_metal.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import mb, note, section, subsection

GROUP_SIZE = 128
BITS = 4


# --------------------------------------------------------------------------- #
# Metal-ядра
# --------------------------------------------------------------------------- #

_MATVEC_SOURCE = r"""
    constexpr int packs_per_item = 32 / BITS;
    const int lane = thread_position_in_grid.x;
    const int n = thread_position_in_grid.y;
    const int m = thread_position_in_grid.z;

    float acc = 0.0f;
    for (int k = lane; k < K; k += 32) {
        const uint32_t packed = w[n * (K / packs_per_item) + k / packs_per_item];
        const int shift = (k % packs_per_item) * BITS;
        const float q = static_cast<float>(
            (packed >> shift) & ((1u << BITS) - 1u));
        const int g = k / GROUP_SIZE;
        const float scale = static_cast<float>(
            scales[n * (K / GROUP_SIZE) + g]);
        const float bias = static_cast<float>(
            biases[n * (K / GROUP_SIZE) + g]);
        acc += (q * scale + bias) * static_cast<float>(x[m * K + k]);
    }
    acc = simd_sum(acc);
    if (lane == 0) {
        y[m * N + n] = static_cast<T>(acc);
    }
"""

_MATMUL_NAIVE_SOURCE = r"""
    constexpr int packs_per_item = 32 / BITS;
    const int n = thread_position_in_grid.x;
    const int m = thread_position_in_grid.y;

    float acc = 0.0f;
    for (int k = 0; k < K; k++) {
        const uint32_t packed = w[n * (K / packs_per_item) + k / packs_per_item];
        const int shift = (k % packs_per_item) * BITS;
        const float q = static_cast<float>(
            (packed >> shift) & ((1u << BITS) - 1u));
        const int g = k / GROUP_SIZE;
        const float scale = static_cast<float>(
            scales[n * (K / GROUP_SIZE) + g]);
        const float bias = static_cast<float>(
            biases[n * (K / GROUP_SIZE) + g]);
        acc += (q * scale + bias) * static_cast<float>(x[m * K + k]);
    }
    y[m * N + n] = static_cast<T>(acc);
"""


def _make_kernels():
    if not mx.metal.is_available():
        return None, None
    matvec = mx.fast.metal_kernel(
        name="w4_matvec_lab",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        source=_MATVEC_SOURCE,
    )
    naive = mx.fast.metal_kernel(
        name="w4_matmul_naive_lab",
        input_names=["x", "w", "scales", "biases"],
        output_names=["y"],
        source=_MATMUL_NAIVE_SOURCE,
    )
    return matvec, naive


_MATVEC, _MATMUL_NAIVE = _make_kernels()


def _check_shapes(x, wq, scales, biases, group_size, bits):
    if bits != 4:
        raise ValueError("лаборатория поддерживает только 4 бита")
    if group_size % 32 != 0:
        raise ValueError("group_size должен делиться на 32")
    m, k = x.shape
    n, packed_cols = wq.shape
    if packed_cols != k // (32 // bits):
        raise ValueError("упакованная ширина не совпадает с K")
    if scales.shape != (n, k // group_size):
        raise ValueError("форма scales не совпадает с (N, K/group_size)")
    if biases.shape != scales.shape:
        raise ValueError("форма biases не совпадает с scales")
    return m, n, k


def w4_matvec(x, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS):
    """Decode-путь: одна SIMD-группа на output-элемент."""
    m, n, k = _check_shapes(x, wq, scales, biases, group_size, bits)
    out = _MATVEC(
        inputs=[x, wq, scales, biases],
        template=[
            ("T", x.dtype),
            ("K", k),
            ("N", n),
            ("BITS", bits),
            ("GROUP_SIZE", group_size),
        ],
        grid=(32, n, m),
        threadgroup=(32, 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]
    return out


def w4_matmul_naive(x, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS):
    """Наивный matmul: один поток на (m, n), без переиспользования весов."""
    m, n, k = _check_shapes(x, wq, scales, biases, group_size, bits)
    out = _MATMUL_NAIVE(
        inputs=[x, wq, scales, biases],
        template=[
            ("T", x.dtype),
            ("K", k),
            ("N", n),
            ("BITS", bits),
            ("GROUP_SIZE", group_size),
        ],
        grid=(n, m, 1),
        threadgroup=(min(n, 256), 1, 1),
        output_shapes=[(m, n)],
        output_dtypes=[x.dtype],
    )[0]
    return out


# --------------------------------------------------------------------------- #
# Эталоны и замеры
# --------------------------------------------------------------------------- #


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


def dequantized_reference(x, wq, scales, biases, group_size=GROUP_SIZE, bits=BITS):
    w = mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits)
    return x @ w.T


def _assert_close(name: str, actual: mx.array, expected: mx.array) -> None:
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


def correctness(shape_name: str, n: int, k: int) -> tuple[mx.array, mx.array, mx.array, mx.array]:
    subsection(f"A. Корректность: {shape_name} (N={n}, K={k})")
    mx.random.seed(123)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(wq, scales, biases)

    x = (mx.random.normal((4, k)) * 0.5).astype(mx.bfloat16)
    ref = mlx_reference(x, wq, scales, biases)
    deq = dequantized_reference(x, wq, scales, biases)
    vec = w4_matvec(x, wq, scales, biases)
    naive = w4_matmul_naive(x, wq, scales, biases)

    _assert_close("w4_matvec vs mx.quantized_matmul", vec, ref)
    _assert_close("w4_matmul_naive vs mx.quantized_matmul", naive, ref)
    # 4-битная квантизация сама по себе вносит ошибку, поэтому здесь допуск шире.
    _assert_close("mx.quantized_matmul vs dequantize", ref, deq)

    packed_bytes = wq.nbytes + scales.nbytes + biases.nbytes
    dense_bytes = w.nbytes
    print(
        f"  вес: packed {mb(packed_bytes)} vs BF16 {mb(dense_bytes)} "
        f"({packed_bytes / dense_bytes:.1%} от BF16)"
    )
    print(
        f"  группа: {GROUP_SIZE}, бит: {BITS}, "
        f"на выход: {wq.shape[1]} uint32 на строку"
    )
    return wq, scales, biases, x


def benchmark(n: int = 2048, k: int = 2048) -> None:
    subsection("B. M-sweep: matvec vs наивный matmul vs MLX")
    mx.random.seed(5)
    w = (mx.random.normal((n, k)) * 0.02).astype(mx.bfloat16)
    wq, scales, biases = mx.quantize(w, group_size=GROUP_SIZE, bits=BITS)
    mx.eval(wq, scales, biases)

    print(
        f"  N={n}, K={k}; время в мс, медиана; "
        f"packed-вес {mb(wq.nbytes + scales.nbytes + biases.nbytes)}"
    )
    print(f"  {'M':>4} {'matvec':>11} {'naive':>11} {'MLX qmm':>11} {'MLX/matvec':>12}")
    for m in (1, 2, 4, 8, 32, 128):
        x = (mx.random.normal((m, k)) * 0.5).astype(mx.bfloat16)
        t_vec = _bench(lambda: w4_matvec(x, wq, scales, biases), repeats=2)
        t_naive = _bench(lambda: w4_matmul_naive(x, wq, scales, biases), repeats=2)
        t_mlx = _bench(lambda: mlx_reference(x, wq, scales, biases), repeats=3)
        print(
            f"  {m:>4} {t_vec * 1e3:>11.3f} {t_naive * 1e3:>11.3f} "
            f"{t_mlx * 1e3:>11.3f} {t_mlx / t_vec:>10.2f}x"
        )

    note(
        "Matvec — это decode-путь: одна SIMD-группа на выход, без dense-"
        "декомпрессии всей матрицы. В этой учебной реализации он не быстрее "
        "наивного ядра, потому что N велико и output-грид и так заполнен; "
        "зато он показывает правильную схему reduce по K."
    )
    note(
        "С ростом M оба наших ядра читают строку весов на каждую строку "
        "активаций. Настоящий prefill-путь — tiled matmul: загрузить плитку "
        "весов в threadgroup и переиспользовать её для многих M. Поэтому "
        "MLX qmm обгоняет нас всё сильнее."
    )
    note(
        "MLX qmm — оптимизированный эталон; наша цель — понять механику, "
        "а не обогнать библиотеку."
    )


def main() -> None:
    section("09. W4 quantized matvec на Metal")
    if _MATVEC is None or _MATMUL_NAIVE is None:
        note("Metal недоступен: лаборатория пропущена.")
        return

    print(f"  устройство: {mx.device_info().get('device_name', 'unknown')}")
    correctness("Qwen2.5-3B q_proj", 2048, 2048)
    correctness("Qwen2.5-3B gate/up", 11008, 2048)
    benchmark()

    section("Вывод")
    note("W4 хранит 8 значений в uint32; распаковка должна идти в регистре.")
    note("Matvec: одна SIMD-группа на выход, K делится по lane, simd_sum.")
    note("Matvec — decode-путь; для prefill нужен tiled matmul с reuse весов.")
    note("Сравнение с mx.quantized_matmul показывает цену учебной реализации.")


if __name__ == "__main__":
    main()
