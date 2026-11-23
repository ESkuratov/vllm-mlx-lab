# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 10: fused RMSNorm, RoPE и SwiGLU на Metal.

Идея
----
В Week 2 tiny-llm три маленьких оператора перестают быть цепочкой
MLX-операций и становятся отдельными Metal-ядрами:

* RMSNorm — один проход: сумма квадратов, rsqrt, умножение на weight;
* RoPE — поворот пар (x[i], x[i + D/2]) по фазе позиции;
* SwiGLU — `silu(gate) * up` без промежуточного тензора.

Ядра живут в `lab/metal_ops.py` и компилируются через
`mx.fast.metal_kernel`: CMake/nanobind не нужны. Их же использует
model/13 для сборки реальной модели Qwen.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/10_fused_ops_metal.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn

from common import note, section, subsection
from metal_ops import (
    HAS_METAL,
    ROPE_BASE,
    metal_rms_norm,
    metal_rope,
    metal_swiglu,
    rms_norm_ref,
    rope_ref,
    swiglu_ref,
)


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


def _bench(fn, repeats: int = 5) -> float:
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


def correctness() -> None:
    subsection("A. Корректность: ядро vs MLX-эталон")
    mx.random.seed(17)

    rows, d = 64, 2048
    x = mx.random.normal((rows, d)).astype(mx.float16)
    w = mx.random.normal((d,)).astype(mx.float16)
    _assert_close("RMSNorm 64×2048", metal_rms_norm(x, w), rms_norm_ref(x, w))

    x = mx.random.normal((2, 16, 8, 128)).astype(mx.float16)
    offsets = mx.array([0, 37], dtype=mx.int32)
    _assert_close(
        "RoPE B=2, S=16, H=8, D=128",
        metal_rope(x, offsets),
        rope_ref(x, offsets),
    )

    x1 = mx.random.normal((1, 8, 4, 128)).astype(mx.float16)
    off1 = mx.array([0], dtype=mx.int32)
    # mx.fast.rope вращает по оси T (вторая с конца), поэтому переставляем
    # [B, S, H, D] -> [B, H, S, D] и обратно.
    production = mx.fast.rope(
        x1.transpose(0, 2, 1, 3),
        dims=128,
        traditional=False,
        base=ROPE_BASE,
        scale=1.0,
        offset=0,
    ).transpose(0, 2, 1, 3)
    _assert_close("RoPE kernel vs mx.fast.rope", metal_rope(x1, off1), production)

    gate = mx.random.normal((128, 11008)).astype(mx.float16)
    up = mx.random.normal((128, 11008)).astype(mx.float16)
    _assert_close("SwiGLU 128×11008", metal_swiglu(gate, up), swiglu_ref(gate, up))


def benchmark() -> None:
    subsection("B. Сколько стоит fused-путь")
    mx.random.seed(23)

    rows, d = 128, 2048
    x = mx.random.normal((rows, d)).astype(mx.float16)
    w = mx.random.normal((d,)).astype(mx.float16)
    t_kernel = _bench(lambda: metal_rms_norm(x, w))
    t_ref = _bench(lambda: rms_norm_ref(x, w))
    compiled_rms = mx.compile(rms_norm_ref)
    t_compile = _bench(lambda: compiled_rms(x, w))
    print(
        f"  RMSNorm {rows}×{d}: kernel {t_kernel * 1e3:.3f} ms, "
        f"MLX {t_ref * 1e3:.3f} ms, compile {t_compile * 1e3:.3f} ms"
    )

    for s in (1, 128):
        x = mx.random.normal((1, s, 16, 128)).astype(mx.float16)
        off = mx.array([512], dtype=mx.int32)
        t_kernel = _bench(lambda: metal_rope(x, off))
        t_ref = _bench(lambda: rope_ref(x, off))
        t_prod = _bench(
            lambda: mx.fast.rope(
                x.transpose(0, 2, 1, 3),
                dims=128,
                traditional=False,
                base=ROPE_BASE,
                scale=1.0,
                offset=512,
            ).transpose(0, 2, 1, 3)
        )
        print(
            f"  RoPE S={s:>3}: kernel {t_kernel * 1e3:.3f} ms, "
            f"MLX {t_ref * 1e3:.3f} ms, mx.fast.rope {t_prod * 1e3:.3f} ms"
        )

    gate = mx.random.normal((128, 11008)).astype(mx.float16)
    up = mx.random.normal((128, 11008)).astype(mx.float16)
    t_kernel = _bench(lambda: metal_swiglu(gate, up))
    t_ref = _bench(lambda: swiglu_ref(gate, up))
    compiled_swiglu = mx.compile(lambda a, b: nn.silu(a) * b)
    t_compile = _bench(lambda: compiled_swiglu(gate, up))
    print(
        f"  SwiGLU 128×11008: kernel {t_kernel * 1e3:.3f} ms, "
        f"MLX {t_ref * 1e3:.3f} ms, compile {t_compile * 1e3:.3f} ms"
    )

    note(
        "Fused-ядро убирает промежуточные тензоры и лишние dispatch. "
        "На маленьких формах выигрыш может быть незаметен: важен полный "
        "forward, а не изолированный микро-бенчмарк."
    )
    note(
        "mx.fast.rope — оптимизированный production-путь MLX; наше ядро "
        "показывает семантику, а не пытается его обогнать."
    )
    note(
        "Наши ядра учебные: без векторизованных load/store и тюнинга "
        "threadgroup. На изолированном операторе MLX часто быстрее — "
        "смотрите на корректность и на полный forward, а не на микро-бенч."
    )


def main() -> None:
    section("10. Fused RMSNorm, RoPE и SwiGLU на Metal")
    if not HAS_METAL:
        note("Metal недоступен: лаборатория пропущена.")
        return

    print(f"  устройство: {mx.device_info().get('device_name', 'unknown')}")
    correctness()
    benchmark()

    section("Вывод")
    note("RMSNorm: sum(x²) по SIMD-группе, rsqrt, weight — один проход.")
    note("RoPE: фаза = (offset + позиция) × inv_freq; пары (i, i + D/2).")
    note("SwiGLU: silu(gate) * up без промежуточного тензора.")
    note("Такие ядра — строительные блоки Week 2 в tiny-llm.")


if __name__ == "__main__":
    main()
