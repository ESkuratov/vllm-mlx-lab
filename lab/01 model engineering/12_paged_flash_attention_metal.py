# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 12: paged attention на Metal (online softmax).

Идея
----
В inference/19 мы реализовали page-walk на Python. Здесь тот же алгоритм
переносится в Metal-ядро:

* один threadgroup (32 lane) на один query-вектор;
* K/V читаются прямо по ``block_table``, без dense-реконструкции;
* score считается через ``simd_sum`` по head_dim;
* softmax обновляется online: max/sum/acc пересчитываются на каждом слоте.

Это FlashAttention-подход по смыслу (не materialize-им матрицу scores),
но без tiled MMA. Один query на threadgroup — учебный компромисс;
production-ядро (tiny-llm Week 3 Day 5) добавляет плитки и simdgroup_matrix.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/12_paged_flash_attention_metal.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import bench, mb, note, section, subsection

PAGE_SIZE = 16
_D = 128
_HQ = 32
_HKV = 8

_SOURCE = r"""
    const int lane = thread_index_in_threadgroup;
    const int qidx = threadgroup_position_in_grid.y;

    const int b = qidx / (HQ * L);
    const int rem = qidx - b * (HQ * L);
    const int h = rem / L;
    const int pos = rem - h * L;
    const int kvh = h / (HQ / HKV);

    const int context = context_lens[b];
    int visible = context;
    if (IS_CAUSAL) {
        visible = context - L + pos + 1;
        visible = clamp(visible, 0, context);
    }
    const int visible_pages = min(
        MAX_PAGES, (visible + PAGE_SIZE - 1) / PAGE_SIZE);

    constexpr int values_per_lane = (D + 31) / 32;
    float qv[values_per_lane];
    float acc[values_per_lane];
    #pragma clang loop unroll(full)
    for (int i = 0; i < values_per_lane; i++) {
        const int d = lane + i * 32;
        qv[i] = d < D ? static_cast<float>(q[qidx * D + d]) * scale : 0.0f;
        acc[i] = 0.0f;
    }

    float max_score = -1.0e30f;
    float sum = 0.0f;

    for (int logical_page = 0; logical_page < visible_pages; logical_page++) {
        const int page_id = block_table[b * MAX_PAGES + logical_page];
        if (page_id < 0) continue;
        const int page_start = logical_page * PAGE_SIZE;
        const int live = min(PAGE_SIZE, visible - page_start);
        if (live <= 0) break;

        for (int slot = 0; slot < live; slot++) {
            const int offset =
                ((page_id * HKV + kvh) * PAGE_SIZE + slot) * D;

            float partial = 0.0f;
            #pragma clang loop unroll(full)
            for (int i = 0; i < values_per_lane; i++) {
                const int d = lane + i * 32;
                if (d < D) {
                    partial += qv[i] *
                        static_cast<float>(key_pages[offset + d]);
                }
            }
            const float score = simd_sum(partial);

            const float new_max = max(max_score, score);
            const float alpha = metal::exp(max_score - new_max);
            const float beta = metal::exp(score - new_max);
            sum = sum * alpha + beta;
            #pragma clang loop unroll(full)
            for (int i = 0; i < values_per_lane; i++) {
                const int d = lane + i * 32;
                if (d < D) {
                    acc[i] = acc[i] * alpha +
                        beta * static_cast<float>(value_pages[offset + d]);
                }
            }
            max_score = new_max;
        }
    }

    #pragma clang loop unroll(full)
    for (int i = 0; i < values_per_lane; i++) {
        const int d = lane + i * 32;
        if (d < D) {
            out[qidx * D + d] = static_cast<T>(
                sum == 0.0f ? 0.0f : acc[i] / sum);
        }
    }
"""


def _make_kernel():
    if not mx.metal.is_available():
        return None
    return mx.fast.metal_kernel(
        name="paged_flash_attention",
        input_names=[
            "q",
            "key_pages",
            "value_pages",
            "block_table",
            "context_lens",
            "scale",
        ],
        output_names=["out"],
        source=_SOURCE,
    )


_KERNEL = _make_kernel()


# --------------------------------------------------------------------------- #
# Обёртка
# --------------------------------------------------------------------------- #


def paged_flash_attention(
    q: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    page_size: int,
    scale: float,
    causal: bool = True,
) -> mx.array:
    """q [B, H_q, L, D], pages [P, H_kv, page_size, D], table [B, max_pages]."""
    if _KERNEL is None:
        raise RuntimeError("Metal недоступен")
    if q.ndim != 4 or key_pages.ndim != 4:
        raise ValueError("ожидается q [B,H_q,L,D] и pages [P,H_kv,page,D]")
    b, hq, length, d = q.shape
    _, hkv, stored_page, stored_d = key_pages.shape
    if key_pages.shape != value_pages.shape:
        raise ValueError("key/value pages должны совпадать")
    if stored_page != page_size or stored_d != d:
        raise ValueError("page_size/D не совпадают")
    if block_table.shape[0] != b or context_lens.shape[0] != b:
        raise ValueError("batch size не совпадает")
    if hq % hkv != 0:
        raise ValueError("H_q должно делиться на H_kv")

    flat_q = mx.contiguous(q.reshape(b * hq * length, d))
    out = _KERNEL(
        inputs=[
            flat_q,
            mx.contiguous(key_pages),
            mx.contiguous(value_pages),
            mx.contiguous(block_table),
            mx.contiguous(context_lens),
            mx.array(scale, dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype),
            ("HQ", hq),
            ("HKV", hkv),
            ("L", length),
            ("D", d),
            ("PAGE_SIZE", page_size),
            ("MAX_PAGES", block_table.shape[1]),
            ("IS_CAUSAL", 1 if causal else 0),
        ],
        grid=(32, b * hq * length, 1),
        threadgroup=(32, 1, 1),
        output_shapes=[(b * hq * length, d)],
        output_dtypes=[q.dtype],
    )[0]
    return out.reshape(b, hq, length, d)


# --------------------------------------------------------------------------- #
# Dense-эталон
# --------------------------------------------------------------------------- #


def dense_reference(
    q: mx.array,
    key_pages: mx.array,
    value_pages: mx.array,
    block_table: mx.array,
    context_lens: mx.array,
    causal: bool = True,
) -> mx.array:
    b, hq, length, d = q.shape
    hkv = key_pages.shape[1]
    repeats = hq // hkv
    outs = []
    for i in range(b):
        ctx = int(context_lens[i])
        pages = [int(p) for p in block_table[i] if int(p) >= 0]
        k = mx.concatenate([key_pages[p] for p in pages], axis=1)[:, :ctx]
        v = mx.concatenate([value_pages[p] for p in pages], axis=1)[:, :ctx]
        qb = q[i]
        kb = mx.repeat(k, repeats, axis=0)
        vb = mx.repeat(v, repeats, axis=0)
        scores = mx.matmul(qb, kb.swapaxes(-2, -1)) * (d**-0.5)
        if causal:
            rows = mx.arange(length)[:, None]
            cols = mx.arange(ctx)[None, :]
            mask = mx.where(
                cols <= (ctx - length + rows), 0.0, -mx.inf
            ).astype(scores.dtype)
            scores = scores + mask[None]
        probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
        outs.append(mx.matmul(probs, vb))
    return mx.stack(outs, axis=0)


# --------------------------------------------------------------------------- #
# Проверки и замеры
# --------------------------------------------------------------------------- #


def _make_case(
    seed: int,
    context_len: int,
    query_len: int,
    page_size: int,
    hq: int,
    hkv: int,
    d: int,
    num_pages: int,
):
    mx.random.seed(seed)
    pages = mx.random.normal((num_pages, hkv, page_size, d)).astype(mx.bfloat16)
    values = mx.random.normal((num_pages, hkv, page_size, d)).astype(mx.bfloat16)
    # Логические страницы идут не по порядку физических id.
    physical = list(range(num_pages))[::-1]
    needed = (context_len + page_size - 1) // page_size
    table = physical[:needed] + [-1] * (needed + 1)
    block_table = mx.array([table], dtype=mx.int32)
    context_lens = mx.array([context_len], dtype=mx.int32)
    q = mx.random.normal((1, hq, query_len, d)).astype(mx.bfloat16)
    return q, pages, values, block_table, context_lens


def _assert_close(name: str, actual: mx.array, expected: mx.array) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: {actual.shape} != {expected.shape}")
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


def correctness():
    subsection("A. Kernel vs dense-эталон")
    cases = [
        # context, query_len, page_size, Hq, Hkv, D
        (13, 1, 8, 4, 2, 64),
        (17, 5, 8, 4, 2, 64),
        (35, 9, 16, 8, 2, 128),
    ]
    for idx, (ctx, length, page, hq, hkv, d) in enumerate(cases):
        num_pages = (ctx + page - 1) // page + 2
        q, k, v, table, lens = _make_case(
            idx + 1, ctx, length, page, hq, hkv, d, num_pages
        )
        out = paged_flash_attention(
            q, k, v, table, lens, page, d**-0.5, causal=True
        )
        ref = dense_reference(q, k, v, table, lens, causal=True)
        _assert_close(
            f"ctx={ctx}, L={length}, page={page}, H={hq}/{hkv}, D={d}",
            out,
            ref,
        )


def benchmark():
    subsection("B. Decode и prefill на длинном контексте")
    ctx = 1024
    num_pages = ctx // PAGE_SIZE
    mx.random.seed(7)
    k = mx.random.normal((num_pages, _HKV, PAGE_SIZE, _D)).astype(mx.bfloat16)
    v = mx.random.normal((num_pages, _HKV, PAGE_SIZE, _D)).astype(mx.bfloat16)
    table = mx.array([list(range(num_pages))], dtype=mx.int32)
    lens = mx.array([ctx], dtype=mx.int32)

    for length in (1, 16):
        q = mx.random.normal((1, _HQ, length, _D)).astype(mx.bfloat16)
        t_kernel = bench(
            lambda: mx.eval(
                paged_flash_attention(
                    q, k, v, table, lens, PAGE_SIZE, _D**-0.5, causal=True
                )
            ),
            repeats=3,
        )
        t_ref = bench(
            lambda: mx.eval(dense_reference(q, k, v, table, lens, causal=True)),
            repeats=2,
        )
        print(
            f"  ctx={ctx}, L={length:>2}: kernel {t_kernel * 1e3:7.2f} ms, "
            f"dense {t_ref * 1e3:7.2f} ms, "
            f"KV {mb(k.nbytes + v.nbytes)}"
        )

    note(
        "Kernel не строит dense K/V и не materialize-ит scores: он идёт по "
        "страницам и обновляет online-softmax. Для decode это особенно важно: "
        "история читается один раз, без копии O(контекст)."
    )
    note(
        "Один threadgroup на query — простая, но не оптимальная раскладка. "
        "Для длинного prefill production-ядро tiles queries и использует MMA "
        "(см. model/11)."
    )


def main():
    section("12. Paged FlashAttention на Metal")
    if _KERNEL is None:
        note("Metal недоступен: лаборатория пропущена.")
        return
    print(f"  устройство: {mx.device_info().get('device_name', 'unknown')}")
    correctness()
    benchmark()

    section("Вывод")
    note("Page-walk + online-softmax можно делать прямо в Metal.")
    note("K/V читаются по block_table; dense-реконструкции нет.")
    note("Score — simd_sum по D; V аккумулируется по lane.")
    note("Следующий шаг — tiled MMA и несколько query на threadgroup.")


if __name__ == "__main__":
    main()
