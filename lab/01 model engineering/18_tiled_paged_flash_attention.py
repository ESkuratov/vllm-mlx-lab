# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 18: tiled paged FlashAttention.

Идея
----
Это объединение model/12 (page-walk) и model/16 (tiled dense attention):

* один threadgroup считает 8 query-строк;
* QK^T идёт через `simdgroup_matrix`;
* K/V читаются по `block_table` из страниц, без dense-реконструкции;
* causal mask и online-softmax обновляются по key-плиткам;
* P·V пока скалярный, как в `model/16`.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/18_tiled_paged_flash_attention.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import note, section, subsection

PAGE_SIZE = 16
_SOURCE = r"""
    const int lane = thread_index_in_threadgroup;
    const int tile_idx = threadgroup_position_in_grid.y;
    const int tiles_per_head = (L + 7) / 8;
    const int b = tile_idx / (HQ * tiles_per_head);
    const int rem = tile_idx - b * (HQ * tiles_per_head);
    const int h = rem / tiles_per_head;
    const int qm0 = (rem - h * tiles_per_head) * 8;
    const int kvh = h / (HQ / HKV);

    const int context = context_lens[b];
    threadgroup T q_tile[8 * D];
    threadgroup T kt_tile[D * 8];
    threadgroup T v_tile[8 * D];
    threadgroup float s_tile[64];
    threadgroup float acc[8 * D];
    threadgroup float pv_tile[8 * D];
    threadgroup float row_max[8];
    threadgroup float row_sum[8];
    threadgroup float alpha_row[8];
    threadgroup float beta_row[8];

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

    const int visible_pages = min(
        MAX_PAGES, (context + PAGE_SIZE - 1) / PAGE_SIZE);

    for (int logical_page = 0; logical_page < visible_pages; logical_page++) {
        const int page_id = block_table[b * MAX_PAGES + logical_page];
        if (page_id < 0) continue;
        const int page_start = logical_page * PAGE_SIZE;

        for (int slot0 = 0; slot0 < PAGE_SIZE; slot0 += 8) {
            if (page_start + slot0 >= context) break;

            // K^T tile [D, 8] из страницы.
            for (int idx = lane; idx < D * 8; idx += 32) {
                const int d = idx / 8;
                const int kk = idx - d * 8;
                const int slot = slot0 + kk;
                const int kg = page_start + slot;
                if (slot < PAGE_SIZE && kg < context) {
                    const int offset =
                        ((page_id * HKV + kvh) * PAGE_SIZE + slot) * D;
                    kt_tile[idx] = key_pages[offset + d];
                } else {
                    kt_tile[idx] = T(0);
                }
            }
            // V tile [8, D].
            for (int idx = lane; idx < 8 * D; idx += 32) {
                const int kk = idx / D;
                const int d = idx - kk * D;
                const int slot = slot0 + kk;
                const int kg = page_start + slot;
                if (slot < PAGE_SIZE && kg < context) {
                    const int offset =
                        ((page_id * HKV + kvh) * PAGE_SIZE + slot) * D;
                    v_tile[idx] = value_pages[offset + d];
                } else {
                    v_tile[idx] = T(0);
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

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

            if (lane < 8) {
                const int r = lane;
                const int qpos = qm0 + r;
                int visible = context;
                if (CAUSAL) {
                    visible = context - L + qpos + 1;
                    visible = clamp(visible, 0, context);
                }
                if (qpos >= L) visible = 0;

                float tile_max = -1.0e30f;
                for (int c = 0; c < 8; c++) {
                    const int kg = page_start + slot0 + c;
                    if (kg < visible) {
                        tile_max = max(
                            tile_max, s_tile[r * 8 + c] * scale);
                    }
                }
                float tile_sum = 0.0f;
                for (int c = 0; c < 8; c++) {
                    const int kg = page_start + slot0 + c;
                    if (kg < visible) {
                        const float p = metal::exp(
                            s_tile[r * 8 + c] * scale - tile_max);
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

            // P·V через MMA: 8×8 output tiles.
            for (int d0 = 0; d0 < D; d0 += 8) {
                simdgroup_matrix<float, 8, 8> pv_acc;
                pv_acc.thread_elements()[0] = 0.0f;
                pv_acc.thread_elements()[1] = 0.0f;
                simdgroup_matrix<float, 8, 8> pm;
                simdgroup_matrix<T, 8, 8> vm;
                simdgroup_load(pm, s_tile, 8);
                simdgroup_load(vm, v_tile + d0, D);
                simdgroup_multiply_accumulate(pv_acc, pm, vm, pv_acc);
                simdgroup_store(pv_acc, pv_tile + d0, D);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int idx = lane; idx < 8 * D; idx += 32) {
                const int r = idx / D;
                acc[idx] = acc[idx] * alpha_row[r] + beta_row[r] * pv_tile[idx];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
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
        name="tiled_paged_flash_attention",
        input_names=[
            "q", "key_pages", "value_pages", "block_table",
            "context_lens", "scale",
        ],
        output_names=["out"],
        source=_SOURCE,
        header="#include <metal_simdgroup_matrix>\n",
    )


_KERNEL = _make_kernel()


def paged_tiled_attention(
    q, key_pages, value_pages, block_table, context_lens,
    page_size: int, scale: float, causal: bool = True,
):
    if _KERNEL is None:
        raise RuntimeError("Metal недоступен")
    if page_size % 8 != 0:
        raise ValueError("page_size должен быть кратен 8")
    b, hq, length, d = q.shape
    _, hkv, stored_page, stored_d = key_pages.shape
    if stored_page != page_size or stored_d != d:
        raise ValueError("page_size/D не совпадают")
    if hq % hkv != 0:
        raise ValueError("H_q должно делиться на H_kv")
    return _KERNEL(
        inputs=[
            mx.contiguous(q), mx.contiguous(key_pages),
            mx.contiguous(value_pages), mx.contiguous(block_table),
            mx.contiguous(context_lens), mx.array(scale, dtype=mx.float32),
        ],
        template=[
            ("T", q.dtype), ("HQ", hq), ("HKV", hkv), ("L", length),
            ("D", d), ("PAGE_SIZE", page_size),
            ("MAX_PAGES", block_table.shape[1]),
            ("CAUSAL", 1 if causal else 0),
        ],
        grid=(32, b * hq * ((length + 7) // 8), 1),
        threadgroup=(32, 1, 1),
        output_shapes=[q.shape],
        output_dtypes=[q.dtype],
    )[0]


def dense_reference(q, key_pages, value_pages, block_table, context_lens, causal=True):
    b, hq, length, d = q.shape
    hkv = key_pages.shape[1]
    repeats = hq // hkv
    outs = []
    for i in range(b):
        ctx = int(context_lens[i])
        pages = [int(p) for p in block_table[i] if int(p) >= 0]
        k = mx.concatenate([key_pages[p] for p in pages], axis=1)[:, :ctx]
        v = mx.concatenate([value_pages[p] for p in pages], axis=1)[:, :ctx]
        kb = mx.repeat(k, repeats, axis=0)
        vb = mx.repeat(v, repeats, axis=0)
        scores = mx.matmul(q[i], kb.swapaxes(-2, -1)) * (d**-0.5)
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


def _assert_close(name, actual, expected):
    ok = mx.allclose(
        actual.astype(mx.float32), expected.astype(mx.float32),
        rtol=3e-2, atol=3e-2,
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


def make_case(seed, ctx, length, page, hq, hkv, d):
    mx.random.seed(seed)
    num_pages = (ctx + page - 1) // page + 2
    k = mx.random.normal((num_pages, hkv, page, d)).astype(mx.bfloat16)
    v = mx.random.normal((num_pages, hkv, page, d)).astype(mx.bfloat16)
    physical = list(range(num_pages))[::-1]
    needed = (ctx + page - 1) // page
    table = physical[:needed] + [-1] * (needed + 1)
    q = mx.random.normal((1, hq, length, d)).astype(mx.bfloat16)
    return q, k, v, mx.array([table], dtype=mx.int32), mx.array([ctx], dtype=mx.int32)


def correctness():
    subsection("A. Kernel vs dense-эталон")
    cases = [
        (13, 1, 8, 4, 2, 64),
        (17, 5, 8, 4, 2, 64),
        (35, 9, 16, 8, 2, 128),
        (64, 16, 16, 8, 2, 128),
    ]
    for i, (ctx, length, page, hq, hkv, d) in enumerate(cases):
        q, k, v, table, lens = make_case(i + 1, ctx, length, page, hq, hkv, d)
        out = paged_tiled_attention(q, k, v, table, lens, page, d**-0.5)
        ref = dense_reference(q, k, v, table, lens)
        _assert_close(
            f"ctx={ctx}, L={length}, page={page}, H={hq}/{hkv}, D={d}",
            out, ref,
        )


def benchmark():
    subsection("B. Decode и prefill")
    ctx, page, hq, hkv, d = 1024, 16, 32, 8, 128
    mx.random.seed(7)
    num_pages = ctx // page
    k = mx.random.normal((num_pages, hkv, page, d)).astype(mx.bfloat16)
    v = mx.random.normal((num_pages, hkv, page, d)).astype(mx.bfloat16)
    table = mx.array([list(range(num_pages))], dtype=mx.int32)
    lens = mx.array([ctx], dtype=mx.int32)
    for length in (1, 16, 64):
        q = mx.random.normal((1, hq, length, d)).astype(mx.bfloat16)
        t_kernel = _bench(
            lambda: paged_tiled_attention(q, k, v, table, lens, page, d**-0.5)
        )
        t_ref = _bench(lambda: dense_reference(q, k, v, table, lens), repeats=2)
        print(
            f"  ctx={ctx}, L={length:>2}: kernel {t_kernel * 1e3:7.2f} ms, "
            f"dense {t_ref * 1e3:7.2f} ms"
        )
    note(
        "8 query-строк на threadgroup, QK и P·V через MMA, K/V из страниц. "
        "Плитка 8×8; 32×32 multi-group — следующий шаг."
    )


def main():
    section("18. Tiled paged FlashAttention")
    if _KERNEL is None:
        note("Metal недоступен: лаборатория пропущена.")
        return
    print(f"  устройство: {mx.device_info().get('device_name', 'unknown')}")
    correctness()
    benchmark()

    section("Вывод")
    note("Page-walk и tiled QK совмещены в одном ядре.")
    note("K/V читаются по block_table, dense-реконструкции нет.")
    note("Online-softmax переносит состояние между страницами и плитками.")
    note("P·V тоже идёт через simdgroup_matrix; 32×32 tile — следующий шаг.")


if __name__ == "__main__":
    main()
