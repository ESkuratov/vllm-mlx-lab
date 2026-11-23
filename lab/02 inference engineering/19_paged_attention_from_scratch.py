# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 19: paged attention с нуля.

Идея
----
Paged attention — это виртуальная память для KV-кэша. Вместо одного
непрерывного тензора на запрос мы храним KV в страницах фиксированного
размера и держим таблицу «логическая страница → физическая страница».
Тогда:

* память не фрагментируется по длинам запросов;
* страницы можно переиспользовать и разделять;
* attention умеет читать K/V прямо по таблице, не собирая dense-историю.

В этом эксперименте мы пишем три вещи руками:

1. ``PagePool`` — физические страницы и аллокатор;
2. ``PagedRequest`` — логическая последовательность и ``block_table``;
3. два варианта attention:
   * ``paged_attention_gather`` — сначала собрать dense K/V, потом SDPA
     (просто, но копирует всю историю);
   * ``paged_attention_direct`` — идти по страницам и обновлять
     online-softmax (то, что делает настоящее paged-ядро).

Оба варианта сверяются с dense-эталоном. Затем мы меряем цену gather
и direct на длинном контексте.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/19_paged_attention_from_scratch.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import bench, mb, note, section, subsection


# --------------------------------------------------------------------------- #
# Физические страницы
# --------------------------------------------------------------------------- #


class PagePool:
    """Фиксированный пул страниц: [num_pages, H_kv, page_size, D]."""

    def __init__(
        self,
        num_pages: int,
        num_kv_heads: int,
        page_size: int,
        head_dim: int,
        dtype=mx.float16,
    ):
        if min(num_pages, num_kv_heads, page_size, head_dim) <= 0:
            raise ValueError("все размеры пула должны быть положительными")
        self.num_pages = num_pages
        self.num_kv_heads = num_kv_heads
        self.page_size = page_size
        self.head_dim = head_dim
        self.dtype = dtype
        self.keys = mx.zeros(
            (num_pages, num_kv_heads, page_size, head_dim), dtype=dtype
        )
        self.values = mx.zeros(
            (num_pages, num_kv_heads, page_size, head_dim), dtype=dtype
        )
        # Свободные физические id. Стек, чтобы переиспользование было видно.
        self.free = list(range(num_pages - 1, -1, -1))
        self.used: set[int] = set()
        self.allocations = 0
        self.reuses = 0
        self.writes = 0

    def allocate(self) -> int:
        if not self.free:
            raise RuntimeError("пул страниц исчерпан")
        page_id = self.free.pop()
        self.used.add(page_id)
        self.allocations += 1
        if page_id != self.allocations - 1:
            self.reuses += 1
        return page_id

    def free_page(self, page_id: int) -> None:
        if page_id not in self.used:
            raise ValueError(f"страница {page_id} не занята")
        self.used.remove(page_id)
        self.free.append(page_id)

    def write(self, page_id: int, start: int, key: mx.array, value: mx.array) -> None:
        """Записать [H, S, D] в страницу, начиная со слота start."""
        if page_id not in self.used:
            raise ValueError(f"запись в свободную страницу {page_id}")
        s = key.shape[1]
        if key.shape != value.shape or key.shape[0] != self.num_kv_heads:
            raise ValueError("ожидается key/value [H_kv, S, D]")
        if key.shape[2] != self.head_dim:
            raise ValueError("несовпадение head_dim")
        if start < 0 or start + s > self.page_size:
            raise ValueError("запись выходит за границу страницы")
        self.keys[page_id, :, start : start + s, :] = key
        self.values[page_id, :, start : start + s, :] = value
        self.writes += 1

    def read(self, page_id: int) -> tuple[mx.array, mx.array]:
        if page_id not in self.used:
            raise ValueError(f"чтение свободной страницы {page_id}")
        return self.keys[page_id], self.values[page_id]

    @property
    def nbytes(self) -> int:
        return self.keys.nbytes + self.values.nbytes

    @property
    def used_pages(self) -> int:
        return len(self.used)


class PagedRequest:
    """Логическая последовательность, разложенная по страницам пула."""

    def __init__(self, pool: PagePool):
        self.pool = pool
        self.page_ids: list[int] = []
        self.length = 0
        self.dense_key_chunks: list[mx.array] = []
        self.dense_value_chunks: list[mx.array] = []

    def append(self, key: mx.array, value: mx.array) -> None:
        """Дописать K/V [H_kv, S, D], при необходимости открывая новые страницы."""
        if key.shape != value.shape:
            raise ValueError("key и value должны совпадать по форме")
        remaining = key.shape[1]
        offset = 0
        while remaining > 0:
            slot = self.length % self.pool.page_size
            if slot == 0:
                self.page_ids.append(self.pool.allocate())
            take = min(remaining, self.pool.page_size - slot)
            self.pool.write(
                self.page_ids[-1],
                slot,
                key[:, offset : offset + take, :],
                value[:, offset : offset + take, :],
            )
            self.length += take
            offset += take
            remaining -= take
        # Копия для dense-эталона: она не участвует в paged-пути.
        self.dense_key_chunks.append(key)
        self.dense_value_chunks.append(value)

    def block_table(self, max_pages: int) -> mx.array:
        if len(self.page_ids) > max_pages:
            raise ValueError("block_table меньше числа страниц запроса")
        row = self.page_ids + [-1] * (max_pages - len(self.page_ids))
        return mx.array(row, dtype=mx.int32)

    def dense_kv(self) -> tuple[mx.array, mx.array]:
        return (
            mx.concatenate(self.dense_key_chunks, axis=1),
            mx.concatenate(self.dense_value_chunks, axis=1),
        )

    @property
    def internal_fragmentation(self) -> int:
        return len(self.page_ids) * self.pool.page_size - self.length


# --------------------------------------------------------------------------- #
# Attention: dense-эталон и два paged-варианта
# --------------------------------------------------------------------------- #


def causal_mask(query_len: int, context_len: int, dtype=mx.float16) -> mx.array:
    """query i видит key j <= context_len - query_len + i."""
    rows = mx.arange(query_len)[:, None]
    cols = mx.arange(context_len)[None, :]
    return mx.where(cols <= (context_len - query_len + rows), 0.0, -mx.inf).astype(
        dtype
    )


def _expand_kv(kv: mx.array, num_heads: int) -> mx.array:
    num_kv_heads = kv.shape[0]
    if num_heads % num_kv_heads != 0:
        raise ValueError("H_q должно делиться на H_kv")
    return mx.repeat(kv, num_heads // num_kv_heads, axis=0)


def dense_attention(
    query: mx.array,
    key: mx.array,
    value: mx.array,
    scale: float | None = None,
    causal: bool = True,
) -> mx.array:
    """Обычный [H_q, L, D] attention по dense K/V [H_kv, S, D]."""
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("ожидается query [H_q, L, D] и key/value [H_kv, S, D]")
    num_heads, query_len, head_dim = query.shape
    ctx_len = key.shape[1]
    factor = head_dim**-0.5 if scale is None else scale
    key = _expand_kv(key, num_heads)
    value = _expand_kv(value, num_heads)
    scores = mx.matmul(query, key.swapaxes(-2, -1)) * factor
    if causal:
        scores = scores + causal_mask(query_len, ctx_len, scores.dtype)[None]
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(query.dtype)
    return mx.matmul(probs, value)


def paged_attention_gather(
    query: mx.array,
    pool: PagePool,
    request: PagedRequest,
    scale: float | None = None,
) -> mx.array:
    """Собрать dense K/V по block_table и вызвать обычный attention."""
    pages = [pool.read(page_id) for page_id in request.page_ids]
    key = mx.concatenate([page[0] for page in pages], axis=1)[:, : request.length]
    value = mx.concatenate([page[1] for page in pages], axis=1)[:, : request.length]
    return dense_attention(query, key, value, scale=scale)


def paged_attention_direct(
    query: mx.array,
    pool: PagePool,
    request: PagedRequest,
    scale: float | None = None,
) -> mx.array:
    """Идти по логическим страницам и обновлять online-softmax."""
    num_heads, query_len, head_dim = query.shape
    num_kv_heads = pool.num_kv_heads
    repeats = num_heads // num_kv_heads
    factor = head_dim**-0.5 if scale is None else scale

    # [H_kv, repeats, L, D] — так проще делить головы между KV-головами.
    q = query.reshape(num_kv_heads, repeats, query_len, head_dim)
    acc = mx.zeros((num_kv_heads, repeats, query_len, head_dim), dtype=mx.float32)
    running_max = mx.full((num_kv_heads, repeats, query_len, 1), -mx.inf)
    running_sum = mx.zeros((num_kv_heads, repeats, query_len, 1))
    ctx_len = request.length

    for logical_page, page_id in enumerate(request.page_ids):
        start = logical_page * pool.page_size
        if start >= ctx_len:
            break
        valid = min(pool.page_size, ctx_len - start)
        k_page, v_page = pool.read(page_id)
        k_page = k_page[:, :valid, :]
        v_page = v_page[:, :valid, :]

        # scores: [H_kv, repeats, L, valid]
        scores = mx.matmul(
            q, k_page[:, None, :, :].swapaxes(-2, -1)
        ) * factor

        # Маска считается по глобальным позициям контекста.
        query_pos = mx.arange(query_len)[:, None]
        key_pos = mx.arange(valid)[None, :] + start
        visible = key_pos <= (ctx_len - query_len + query_pos)
        page_mask = mx.where(visible, 0.0, -mx.inf).astype(scores.dtype)
        scores = scores + page_mask[None, None, :, :]

        page_max = mx.max(scores, axis=-1, keepdims=True)
        new_max = mx.maximum(running_max, page_max)
        alpha = mx.exp(running_max - new_max)
        probs = mx.exp(scores - new_max)
        running_sum = running_sum * alpha + mx.sum(probs, axis=-1, keepdims=True)
        acc = acc * alpha + mx.matmul(probs, v_page[:, None, :, :])
        running_max = new_max

    out = mx.where(running_sum == 0, 0.0, acc / running_sum)
    return out.astype(query.dtype).reshape(num_heads, query_len, head_dim)


# --------------------------------------------------------------------------- #
# Проверки и замеры
# --------------------------------------------------------------------------- #


def _assert_close(name: str, actual: mx.array, expected: mx.array) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name}: форма {actual.shape} != {expected.shape}")
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


def correctness_checks() -> None:
    subsection("A. Paged attention совпадает с dense-эталоном")

    cases = [
        # (context_len, query_len, page_size, H_q, H_kv, D, seed)
        (13, 1, 8, 4, 2, 16, 11),   # decode, хвост страницы
        (17, 5, 8, 4, 2, 16, 22),   # prefill, несколько страниц
        (35, 9, 16, 8, 2, 32, 33),  # GQA 4:1 и неполная последняя страница
    ]
    for ctx_len, q_len, page_size, hq, hkv, d, seed in cases:
        mx.random.seed(seed)
        pool = PagePool(
            num_pages=(ctx_len + page_size - 1) // page_size + 2,
            num_kv_heads=hkv,
            page_size=page_size,
            head_dim=d,
        )
        request = PagedRequest(pool)
        key = mx.random.normal((hkv, ctx_len, d)).astype(mx.float16)
        value = mx.random.normal((hkv, ctx_len, d)).astype(mx.float16)
        # Пишем кусками, чтобы потренировать неполные страницы.
        for start in range(0, ctx_len, 7):
            request.append(
                key[:, start : start + 7, :],
                value[:, start : start + 7, :],
            )
        query = mx.random.normal((hq, q_len, d)).astype(mx.float16)

        dense = dense_attention(query, key, value)
        gather = paged_attention_gather(query, pool, request)
        direct = paged_attention_direct(query, pool, request)

        _assert_close(
            f"ctx={ctx_len}, L={q_len}, page={page_size}, H={hq}/{hkv}, D={d}",
            gather,
            dense,
        )
        _assert_close("direct vs dense", direct, dense)
        assert len(set(request.page_ids)) == len(request.page_ids), "алиасинг страниц"
        note(
            f"страниц: {len(request.page_ids)}, "
            f"внутренняя фрагментация: {request.internal_fragmentation} токенов"
        )


def pool_reuse_check() -> None:
    subsection("B. Страницы освобождаются и переиспользуются")
    pool = PagePool(num_pages=4, num_kv_heads=1, page_size=4, head_dim=8)
    first = PagedRequest(pool)
    first.append(
        mx.ones((1, 10, 8), dtype=mx.float16),
        mx.ones((1, 10, 8), dtype=mx.float16),
    )
    print(f"  первый запрос занял страницы: {first.page_ids}")
    for page_id in first.page_ids:
        pool.free_page(page_id)
    second = PagedRequest(pool)
    second.append(
        mx.full((1, 6, 8), 2.0, dtype=mx.float16),
        mx.full((1, 6, 8), 3.0, dtype=mx.float16),
    )
    print(f"  второй запрос получил:       {second.page_ids}")
    print(f"  аллокаций: {pool.allocations}, переиспользований: {pool.reuses}")
    note("Свободные страницы идут в стек — новые запросы забирают их первыми.")


def benchmark() -> None:
    subsection("C. Gather vs direct на длинном контексте")
    ctx_len = 1024
    page_size = 16
    hq, hkv, d = 32, 8, 128
    pool = PagePool(
        num_pages=(ctx_len + page_size - 1) // page_size,
        num_kv_heads=hkv,
        page_size=page_size,
        head_dim=d,
    )
    request = PagedRequest(pool)
    mx.random.seed(7)
    key = mx.random.normal((hkv, ctx_len, d)).astype(mx.float16)
    value = mx.random.normal((hkv, ctx_len, d)).astype(mx.float16)
    for start in range(0, ctx_len, 128):
        request.append(
            key[:, start : start + 128, :],
            value[:, start : start + 128, :],
        )

    query = mx.random.normal((hq, 1, d)).astype(mx.float16)
    t_gather = bench(lambda: paged_attention_gather(query, pool, request), repeats=3)
    t_direct = bench(lambda: paged_attention_direct(query, pool, request), repeats=3)

    dense_bytes = hkv * ctx_len * d * key.dtype.size * 2
    print(f"  контекст: {ctx_len} токенов, page_size={page_size}")
    print(f"  страниц: {len(request.page_ids)}, KV-память: {mb(pool.nbytes)}")
    print(f"  gather: {t_gather * 1e3:8.2f} ms, копирует {mb(dense_bytes)} в dense")
    print(f"  direct: {t_direct * 1e3:8.2f} ms, dense-копии нет")
    note(
        "Gather простой, но каждый шаг копирует всю историю K/V. Direct читает "
        "страницы по одной; на Python накладные расходы выше, зато объём "
        "копирования не растёт с контекстом."
    )
    note(
        "Именно direct-схему реализуют Metal-ядра paged attention: они читают "
        "block_table и сразу считают online-softmax без dense-буфера."
    )


def main() -> None:
    section("19. Paged attention с нуля")
    print("  модель не нужна: работаем с синтетическими K/V и page pool.")
    correctness_checks()
    pool_reuse_check()
    benchmark()

    section("Вывод")
    note("Paged KV = страницы фиксированного размера + block_table.")
    note("Gather: просто и корректно, но O(контекст) копирования на шаг.")
    note("Direct: page-walk + online-softmax, без dense-реконструкции.")
    note("Дальше: вынести direct-цикл в Metal-ядро (см. lab 32/33).")


if __name__ == "__main__":
    main()
