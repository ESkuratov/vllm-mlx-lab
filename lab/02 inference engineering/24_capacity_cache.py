# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 24: свой dense KV-cache с capacity, rewind и release.

Идея
----
tiny-llm Week 2 Day 1 начинает с request-owned dense KV-кэша:

* ``update_and_fetch`` дописывает K/V по оси последовательности;
* ``capacity`` заранее фиксирует размер и убирает growth-копии;
* ``rewind`` откатывает хвост (нужен speculative decoding);
* ``release`` освобождает storage;
* ``materialize`` фиксирует ленивый граф MLX.

В inference/19 мы делали paged-кэш, в inference/23 использовали ``BatchKVCache``.
Здесь — минимальный dense-кэш, который можно читать как эталон для paged.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/24_capacity_cache.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx

from common import mb, note, section, subsection


class TinyKvFullCache:
    """Dense KV-cache для одного запроса: [1, H, capacity, D]."""

    def __init__(self, capacity: int | None = None):
        if capacity is not None and capacity <= 0:
            raise ValueError("capacity must be positive or None")
        self.capacity = capacity
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.offset = 0
        self.growth_copy_bytes = 0
        self.slice_write_bytes = 0
        self.allocations = 0
        self.rewinds = 0

    def _alloc(self, shape, dtype):
        self.allocations += 1
        return mx.zeros(shape, dtype=dtype)

    def update_and_fetch(
        self, key: mx.array, value: mx.array
    ) -> tuple[mx.array, mx.array, int]:
        if key.shape != value.shape or key.ndim != 4:
            raise ValueError("ожидается key/value [1, H, S, D]")
        if key.shape[0] != 1:
            raise ValueError("dense cache поддерживает один запрос")
        _, h, s, d = key.shape
        if s <= 0:
            raise ValueError("append length must be positive")
        if self.capacity is not None and self.offset + s > self.capacity:
            raise ValueError(
                f"capacity overflow: {self.offset}+{s} > {self.capacity}"
            )

        if self.keys is None:
            size = self.capacity if self.capacity is not None else s
            self.keys = self._alloc((1, h, size, d), key.dtype)
            self.values = self._alloc((1, h, size, d), value.dtype)
        elif self.offset + s > self.keys.shape[2]:
            old = self.keys
            new_size = max(self.offset + s, self.keys.shape[2] * 2)
            new_keys = self._alloc((1, h, new_size, d), key.dtype)
            new_values = self._alloc((1, h, new_size, d), value.dtype)
            new_keys[:, :, : self.offset, :] = self.keys[:, :, : self.offset, :]
            new_values[:, :, : self.offset, :] = self.values[
                :, :, : self.offset, :
            ]
            self.growth_copy_bytes += old.nbytes + self.values.nbytes
            self.keys, self.values = new_keys, new_values

        self.keys[:, :, self.offset : self.offset + s, :] = key
        self.values[:, :, self.offset : self.offset + s, :] = value
        self.slice_write_bytes += key.nbytes + value.nbytes
        self.offset += s
        return self.keys[:, :, : self.offset, :], self.values[
            :, :, : self.offset, :
        ], self.offset

    def rewind(self, n: int) -> int:
        n = min(max(0, n), self.offset)
        self.offset -= n
        self.rewinds += 1
        return n

    def release(self) -> None:
        self.keys = None
        self.values = None
        self.offset = 0

    def materialize(self) -> None:
        if self.keys is not None:
            mx.eval(self.keys, self.values)

    @property
    def allocated_bytes(self) -> int:
        if self.keys is None:
            return 0
        return self.keys.nbytes + self.values.nbytes

    @property
    def logical_bytes(self) -> int:
        if self.keys is None:
            return 0
        return 2 * self.keys[:, :, : self.offset, :].nbytes

    @property
    def allocated_tokens(self) -> int:
        return 0 if self.keys is None else self.keys.shape[2]


def _append_chunks(cache: TinyKvFullCache, chunks: list[int]):
    h, d = 2, 8
    for i, s in enumerate(chunks):
        k = mx.random.normal((1, h, s, d)).astype(mx.float16)
        v = mx.random.normal((1, h, s, d)).astype(mx.float16)
        cache.update_and_fetch(k, v)


def semantics():
    subsection("A. Append, capacity, rewind, release")
    mx.random.seed(1)
    cache = TinyKvFullCache(capacity=16)
    _append_chunks(cache, [4, 6])
    print(
        f"  after 4+6: offset={cache.offset}, "
        f"allocated={cache.allocated_tokens} токенов, "
        f"storage={mb(cache.allocated_bytes)}"
    )
    assert cache.offset == 10 and cache.allocated_tokens == 16

    # Rewind 3: хвост откатывается, storage остаётся.
    trimmed = cache.rewind(3)
    print(f"  rewind(3): trimmed={trimmed}, offset={cache.offset}")
    assert cache.offset == 7

    # Дописываем снова — capacity переиспользуется, growth нет.
    growth_before = cache.growth_copy_bytes
    _append_chunks(cache, [5])
    print(
        f"  append(5): offset={cache.offset}, "
        f"growth_copy={cache.growth_copy_bytes - growth_before} B"
    )
    assert cache.offset == 12 and cache.growth_copy_bytes == growth_before

    # Overflow.
    try:
        _append_chunks(cache, [5])
    except ValueError as error:
        print(f"  ✓ overflow отклонён: {error}")
    else:
        raise AssertionError("overflow не был отклонён")

    cache.materialize()
    cache.release()
    print(f"  release: offset={cache.offset}, storage={cache.allocated_bytes} B")
    assert cache.keys is None and cache.offset == 0


def capacity_vs_growth():
    subsection("B. Capacity vs динамический рост")
    mx.random.seed(2)
    chunks = [1] * 64

    dynamic = TinyKvFullCache(capacity=None)
    _append_chunks(dynamic, chunks)
    fixed = TinyKvFullCache(capacity=64)
    _append_chunks(fixed, chunks)

    print(
        f"  dynamic: allocations={dynamic.allocations}, "
        f"growth_copy={dynamic.growth_copy_bytes} B, "
        f"storage={dynamic.allocated_bytes} B"
    )
    print(
        f"  capacity=64: allocations={fixed.allocations}, "
        f"growth_copy={fixed.growth_copy_bytes} B, "
        f"storage={fixed.allocated_bytes} B"
    )
    note(
        "Без capacity каждый рост перекладывает весь накопленный KV. "
        "С capacity storage выделяется один раз; плата — резерв под максимум."
    )


def attention_check():
    subsection("C. Кэш как источник K/V для attention")
    mx.random.seed(3)
    cache = TinyKvFullCache(capacity=12)
    h, d = 2, 8
    k = mx.random.normal((1, h, 12, d)).astype(mx.float16)
    v = mx.random.normal((1, h, 12, d)).astype(mx.float16)
    cache.update_and_fetch(k[:, :, :7], v[:, :, :7])
    cache.update_and_fetch(k[:, :, 7:], v[:, :, 7:])
    keys, values, offset = cache.keys, cache.values, cache.offset
    assert offset == 12

    q = mx.random.normal((h, 1, d)).astype(mx.float16)
    scores = mx.matmul(q, keys[0].swapaxes(-2, -1)) * (d**-0.5)
    probs = mx.softmax(scores.astype(mx.float32), axis=-1).astype(q.dtype)
    out = mx.matmul(probs, values[0])
    ref_scores = mx.matmul(q, k[0].swapaxes(-2, -1)) * (d**-0.5)
    ref_probs = mx.softmax(ref_scores.astype(mx.float32), axis=-1).astype(q.dtype)
    ref = mx.matmul(ref_probs, v[0])
    ok = mx.allclose(out, ref, rtol=2e-2, atol=2e-2)
    print(f"  ✓ attention по cache совпал с dense: {bool(ok)}")
    assert bool(ok)


def main():
    section("24. Dense KV-cache с capacity")
    semantics()
    capacity_vs_growth()
    attention_check()

    note(
        "Этот кэш — эталон для paged (inference/19) и BatchKVCache (inference/23): "
        "он хранит непрерывный [1, H, capacity, D]."
    )
    note(
        "rewind нужен speculative decoding (inference/21), release — завершению "
        "запроса, materialize — ограничению ленивого графа MLX."
    )

    section("Вывод")
    note("capacity убирает growth-копии ценой резерва памяти.")
    note("offset — единственная логическая граница; storage может быть больше.")
    note("rewind откатывает хвост, не освобождая capacity.")
    note("release возвращает кэш в пустое состояние.")


if __name__ == "__main__":
    main()
