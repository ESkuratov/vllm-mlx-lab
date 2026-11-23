# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 4: paged attention и переиспользование префиксов.

Идея
----
KV-кэш — это огромный изменяемый буфер. Если выделять его «непрерывно» под
каждый запрос, память фрагментируется: запросы приходят/уходят, длины разные.
Paged attention нарезает кэш на блоки фиксированного размера (например, 16
токенов) и ведёт таблицу «какие блоки принадлежат какому запросу» — как
страницы виртуальной памяти в ОС.

Из этого бесплатно получаются:
* разделение одинаковых блоков (общий системный промпт) через ref-count;
* copy-on-write при ветвлении;
* вытеснение по LRU, когда память кончается.

Здесь используется настоящий менеджер из vllm-mlx:
``vllm_mlx/paged_cache.py`` (адаптация BlockPool из vLLM). Модель не нужна —
работаем с метаданными блоков.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/04_paged_attention.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm_mlx.paged_cache import PagedCacheManager

from common import note, section, subsection

BLOCK = 16


def add_request(mgr: PagedCacheManager, req_id: str, tokens: list[int]):
    """Аллокация блоков под запрос с переиспользованием общего префикса."""
    table = mgr.create_block_table(req_id)

    # 1) найти уже посчитанные (полные) блоки префикса
    cached, n_cached = mgr.get_computed_blocks(tokens)
    for block in cached:
        mgr.increment_ref(block.block_id)  # защищаем общий блок от вытеснения
        table.block_ids.append(block.block_id)
        table.num_tokens += mgr.block_size

    # 2) выделить новые блоки под остаток
    remaining = len(tokens) - n_cached
    n_new = (remaining + mgr.block_size - 1) // mgr.block_size
    new_blocks = mgr.get_new_blocks(n_new) if n_new else []
    for i, block in enumerate(new_blocks):
        tokens_in_block = min(mgr.block_size, remaining - i * mgr.block_size)
        mgr.add_block_to_table(table, block, tokens_in_block)

    # 3) зарегистрировать полные блоки в hash-кэше (для будущего переиспользования)
    all_blocks = cached + new_blocks
    num_full = len(tokens) // mgr.block_size
    mgr.cache_full_blocks(
        all_blocks, tokens, num_cached_blocks=len(cached), num_full_blocks=num_full
    )
    return table


def main() -> None:
    section("4. Paged attention и разделение префиксов")
    print(f"  block_size = {BLOCK} токенов")

    # ---------------------------------------------------- блоки как единица
    subsection("A. Память выделяется блоками, а не запросами целиком")
    mgr = PagedCacheManager(block_size=BLOCK, max_blocks=64)
    tokens = list(range(40))
    table = add_request(mgr, "req-1", tokens)
    print(f"  40 токенов → {len(table.block_ids)} блока: {table.block_ids}")
    print(
        f"  ёмкость: {len(table.block_ids) * BLOCK} токенов, "
        f"полезно: 40, внутренняя фрагментация: {len(table.block_ids) * BLOCK - 40}"
    )
    note(
        "Последний блок заполнен не полностью, но это цена фиксированного "
        "размера блока: зато все блоки взаимозаменяемы."
    )

    # --------------------------------------------------- разделение префикса
    subsection("B. Общий префикс не дублируется")
    mgr = PagedCacheManager(block_size=BLOCK, max_blocks=64)
    system = list(range(0, 48))          # 3 полных блока
    r1_tokens = system + list(range(100, 116))
    r2_tokens = system + list(range(200, 216))
    t1 = add_request(mgr, "user-A", r1_tokens)
    hits_before = mgr.get_stats().cache_hits
    t2 = add_request(mgr, "user-B", r2_tokens)

    shared_ids = [b for b in t2.block_ids if b in t1.block_ids]
    print(f"  user-A блоки: {t1.block_ids}")
    print(f"  user-B блоки: {t2.block_ids}")
    print(f"  общие физические блоки: {shared_ids}")
    print(
        f"  ref-count общих блоков: "
        f"{[mgr.allocated_blocks[b].ref_count for b in shared_ids]}"
    )
    print(f"  cache hits при обработке user-B: {mgr.get_stats().cache_hits - hits_before}")
    note(
        "user-B переиспользует 3 блока системного промпта вместо повторного "
        "prefill и повторной памяти — это и есть prefix caching."
    )

    # -------------------------------------------------------------- COW
    subsection("C. Copy-on-Write при ветвлении")
    before_copies = mgr.get_stats().cow_copies
    _, copied = mgr.get_blocks_for_generation(t2)
    after = mgr.get_stats()
    print(f"  был COW: {copied}, cow_copies: {before_copies} → {after.cow_copies}")
    print(
        f"  блоки user-A после COW: {t1.block_ids}\n"
        f"  блоки user-B после COW: {t2.block_ids}"
    )
    print(f"  shared_blocks теперь: {after.shared_blocks}")
    note(
        "Как только user-B собирается писать в общий блок, он копирует его к "
        "себе. user-A ничего не замечает — оригинал не тронут."
    )

    # ----------------------------------------------------------- LRU-eviction
    subsection("D. LRU-вытеснение при нехватке памяти")
    mgr = PagedCacheManager(block_size=BLOCK, max_blocks=32)
    # Поток коротких запросов: каждый приходит, получает блоки и завершается.
    # Освободившиеся блоки возвращаются в пул и переиспользуются под новые
    # запросы — старые записи кэша при этом вытесняются (LRU).
    for r in range(40):
        toks = list(range(r * 1000, r * 1000 + 48))
        add_request(mgr, f"req-{r}", toks)
        mgr.delete_block_table(f"req-{r}")  # запрос завершился
    s = mgr.get_stats()
    print(
        f"  после 40 коротких запросов: allocated={s.allocated_blocks}, "
        f"free={s.free_blocks}, evictions={s.evictions}"
    )
    evicted = mgr.evict_lru_blocks(4)
    print(f"  evict_lru_blocks(4): вытеснено ещё {evicted}")
    miss, n = mgr.get_computed_blocks(list(range(0, 48)))
    print(f"  самый старый префикс ещё в кэше? найдено токенов={n}")
    note(
        "Вытесняются в первую очередь блоки, которые давно не трогали "
        "(LRU), причём только неразделяемые (ref_count == 1)."
    )

    # -------------------------------------------------- экономия памяти
    subsection("E. Сколько экономит блочная нарезка")
    n_requests, max_seq = 32, 1024
    used_tokens = 32 * 130  # реально сгенерировано суммарно
    block_bytes = 1024  # условных байт на токен
    naive = n_requests * max_seq * block_bytes
    paged = used_tokens * block_bytes
    print(f"  наивно (резерв под max_seq={max_seq} на запрос): {naive / 1024**2:>8.1f} MB")
    print(f"  paged (только реально занятые токены):            {paged / 1024**2:>8.1f} MB")
    print(f"  экономия: {naive / paged:.1f}x")
    note(
        "Блоки выделяются по мере роста контекста и переиспользуются между "
        "запросами — вместо резервирования максимума на каждый запрос."
    )

    section("Вывод")
    note("Paged attention = виртуальная память для KV-кэша.")
    note("Блоки → разделение префиксов, COW и LRU-вытеснение.")
    note(
        "Во vllm-mlx это включается флагом --use-paged-cache "
        "(--paged-cache-block-size, --max-cache-blocks)."
    )


if __name__ == "__main__":
    main()
