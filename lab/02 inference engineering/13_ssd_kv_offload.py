# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 13: выгрузка KV-кэша на диск (SSD-тир).

Идея
----
Prefix caching держит KV-кэши в RAM. Если место кончилось, кэш можно
сериализовать на диск и при следующем похожем запросе загрузить обратно —
дешевле, чем пересчитывать prefill заново.

Здесь используем штатные save/load из mlx-lm (ровно это делает
``vllm_mlx/ssd_cache.py``: сериализует слои кэша в safetensors и кладёт на диск).

Что измеряем: размер файла, время сохранения/загрузки и сравнение с повторным
prefill.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/13_ssd_kv_offload.py"
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.models.cache import (
    load_prompt_cache,
    make_prompt_cache,
    save_prompt_cache,
)

from common import (
    MODEL_ID,
    base_token_ids,
    load_model,
    note,
    section,
    subsection,
)

CONTEXT = 1024


def fresh_prefill(model, ids):
    cache = make_prompt_cache(model)
    t0 = time.perf_counter()
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    mx.synchronize()
    return cache, time.perf_counter() - t0


def main() -> None:
    section("13. SSD-тир: выгрузка KV-кэша на диск")
    print(f"  модель: {MODEL_ID}, контекст: {CONTEXT}")
    model, tok = load_model()
    ids = base_token_ids(tok, CONTEXT)

    # -------------------------------------------------------- prefill
    subsection("A. Prefill в память")
    cache, t_prefill = fresh_prefill(model, ids)
    size_mem = sum(getattr(c, "nbytes", 0) for c in cache)
    print(f"  prefill: {t_prefill * 1000:.0f} ms, кэш в RAM: {size_mem / 1024**2:.1f} MB")

    # ----------------------------------------------------------- save
    subsection("B. Сохранение на диск и загрузка")
    tmp = tempfile.mkdtemp(prefix="kv-ssd-")
    path = os.path.join(tmp, "prompt_cache.safetensors")
    t0 = time.perf_counter()
    save_prompt_cache(path, cache)
    t_save = time.perf_counter() - t0
    size_disk = os.path.getsize(path)
    print(f"  сохранение: {t_save * 1000:>7.0f} ms, файл: {size_disk / 1024**2:.1f} MB")

    t0 = time.perf_counter()
    loaded = load_prompt_cache(path)
    arrays = [c.keys for c in loaded if getattr(c, "keys", None) is not None]
    arrays += [c.values for c in loaded if getattr(c, "values", None) is not None]
    mx.eval(*arrays)  # подтянуть данные с диска в память
    mx.synchronize()
    t_load = time.perf_counter() - t0
    print(f"  загрузка:   {t_load * 1000:>7.0f} ms")
    note(
        "Загрузка быстрая, потому что файл ещё в страничном кэше ОС; "
        "с холодного диска будет медленнее."
    )

    # -------------------------------------------------------- сравнение
    subsection("C. Что дешевле: загрузить с диска или пересчитать")
    cache2, t_re = fresh_prefill(model, ids)
    print(f"  повторный prefill: {t_re * 1000:>7.0f} ms")
    print(f"  загрузка с диска:  {t_load * 1000:>7.0f} ms")
    if t_load < t_re:
        print(f"  → загрузка быстрее в {t_re / t_load:.2f}x")
    else:
        print(f"  → prefill быстрее (диск медленный, но освобождает RAM)")

    note(
        "Смысл SSD-тира не всегда в скорости, а в том, что кэш переживает "
        "вытеснение из RAM и даже перезапуск процесса."
    )
    note(
        "Во vllm-mlx: --ssd-cache-dir, --ssd-cache-max-gb; код в "
        "vllm_mlx/ssd_cache.py (SSDCacheTier, SSDIndex)."
    )

    section("Вывод")
    note("KV-кэш сериализуется в safetensors и живёт на диске.")
    note("Диск медленнее RAM, но дешевле и переживает вытеснение/рестарт.")

    # уборка
    try:
        os.remove(path)
        os.rmdir(tmp)
    except OSError:
        pass


if __name__ == "__main__":
    main()
