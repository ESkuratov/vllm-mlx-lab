# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 11: roofline — где кончается «упор в память».

Идея
----
У инференса два потолка:

* **memory roofline** — за один decode-шаг модель читает все веса один раз,
  сколько бы последовательностей ни было в батче. Значит предел
  tokens/s = B × bandwidth / размер_весов — растёт линейно с батчем.
* **compute roofline** — на каждый токен нужно ~2·N параметров FLOPs.
  Предел tokens/s = пиковые FLOPs / (2·N) — не зависит от батча.

Пока мы упираемся в память, пропускная способность растёт с батчем. Когда
рост прекращается — начали упираться в compute.

Что измеряем: достигнутую пропускную способность и «достигнутый bandwidth»
(размер весов / время шага) при разных размерах батча.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/11_roofline.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mlx_lm.generate import batch_generate
from mlx_lm.sample_utils import make_sampler

from common import (
    MODEL_ID,
    base_token_ids,
    gb,
    load_model,
    model_nbytes,
    note,
    section,
    subsection,
)

N_PROMPTS = 16
PROMPT = 64
MAX_TOKENS = 32
BATCHES = (1, 2, 4, 8, 16)


def bar(value: float, max_value: float, width: int = 30) -> str:
    filled = int(round(width * value / max_value)) if max_value else 0
    return "█" * filled + "·" * (width - filled)


def main() -> None:
    section("11. Roofline: memory-bound vs compute-bound")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    sampler = make_sampler(temp=0.0)
    weights = model_nbytes(model)
    print(f"  размер весов: {gb(weights)}, промптов: {N_PROMPTS}")
    prompts = [base_token_ids(tok, PROMPT).tolist() for _ in range(N_PROMPTS)]

    try:
        from vllm_mlx.optimizations import detect_hardware

        bw = detect_hardware().memory_bandwidth_gbs
        print(f"  bandwidth (спека): {bw} GB/s")
    except Exception:
        bw = None

    subsection("A. Пропускная способность и достигнутый bandwidth")
    print(
        f"  {'батч':>5}{'мс/шаг':>9}{'ток./шаг':>10}{'tokens/s':>10}"
        f"{'GB/s шага':>11}"
    )
    rows = []
    for b in BATCHES:
        r = batch_generate(
            model, tok, prompts, max_tokens=MAX_TOKENS,
            completion_batch_size=b, prefill_batch_size=b, sampler=sampler,
        )
        s = r.stats
        step_s = s.decode_time / max(s.generation_steps, 1)
        tps = s.generation_tps
        achieved_bw = weights / step_s / 1e9 if step_s else 0.0
        rows.append((b, step_s * 1000, s.generation_tokens / max(s.generation_steps, 1), tps, achieved_bw))
        print(
            f"  {b:>5}{step_s * 1000:>9.2f}"
            f"{s.generation_tokens / max(s.generation_steps, 1):>10.1f}"
            f"{tps:>10.0f}{achieved_bw:>11.0f}"
        )

    subsection("B. Где потолок (наглядно)")
    tps_max = max(r[3] for r in rows)
    for b, _, _, tps, achieved_bw in rows:
        print(f"  B={b:<3}{tps:>8.0f} tok/s  {bar(tps, tps_max)}")
    note(
        "Пока «GB/s шага» держится у предела памяти, рост throughput почти "
        "линейный. Когда он падает, а ток./шаг растёт медленнее — включился compute."
    )

    if bw:
        subsection("C. Сравнение с теоретическими потолками")
        for b, step_ms, tok_step, tps, achieved in rows:
            mem_ceiling = b * bw * 1e9 / weights
            print(
                f"  B={b:<3} memory-roofline ≤ {mem_ceiling:>6.0f} tok/s, "
                f"факт {tps:>6.0f} ({100 * tps / mem_ceiling:>3.0f}%)"
            )
        note(
            "Memory-roofline растёт линейно с B, но фактическая кривая "
            "выравнивается: значит дальше рост ограничивает уже compute "
            "(и накладные расходы), а не чтение весов."
        )

    section("Вывод")
    note("Батчинг поднимает memory-roofline (одно чтение весов → много токенов).")
    note("Но упирается в compute-roofline: на каждый токен нужны FLOPs.")
    note("Отсюда баланс batch size ↔ latency ↔ throughput во vllm-mlx.")


if __name__ == "__main__":
    main()
