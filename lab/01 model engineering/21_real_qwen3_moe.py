# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 21: реальный Qwen3-MoE 30B-A3B (опционально, тяжёлый).

Идея
----
Lab/51 собирает уменьшенную Qwen3-MoE на синтетике. Здесь мы проверяем
реальный MLX-чекпоинт:

* загружаем 4-битный Qwen3-Coder-30B-A3B;
* смотрим конфиг: число экспертов, top_k, MoE intermediate;
* проверяем, что слой действительно sparse MoE (`switch_mlp`);
* генерируем несколько токенов и меряем пиковую память.

Это тяжёлая лаборатория: ~17 GB весов и близко к рабочему set на 24 GB
машине. Если не хватает памяти — скрипт печатает диагностику и выходит,
не ломая остальные лабы.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/21_real_qwen3_moe.py"

Своя модель:
    LAB_MOE_MODEL=... python "lab/01 model engineering/21_real_qwen3_moe.py"
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_sampler

from common import gb, note, section, subsection

DEFAULT_MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
PROMPT = "In one sentence, what is a mixture-of-experts model?"


def main() -> None:
    section("21. Реальный Qwen3-MoE 30B-A3B")
    model_id = os.environ.get("LAB_MOE_MODEL", DEFAULT_MODEL)
    print(f"  модель: {model_id}")
    print("  это ~17 GB весов; на 24 GB машине может быть тесно.")

    try:
        t0 = time.perf_counter()
        model, tok = load(model_id)
        mx.synchronize()
        load_s = time.perf_counter() - t0
    except Exception as error:  # noqa: BLE001 - тяжёлая загрузка
        print(f"  ✗ не удалось загрузить: {error}")
        note(
            "Освободите память или укажите меньшую MoE-модель через "
            "LAB_MOE_MODEL. Механика MoE без 30B — в model/17 и model/19."
        )
        return

    args = model.args
    print(f"  загрузка: {load_s:.1f} с")
    print(
        f"  hidden={args.hidden_size}, layers={args.num_hidden_layers}, "
        f"experts={getattr(args, 'num_experts', '?')}, "
        f"top_k={getattr(args, 'num_experts_per_tok', '?')}, "
        f"moe_intermediate={getattr(args, 'moe_intermediate_size', '?')}"
    )

    subsection("A. Проверка sparse-слоя")
    layer = model.layers[0]
    mlp = layer.mlp
    print(f"  mlp type: {type(mlp).__name__}")
    switch = getattr(mlp, "switch_mlp", None)
    if switch is not None:
        print(f"  switch_mlp: {type(switch).__name__}")
        gate_proj = getattr(switch, "gate_proj", None)
        if gate_proj is not None:
            print(f"  expert weight shape: {gate_proj.weight.shape}")
        note("Это настоящий sparse MoE-слой, а не dense MLP.")
    else:
        note("В этом чекпоинте первый слой — dense; sparse-слои могут быть выше.")

    subsection("B. Генерация")
    messages = [{"role": "user", "content": PROMPT}]
    prompt = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    mx.metal.reset_peak_memory()
    t0 = time.perf_counter()
    text = generate(
        model,
        tok,
        prompt=prompt,
        max_tokens=16,
        sampler=make_sampler(0.0),
        verbose=False,
    )
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    print(f"  prompt: {PROMPT}")
    print(f"  output: {text.strip()[:400]!r}")
    print(f"  время: {elapsed:.2f} с")
    print(f"  peak memory: {gb(mx.metal.get_peak_memory())}")

    subsection("C. Вывод по памяти")
    total = sum(
        p.nbytes for p in model.parameters().values() if hasattr(p, "nbytes")
    )
    # parameters() — дерево; оценим через tree_flatten.
    from mlx.utils import tree_flatten

    total = sum(a.nbytes for _, a in tree_flatten(model.parameters()))
    print(f"  веса: {gb(total)}")
    print(
        "  Если peak близок к recommended working set — уменьшите batch, "
        "контекст или используйте меньшую модель."
    )

    note(
        "Lab/45 и model/19 остаются быстрыми синтетическими проверками "
        "механики. Эта лаборатория — реальный чекпоинт, поэтому тяжёлая."
    )

    section("Вывод")
    note("Реальный Qwen3-MoE загружается и генерирует текст.")
    note("Конфиг и switch_mlp подтверждают sparse MoE.")
    note("30B-A3B — тяжёлый; для CI используйте 45/51.")


if __name__ == "__main__":
    main()
