# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 10: квантизация весов (4 vs 8 бит vs fp16).

Идея
----
Квантизация весов хранит числа не в fp16, а целыми: тензор бьётся на группы
по `group_size`, на группу хранится шкала и смещение. Это в разы уменьшает
память модели, но вносит ошибку в сами веса.

Что измеряем:
  1. сколько весит модель и из чего она состоит;
  2. ошибку восстановления весов при 2/4/8 битах;
  3. оценку памяти модели при разных битностях.

Для сравнения берём один настоящий слой модели: «эталон» — его распакованные
fp16-веса, затем квантизуем их заново с разной битностью.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/10_weight_quantization.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn

from common import (
    MODEL_ID,
    gb,
    load_model,
    model_nbytes,
    note,
    section,
    subsection,
)

GROUP = 64


def num_params_and_layers(model):
    """Число параметров модели и статистика по квантованным слоям."""
    params = 0
    quant = []
    for name, module in model.named_modules():
        if isinstance(module, nn.QuantizedLinear):
            out, packed = module.weight.shape
            params += out * (packed * 32 // module.bits)
            quant.append((name, module))
        elif isinstance(module, nn.Linear):
            params += module.weight.size
    return params, quant


def rel_error(w: mx.array, wq: mx.array) -> float:
    diff = (w - wq).astype(mx.float32)
    ref = w.astype(mx.float32)
    return float(mx.sqrt(mx.sum(diff * diff) / mx.sum(ref * ref)))


def main() -> None:
    section("10. Квантизация весов: 4 vs 8 бит vs fp16")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()

    measured = model_nbytes(model)
    params, quant_layers = num_params_and_layers(model)

    # ------------------------------------------------------- состав модели
    subsection("A. Сколько весит модель и из чего состоит")
    print(f"  параметров: {params / 1e9:.3f} B")
    print(f"  квантованных Linear: {len(quant_layers)}")
    if quant_layers:
        _, sample = quant_layers[0]
        print(
            f"  пример: bits={sample.bits}, group_size={sample.group_size}, "
            f"weight={sample.weight.shape} ({sample.weight.dtype})"
        )
    print(f"  фактический размер: {gb(measured)}")

    subsection("B. Оценка памяти при разной битности весов")
    print(f"  {'бит':>5}{'память (оценка)':>18}{'доля от fp16':>14}")
    base = None
    for bits in (16, 8, 4, 2):
        approx = params * bits / 8
        if base is None:
            base = approx
        print(f"  {bits:>5}{gb(approx):>18}{approx / base:>13.0%}")
    note(
        "fp16 → 4 бита даёт ~4× меньше памяти; именно поэтому 4-битные модели "
        "влезают на ноутбуки. Оценка не учитывает шкалы/смещения и эмбеддинги."
    )

    # -------------------------------------------------- ошибка восстановления
    subsection("C. Ошибка восстановления весов (на примере слоя)")
    # берём самый крупный квантованный слой
    name, layer = max(quant_layers, key=lambda nl: nl[1].weight.size)
    w_ref = mx.dequantize(
        layer.weight, layer.scales, layer.biases,
        group_size=layer.group_size, bits=layer.bits,
    )
    mx.eval(w_ref)
    print(f"  слой: {name}  форма {w_ref.shape}")
    print(f"  {'бит':>5}{'отн. ошибка':>15}{'память слоя':>15}")
    for bits in (8, 4, 2):
        wq, scales, biases = mx.quantize(w_ref, group_size=GROUP, bits=bits)
        w_deq = mx.dequantize(wq, scales, biases, group_size=GROUP, bits=bits)
        mx.eval(w_deq)
        size = wq.nbytes + scales.nbytes + biases.nbytes
        print(f"  {bits:>5}{rel_error(w_ref, w_deq):>15.4%}{size / 1024**2:>13.1f} MB")
    note(
        "8 бит почти не портят веса, 4 — вносят заметный шум, 2 — ломают заметно "
        "сильнее. Но модели на 4 битах всё равно работают: ошибки усредняются."
    )

    subsection("D. Как это включено во vllm-mlx")
    note("Кванты приходят из готовой модели (mlx-community/*-4bit) или задаются")
    note("при конвертации: vllm-mlx model convert ..., либо через mlx-lm.")
    note("Отдельного рантайм-флага для битности весов нет — она «зашита» в чекпойнт.")

    section("Вывод")
    note("Меньше бит → меньше памяти и больше шума в весах.")
    note("4 бита — рабочий компромисс; 8 бит безопаснее; fp16 — эталон качества.")


if __name__ == "__main__":
    main()
