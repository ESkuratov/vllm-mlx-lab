# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 04: kernel fusion и mx.compile.

Идея
----
Каждый запуск ядра на GPU стоит накладных расходов. Если сложить несколько
операций (tanh, multiply, maximum, ...) в одно «слитое» ядро, накладные расходы
падают, а промежуточные массивы не пишутся в память. За это отвечает
`mx.compile` — аналог `torch.compile` / CUDA graphs.

Во vllm-mlx это ровно то, что делает `model_runner.py`: `mx.compile(model.__call__)`.

Что измеряем: эффект компиляции на цепочке операций и на реальном forward.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/04_kernel_fusion.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from common import MODEL_ID, base_token_ids, load_model, note, section, subsection


def bench(fn, n: int = 9) -> float:
    fn()
    mx.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        mx.synchronize()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> None:
    section("04. Kernel fusion и mx.compile")

    # ------------------------------------------------ цепочка операций
    subsection("A. Слияние цепочки elementwise-операций")

    def make_chain(length: int):
        def f(x):
            for _ in range(length):
                x = mx.tanh(x) * x + mx.maximum(x, 0.0)
            return x

        return f

    size = (1024, 1024)
    x = mx.random.normal(size).astype(mx.float32)
    print(f"  тензор {size}, цепочка одинаковых операций:")
    print(f"  {'длина':>7}{'без compile':>14}{'compile':>12}{'ускорение':>11}")
    for length in (1, 4, 16, 64):
        f = make_chain(length)
        fc = mx.compile(f)

        def raw():
            y = f(x)
            mx.eval(y)

        def comp():
            y = fc(x)
            mx.eval(y)

        t_raw = bench(raw)
        t_comp = bench(comp)
        print(f"  {length:>7}{t_raw * 1e3:>12.3f} ms{t_comp * 1e3:>10.3f} ms{t_raw / t_comp:>10.2f}x")
    note(
        "Чем длиннее цепочка, тем больше ядер склеивается в одно — накладные "
        "расходы на запуск и промежуточные записи в память сокращаются."
    )

    # ------------------------------------------------------ реальный forward
    subsection("B. Реальный forward модели: prefill")
    model, tok = load_model()
    ids = base_token_ids(tok, 32)
    try:
        cf = mx.compile(model.__call__)
        mx.eval(cf(ids[None]))
    except Exception as exc:  # pragma: no cover
        note(f"mx.compile недоступен: {exc}")
        return

    def raw_fwd():
        mx.eval(model(ids[None]))

    def comp_fwd():
        mx.eval(cf(ids[None]))

    t_raw = bench(raw_fwd)
    t_comp = bench(comp_fwd)
    print(f"  prefill: без compile {t_raw * 1e3:.1f} ms, compile {t_comp * 1e3:.1f} ms, "
          f"ускорение {t_raw / t_comp:.2f}x")
    note(
        "На большом forward слияние почти не помогает: модель и так состоит из "
        "крупных эффективных ядер, а компиляция целого графа не бесплатна."
    )

    # ----------------------------------------------------------- decode
    subsection("C. Реальный forward модели: decode")
    prompt = base_token_ids(tok, 64)

    def decode_time(forward):
        cache = make_prompt_cache(model)
        logits = forward(prompt[None], cache)
        mx.eval(logits)
        tid = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
        times = []
        for _ in range(12):
            t0 = time.perf_counter()
            logits = forward(tid, cache)
            mx.eval(logits)
            times.append(time.perf_counter() - t0)
            tid = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
        return statistics.median(times[2:])

    raw_t = decode_time(model.__call__)
    try:
        cf2 = mx.compile(model.__call__)
        comp_t = decode_time(cf2)
        print(f"  decode: без compile {raw_t * 1e3:.2f} ms/шаг, compile {comp_t * 1e3:.2f} ms/шаг, "
              f"ускорение {raw_t / comp_t:.2f}x")
    except Exception as exc:
        print(f"  decode: без compile {raw_t * 1e3:.2f} ms/шаг; compile не удался: {exc}")
        note(
            "mx.compile принимает только деревья массивов, но не объект KVCache. "
            "Значит stateful-decode с кэшем этим способом не ускорить — "
            "компилировать можно только stateless-функции (например, префилл)."
        )

    subsection("D. Когда fusion помогает, а когда нет")
    note("Помогает: длинные цепочки мелких/память-связанных операций, много launch-ов.")
    note("Не помогает: крупные GEMM/attention, которые и так оптимальны.")
    note("Во vllm-mlx mx.compile включается в model_runner.py, но эффект скромный.")

    section("Вывод")
    note("mx.compile = слияние графа в меньшее число ядер.")
    note("Выигрыш тем больше, чем больше мелких операций подряд.")
    note("Для LLM на MLX основной выигрыш — в памяти/батчинге, а не в fusion.")


if __name__ == "__main__":
    main()
