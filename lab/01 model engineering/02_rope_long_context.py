# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 02: RoPE и длинный контекст.

Идея
----
Внимание само по себе не знает порядок токенов: для него «A B» и «B A» —
одинаковый мешок. Порядок задаёт позиционное кодирование, в современных LLM —
RoPE (rotary position embedding): векторы q/k поворачиваются на угол,
пропорциональный позиции. Тогда скалярное произведение зависит только от
РАЗНОСТИ позиций.

Проблема: чем длиннее контекст, тем больше углы. Для «быстрых» измерений фаза
успевает намотаться много раз, и на очень больших расстояниях позиция
разрешается всё хуже — отсюда деградация на длинном контексте и трюки вроде
YaRN/NTK-масштабирования.

Что показываем:
  1. RoPE делает внимание относительным (зависит от i−j);
  2. «память» позиции угасает с расстоянием (декогеренция);
  3. реальный эффект «lost in the middle» на длинном промпте.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/02_rope_long_context.py"
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.models.cache import make_prompt_cache
from mlx_lm.sample_utils import make_sampler

from common import MODEL_ID, load_model, note, section, subsection


def rope(x: mx.array, positions: mx.array, base: float) -> mx.array:
    """Классический RoPE над последним измерением (L, dim)."""
    dim = x.shape[-1]
    half = dim // 2
    idx = mx.arange(half, dtype=mx.float32)
    theta = base ** (-2.0 * idx / dim)  # (half,)
    angle = positions[:, None].astype(mx.float32) * theta[None, :]  # (L, half)
    cos, sin = mx.cos(angle), mx.sin(angle)
    x1, x2 = x[..., :half], x[..., half:]
    return mx.concatenate([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)


def main() -> None:
    section("02. RoPE и длинный контекст")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    args = model.args
    head_dim = getattr(args, "head_dim", 128)
    base = getattr(args, "rope_theta", 10000.0)
    max_pos = getattr(args, "max_position_embeddings", None)
    print(f"  head_dim={head_dim}, rope_theta={base}, max_position_embeddings={max_pos}")

    # ------------------------------------------------ относительность
    subsection("A. RoPE делает внимание относительным")
    v = mx.random.normal((head_dim,)).astype(mx.float32)
    v = v / mx.linalg.norm(v)

    def score(i: int, j: int) -> float:
        qi = rope(v[None, :], mx.array([i]), base)[0]
        kj = rope(v[None, :], mx.array([j]), base)[0]
        return float(mx.sum(qi * kj))

    print(f"  {'i':>6}{'j':>6}{'i-j':>6}{'q·k':>12}")
    for i, j in [(0, 0), (5, 5), (10, 7), (100, 97), (1000, 997), (0, 3), (50, 53)]:
        print(f"  {i:>6}{j:>6}{i - j:>6}{score(i, j):>12.4f}")
    note(
        "Пары с одинаковой разностью (i−j) дают одинаковый q·k — позиция влияет "
        "через относительное расстояние, а не через абсолютную координату."
    )

    # ---------------------------------------------------- декогеренция
    subsection("B. Дальняя позиция «размывается»")
    print(f"  {'расстояние':>11}{'q·k':>12}")
    for d in (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096):
        print(f"  {d:>11}{score(0, d):>12.4f}")
    note(
        "С ростом расстояния корреляция падает и осциллирует: быстрые измерения "
        "многократно наматывают фазу и перестают различать позиции."
    )

    # ------------------------------------------------- периоды измерений
    subsection("C. Периоды: где ломается экстраполяция")
    idx = mx.arange(head_dim // 2, dtype=mx.float32)
    thetas = base ** (-2.0 * idx / head_dim)
    periods = 2 * math.pi / thetas
    p = periods.tolist()
    print(f"  самое «быстрое» измерение: период ≈ {p[0]:.2f} токена")
    print(f"  самое «медленное» измерение: период ≈ {p[-1]:.1f} токенов")
    for limit in (1024, 8192, 32768, 131072):
        wrap = sum(1 for x in p if x < limit)
        print(f"  при контексте {limit:>7}: наматываются {wrap:>3} из {head_dim // 2} измерений")
    note(
        "Часть измерений успевает «прокрутиться» уже на коротких дистанциях — "
        "они кодируют тонкую позицию, но плохо переносят большие сдвиги. "
        "Экстраполяция за пределы обучения (max_position_embeddings) теряет "
        "уникальность фаз — отсюда YaRN/NTK и интерполяция позиций."
    )

    # ------------------------------------------- lost in the middle
    subsection("D. Реальный эффект: «lost in the middle»")
    filler = tok.encode("The grass is green and the sky is blue. ")
    decoys = tok.encode("The decoy code is 0000. The decoy code is 1111. ")
    question = tok.encode("\nQuestion: What is the secret code? Answer:")
    codes = ["4821", "7391", "2058"]
    depths = [0.0, 0.5, 1.0]
    eos = set(getattr(tok, "eos_token_ids", {tok.eos_token_id}))

    for total_target in (512, 4096):
        print(f"  контекст {total_target} токенов:")
        print(f"    {'глубина':>9}{'найдено':>10}")
        for frac in depths:
            hits = 0
            for code in codes:
                fact = tok.encode(f"The secret code is {code}. ")
                n_before = int(frac * (total_target - len(fact)))
                before = (filler * (n_before // len(filler) + 1))[:n_before]
                after_len = total_target - len(fact) - len(before)
                after = (filler + decoys) * (after_len // (len(filler) + len(decoys)) + 1)
                after = after[:after_len]
                ids = mx.array(before + fact + after + question)
                cache = make_prompt_cache(model)
                logits = model(ids[None], cache=cache)
                mx.eval(logits)
                text = ""
                tid = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
                for _ in range(10):
                    t = int(tid.item())
                    if t in eos:
                        break
                    text += tok.decode([t])
                    logits = model(tid, cache=cache)
                    mx.eval(logits)
                    tid = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
                hits += int(code in text)
            print(f"    {frac:>8.0%}{hits:>7}/{len(codes)}")
    note(
        "Классический «lost in the middle» — факт в середине извлекается хуже, " 
        "чем в начале/конце. В этом прогоне эффект НЕ проявился: иголка слишком "
        "отличима, а контекст короток для модели. Эффект обычно виден на десятках "
        "тысяч токенов и на более сложных задачах — попробуй увеличить контекст "
        "или добавить похожие факты-обманки."
    )

    section("Вывод")
    note("RoPE: q/k поворачиваются по позиции → внимание относительно.")
    note("На больших расстояниях фазы декогерируют → длинный контекст хуже.")
    note("Лечится масштабированием позиций и окнами/квантизацией KV (02, 05, 13).")


if __name__ == "__main__":
    main()
