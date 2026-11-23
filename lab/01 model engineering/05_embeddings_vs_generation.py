# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 05: эмбеддинги vs генерация — два разных пути модели.

Идея
----
У generative-модели есть два выхода:
* **эмбеддинги** — внутренние векторы токенов/текста (нужен только «энкодер»);
* **генерация** — логиты следующего токена (энкодер + lm_head + авторегрессия).

Путь эмбеддингов гораздо дешевле: один forward на текст, без lm_head и без
пошагового decode. На нём строят поиск и ранжирование. Во vllm-mlx это
`embedding.py` / `rerank.py` и флаги `--embedding-model`, `--rerank-model`.

Что показываем:
  1. эмбеддинги токенов: близкие по смыслу слова ближе в пространстве;
  2. sentence-эмбеддинги: mean-pool + L2-нормализация и косинусная близость;
  3. стоимость: embedding против генерации.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/05_embeddings_vs_generation.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from common import MODEL_ID, load_model, note, section, subsection


def l2norm(x: mx.array) -> mx.array:
    return x / (mx.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def cos(a: mx.array, b: mx.array) -> float:
    return float(mx.sum(l2norm(a[None]) * l2norm(b[None])))


def embed_sentence(model, tok, text: str) -> mx.array:
    ids = mx.array(tok.encode(text))
    hidden = model.model(ids[None])  # «энкодерный» выход: (1, L, H)
    mx.eval(hidden)
    return mx.mean(hidden[0], axis=0)  # mean-pool по токенам


def main() -> None:
    section("05. Эмбеддинги vs генерация")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()

    # -------------------------------------------------- эмбеддинги токенов
    subsection("A. Эмбеддинги токенов: близкие слова — близкие векторы")
    embed = model.model.embed_tokens

    def tok_vec(word: str) -> mx.array:
        ids = mx.array(tok.encode(word))
        v = embed(ids)
        mx.eval(v)
        return mx.mean(v, axis=0)

    pairs = [
        (" king", " queen"),
        (" king", " banana"),
        (" Paris", " London"),
        (" Paris", " potato"),
        (" run", " running"),
    ]
    print(f"  {'пара':<24}{'cos':>8}")
    for a, b in pairs:
        print(f"  {a + ' /' + b:<24}{cos(tok_vec(a), tok_vec(b)):>8.3f}")
    note(
        "Похожие по смыслу слова лежат ближе; несвязанные — дальше. Это и есть "
        "основа семантического поиска."
    )

    # ------------------------------------------------ sentence-эмбеддинги
    subsection("B. Sentence-эмбеддинги: mean-pool + L2-нормализация")
    sentences = {
        "cat1": "A cat sits quietly on the warm mat.",
        "cat2": "A dog lies on the soft rug.",
        "fin": "The stock market crashed yesterday.",
        "code": "The function returns the sum of two integers.",
    }
    embs = {k: embed_sentence(model, tok, v) for k, v in sentences.items()}
    print(f"  {'пара':<14}{'cos':>8}")
    for a, b in [("cat1", "cat2"), ("cat1", "fin"), ("cat1", "code"), ("fin", "code")]:
        print(f"  {a + '/' + b:<14}{cos(embs[a], embs[b]):>8.3f}")
    note(
        "mean-pooling + нормализация дают вектор предложения. Животные ближе "
        "друг к другу, чем к финансам. Специализированные embedding-модели "
        "обучены контрастивно и делают это заметно лучше."
    )

    # ---------------------------------------------------- стоимость
    subsection("C. Стоимость: embedding vs генерация")
    text = "The quick brown fox jumps over the lazy dog and runs away."
    ids = mx.array(tok.encode(text))

    def embed_call():
        h = model.model(ids[None])
        mx.eval(h)

    def embed_time():
        embed_call()
        mx.synchronize()
        ts = []
        for _ in range(7):
            t0 = time.perf_counter()
            embed_call()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts)

    t_embed = embed_time()

    def gen_step_time(n: int = 24) -> float:
        cache = make_prompt_cache(model)
        logits = model(ids[None], cache=cache)
        mx.eval(logits)
        tid = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
        ts = []
        for _ in range(n):
            t0 = time.perf_counter()
            logits = model(tid, cache=cache)
            mx.eval(logits)
            ts.append(time.perf_counter() - t0)
            tid = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
        return statistics.median(ts)

    t_step = gen_step_time()
    print(f"  эмбеддинг ({len(ids)} токенов): {t_embed * 1000:>6.1f} ms — один forward")
    print(f"  один decode-шаг:            {t_step * 1000:>6.1f} ms — и это только 1 токен")
    print(f"  в секунду: эмбеддингов ~{1 / t_embed:>5.0f}, токенов генерации ~{1 / t_step:>5.0f}")
    note(
        "Embedding делает один проход и всё; генерация платит за каждый токен. "
        "Плюс эмбеддинги легко батчатся (нет авторегрессии и KV-кэша)."
    )

    # ---------------------------------------------------- пути
    subsection("D. Два разных «выхода» одной модели")
    print("  ids → embed_tokens → [слои энкодера] ─┬─→ hidden → mean-pool → embedding")
    print("                                         └─→ lm_head → логиты → выбор токена")
    note("Во vllm-mlx: --embedding-model (embedding.py), --rerank-model (rerank.py).")
    note("Embedding-модели обычно меньше и обучаются отдельно (например, BGE/E5).")

    section("Вывод")
    note("Эмбеддинги — дешёвый неавторегрессионный путь для поиска/ранжирования.")
    note("Генерация — дорогой авторегрессионный путь; другой выход той же сети.")
    note("Один и тот же transformer-стек, но разные головы и разная стоимость.")


if __name__ == "__main__":
    main()
