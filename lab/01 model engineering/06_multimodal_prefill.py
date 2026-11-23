# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 06: мультимодальный prefill — картинка как токены.

Идея
----
VLM добавляет к LLM «глаз»: картинка режется на патчи, каждый патч превращается
в вектор (patch embedding), проецируется в размерность LLM и подаётся как
обычные эмбеддинги токенов. Дальше всё как обычно: prefill, KV-кэш, decode.

Поэтому стоимость картинки = стоимость её токенов. Разрешение и размер патча
напрямую задают, сколько токенов «съест» изображение.

Здесь мы не берём настоящий vision-энкодер (это отдельная большая сеть), а
собираем интерфейс: патчи → проекция → `input_embeddings` реальной LLM и меряем
цену. Во vllm-mlx это `multimodal_processor.py` и флаг `--mllm`.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/06_multimodal_prefill.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import make_prompt_cache

from common import MODEL_ID, load_model, note, section, subsection


def patchify(image: np.ndarray, patch: int) -> np.ndarray:
    """(H, W, 3) → (N, patch*patch*3)."""
    h, w, c = image.shape
    ph, pw = h // patch, w // patch
    img = image[: ph * patch, : pw * patch]
    patches = img.reshape(ph, patch, pw, patch, c).transpose(0, 2, 1, 3, 4)
    return patches.reshape(ph * pw, patch * patch * c)


def main() -> None:
    section("06. Мультимодальный prefill: цена картинки")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    hidden = model.args.hidden_size

    # ------------------------------------------------------ «vision-энкодер»
    subsection("A. Картинка → патчи → токены")
    np.random.seed(0)
    image = np.random.rand(224, 224, 3).astype(np.float32)
    patch = 16
    patches = patchify(image, patch)
    print(f"  изображение: {image.shape}, патч {patch}×{patch}")
    print(f"  патчей: {patches.shape[0]}")

    # патч-эмбеддинг + проекция в размерность LLM (имитация обученного VLM)
    patch_dim = patches.shape[1]
    mx.random.seed(0)
    W_patch = mx.random.normal((patch_dim, hidden)) / (patch_dim**0.5)
    W_proj = mx.random.normal((hidden, hidden)) / (hidden**0.5)
    img_emb = mx.array(patches) @ W_patch
    img_emb = mx.tanh(img_emb @ W_proj)  # (N, H)
    img_emb = img_emb[None, :, :]  # (1, N, H)
    mx.eval(img_emb)
    print(f"  эмбеддинги картинки: {img_emb.shape}")

    # ------------------------------------------------------------ prefill
    subsection("B. Prefill картинки через реальную LLM")
    n_img = img_emb.shape[1]

    def image_prefill():
        cache = make_prompt_cache(model)
        logits = model(None, cache=cache, input_embeddings=img_emb)
        mx.eval(logits)
        return logits

    image_prefill()
    mx.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        image_prefill()
        ts.append(time.perf_counter() - t0)
    t_img = statistics.median(ts)

    # для сравнения — prefill столько же текстовых токенов
    text_ids = mx.random.randint(0, 1000, (1, n_img))
    mx.eval(text_ids)

    def text_prefill():
        cache = make_prompt_cache(model)
        logits = model(text_ids, cache=cache)
        mx.eval(logits)

    text_prefill()
    mx.synchronize()
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        text_prefill()
        ts.append(time.perf_counter() - t0)
    t_txt = statistics.median(ts)

    print(f"  {n_img} «картиночных» токенов: {t_img * 1000:>6.1f} ms")
    print(f"  {n_img} текстовых токенов:     {t_txt * 1000:>6.1f} ms")
    note(
        "Как только патчи стали эмбеддингами, LLM обрабатывает их как обычные "
        "токены — стоимость та же, что у текста."
    )

    # ---------------------------------------------- генерация из «мультимодального» префикса
    subsection("C. Продолжение генерации из мультимодального префикса")
    question = tok.encode("\nDescribe this image:")
    q_emb = model.model.embed_tokens(mx.array(question))[None, :, :]
    mx.eval(q_emb)
    full_emb = mx.concatenate([img_emb, q_emb], axis=1)
    cache = make_prompt_cache(model)
    logits = model(None, cache=cache, input_embeddings=full_emb)
    mx.eval(logits)
    tid = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    out = []
    for _ in range(12):
        t = int(tid.item())
        out.append(t)
        logits = model(tid, cache=cache)
        mx.eval(logits)
        tid = mx.argmax(logits[0, -1]).reshape(1, 1).astype(mx.uint32)
    print(f"  вывод (vision-энкодер случайный, поэтому текст бессмысленный):")
    print(f"    {tok.decode(out)!r}")
    note(
        "Смысл в том, что поток данных тот же: картинка + текст → общий префикс → "
        "prefill → decode. Реальный VLM отличается только обученным энкодером."
    )

    # ------------------------------------------------------- цена по разрешению
    subsection("D. Сколько токенов стоит картинка")
    print(f"  {'разрешение':>11}{'патч':>6}{'токенов':>9}{'с merger 2×2':>15}")
    for size in (224, 336, 448, 896):
        for p in (14, 16):
            n = (size // p) ** 2
            print(f"  {size:>9}px{p:>6}{n:>9}{n // 4:>15}")
    note(
        "При 224px и патче 14 получается 256 токенов; merger (2×2 pooling) режет "
        "их до 64. Поэтому в реальных VLM столько внимания уделяют разрешению и "
        "«сжатию» патчей перед подачей в LLM."
    )
    note(
        "Во vllm-mlx: --mllm, multimodal_processor.py, mllm_scheduler.py, "
        "--mllm-prefill-step-size, --max-audio-upload-mb (для аудио)."
    )

    section("Вывод")
    note("Картинка = N векторов-токенов, дальше обычный prefill/decode.")
    note("Разрешение и patch size определяют цену; merger сокращает токены.")
    note("Мультимодальность — надстройка над тем же инференс-конвейером.")


if __name__ == "__main__":
    main()
