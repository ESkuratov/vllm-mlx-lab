# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 21: speculative decoding с нуля.

Идея
----
Decode memory-bound: чтобы выдать один токен, target-модель читает все веса.
Speculative decoding использует маленькую draft-модель:

1. draft предлагает k токенов жадно;
2. target проверяет все k одним forward (как prefill);
3. принимаем совпавший префикс, на первом расхождении берём токен target;
4. откатываем KV-кэш обеих моделей до принятого префикса.

Если draft угадывает часто, за один forward target получаем несколько
токенов. Если нет — платим за draft и проверку, но не теряем корректность:
при greedy-декодировании результат обязан совпасть с обычной генерацией.

В лаборатории мы пишем и draft-цикл, и верификацию, и `rewind` кэша руками.
`mlx_lm.generate.speculative_generate_step` не используется — он есть в
inference/07, а здесь важно увидеть механику.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/21_speculative_from_scratch.py"

Draft-модель задаётся через LAB_DRAFT (по умолчанию 0.5B Qwen2.5-Coder).
"""
from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache

from common import MODEL_ID, load_model, note, section, subsection

DRAFT_ID = os.environ.get(
    "LAB_DRAFT", "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
)
PROMPT = os.environ.get(
    "LAB_PROMPT",
    "Explain what a KV cache is in one sentence. "
    "Then explain why paged attention matters for serving.",
)
MAX_TOKENS = 32
PROPOSAL_LEN = 4


# --------------------------------------------------------------------------- #
# Кэш и одиночные forward
# --------------------------------------------------------------------------- #


def _array(token: int) -> mx.array:
    return mx.array([[token]], dtype=mx.uint32)


def _prefill(model, ids: mx.array):
    cache = make_prompt_cache(model)
    logits = model(ids[None], cache=cache)
    mx.eval(logits)
    return cache, logits[0, -1]


def _forward_one(model, token: int, cache):
    logits = model(_array(token), cache=cache)
    mx.eval(logits)
    return logits[0, -1]


def _argmax(logits: mx.array) -> int:
    return int(mx.argmax(logits))


def _rewind(cache, n: int) -> None:
    """Убрать последние n токенов из всех слоёв KV-кэша."""
    if n <= 0:
        return
    for layer in cache:
        if not layer.is_trimmable():
            raise RuntimeError(
                f"слой {type(layer).__name__} не поддерживает trim/rewind"
            )
        trimmed = layer.trim(n)
        if trimmed != n:
            raise RuntimeError(f"trim вернул {trimmed}, ожидалось {n}")


def _cache_offset(cache) -> int:
    offsets = [getattr(layer, "offset", 0) for layer in cache]
    return int(max(offsets)) if offsets else 0


# --------------------------------------------------------------------------- #
# Baselines
# --------------------------------------------------------------------------- #


def greedy_baseline(model, ids: mx.array, max_tokens: int, eos_id: int | None):
    """Обычный greedy-декод target-моделью."""
    t0 = time.perf_counter()
    cache, logits = _prefill(model, ids)
    tokens: list[int] = []
    for _ in range(max_tokens):
        token = _argmax(logits)
        tokens.append(token)
        if token == eos_id:
            break
        logits = _forward_one(model, token, cache)
    mx.synchronize()
    elapsed = time.perf_counter() - t0
    return tokens, elapsed


# --------------------------------------------------------------------------- #
# Speculative decoding
# --------------------------------------------------------------------------- #


def speculative_generate(
    target,
    draft,
    ids: mx.array,
    max_tokens: int,
    proposal_len: int,
    target_eos: int | None,
    draft_eos: int | None,
):
    t0 = time.perf_counter()
    target_cache, target_logits = _prefill(target, ids)
    draft_cache, draft_logits = _prefill(draft, ids)

    committed: list[int] = []
    proposed = 0
    accepted = 0
    rounds = 0
    corrections = 0
    trace: list[tuple[int, int, bool]] = []

    # Первый токен target выбирает сам; дальше кэши обеих моделей содержат
    # ровно committed, а logits предсказывают следующий токен.
    first = _argmax(target_logits)
    committed.append(first)
    if first != target_eos and max_tokens > 1:
        target_logits = _forward_one(target, first, target_cache)
        draft_logits = _forward_one(draft, first, draft_cache)

    while len(committed) < max_tokens and committed[-1] != target_eos:
        remaining = max_tokens - len(committed)
        k = min(proposal_len, remaining)
        if k <= 0:
            break

        # 1. Draft предлагает до k токенов.
        draft_tokens: list[int] = []
        for _ in range(k):
            token = _argmax(draft_logits)
            draft_tokens.append(token)
            draft_logits = _forward_one(draft, token, draft_cache)
            if token == draft_eos:
                break
        if not draft_tokens:
            break
        proposed += len(draft_tokens)
        rounds += 1

        # 2. Target проверяет весь префикс одним forward.
        verify = target(mx.array([draft_tokens], dtype=mx.uint32), cache=target_cache)
        mx.eval(verify)

        # 3. Принимаем совпавший префикс.
        accepted_here = 0
        corrected: int | None = None
        for i, token in enumerate(draft_tokens):
            expected_logits = target_logits if i == 0 else verify[0, i - 1]
            expected = _argmax(expected_logits)
            if token == expected:
                accepted_here += 1
            else:
                corrected = expected
                break

        if corrected is None:
            # Все draft-токены приняты.
            committed.extend(draft_tokens)
            accepted += accepted_here
            target_logits = verify[0, -1]
            trace.append((len(draft_tokens), accepted_here, False))
            # draft_logits уже указывает на следующий токен после последнего draft.
            continue

        # 4. Расхождение: принимаем префикс + исправленный токен target.
        corrections += 1
        committed.extend(draft_tokens[:accepted_here])
        committed.append(corrected)
        accepted += accepted_here
        trace.append((len(draft_tokens), accepted_here, True))

        rejected = len(draft_tokens) - accepted_here
        _rewind(target_cache, rejected)
        _rewind(draft_cache, rejected)
        target_logits = _forward_one(target, corrected, target_cache)
        draft_logits = _forward_one(draft, corrected, draft_cache)

    mx.synchronize()
    elapsed = time.perf_counter() - t0
    stats = {
        "proposed": proposed,
        "accepted": accepted,
        "rounds": rounds,
        "corrections": corrections,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "elapsed": elapsed,
        "trace": trace,
    }
    return committed, stats


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def _vocab_matches(target_tok, draft_tok) -> bool:
    try:
        return target_tok.get_vocab() == draft_tok.get_vocab()
    except AttributeError:
        return len(target_tok) == len(draft_tok)


def main() -> None:
    section("21. Speculative decoding с нуля")
    print(f"  target: {MODEL_ID}")
    print(f"  draft : {DRAFT_ID}")

    target, target_tok = load_model(MODEL_ID)
    draft, draft_tok = load_model(DRAFT_ID)

    if target_tok.eos_token_id != draft_tok.eos_token_id:
        raise SystemExit(
            "EOS target и draft не совпадают — нужны модели с одним токенизатором"
        )
    if not _vocab_matches(target_tok, draft_tok):
        raise SystemExit("словари target и draft не совпадают")

    target_ids = target_tok.encode(PROMPT, add_special_tokens=False)
    draft_ids = draft_tok.encode(PROMPT, add_special_tokens=False)
    if target_ids != draft_ids:
        raise SystemExit(
            "tokenizers расходятся на промпте — используйте одну модель-семейство"
        )
    ids = mx.array(target_ids, dtype=mx.uint32)
    print(f"  prompt: {len(target_ids)} токенов: {PROMPT!r}")

    # Прогрев: первый вызов компилирует ядра и прогревает GPU.
    greedy_baseline(target, ids, 4, target_tok.eos_token_id)
    speculative_generate(
        target, draft, ids, 4, 2, target_tok.eos_token_id, draft_tok.eos_token_id
    )

    subsection("A. Обычный greedy-декод target")
    baseline_runs = [
        greedy_baseline(target, ids, MAX_TOKENS, target_tok.eos_token_id)
        for _ in range(2)
    ]
    baseline_tokens = baseline_runs[-1][0]
    t_baseline = statistics.median(t for _, t in baseline_runs)
    print(
        f"  токенов: {len(baseline_tokens)}, время: {t_baseline:.3f} с "
        f"(медиана 2 прогонов), {len(baseline_tokens) / t_baseline:.0f} tok/s"
    )

    subsection("B. Speculative decoding")
    spec_runs = [
        speculative_generate(
            target,
            draft,
            ids,
            MAX_TOKENS,
            PROPOSAL_LEN,
            target_tok.eos_token_id,
            draft_tok.eos_token_id,
        )
        for _ in range(2)
    ]
    spec_tokens, stats = spec_runs[-1]
    t_spec = statistics.median(s["elapsed"] for _, s in spec_runs)
    print(f"  раундов: {stats['rounds']}, предложено: {stats['proposed']}")
    print(
        f"  принято: {stats['accepted']} "
        f"({stats['acceptance_rate']:.1%}), исправлений: {stats['corrections']}"
    )
    print(
        f"  токенов: {len(spec_tokens)}, время: {t_spec:.3f} с "
        f"(медиана 2 прогонов), {len(spec_tokens) / t_spec:.0f} tok/s"
    )
    print(f"  ускорение vs greedy: {t_baseline / t_spec:.2f}x")
    print("  первые раунды (draft, принято, было исправление):")
    for i, (proposed, accepted, corrected) in enumerate(stats["trace"][:8]):
        print(f"    {i}: k={proposed}, accepted={accepted}, corrected={corrected}")

    subsection("C. Корректность")
    if spec_tokens == baseline_tokens:
        print("  ✓ speculative выдал ровно те же токены, что greedy target")
    else:
        for i, (a, b) in enumerate(zip(spec_tokens, baseline_tokens)):
            if a != b:
                print(f"  ✗ первое расхождение на позиции {i}: {a} != {b}")
                break
        raise SystemExit("speculative decoding изменил результат — это баг")

    subsection("D. Sweep proposal_len")
    print(
        f"  {'k':>3} {'acceptance':>11} {'corrections':>12} "
        f"{'time, s':>9} {'speedup':>9}"
    )
    for k in (1, 2, 4, 8):
        tokens, sweep = speculative_generate(
            target,
            draft,
            ids,
            MAX_TOKENS,
            k,
            target_tok.eos_token_id,
            draft_tok.eos_token_id,
        )
        if tokens != baseline_tokens:
            raise SystemExit(f"k={k}: результат разошёлся с greedy target")
        print(
            f"  {k:>3} {sweep['acceptance_rate']:>10.1%} "
            f"{sweep['corrections']:>12} {sweep['elapsed']:>9.3f} "
            f"{t_baseline / sweep['elapsed']:>8.2f}x"
        )
    note(
        "Большой k амортизирует один forward target, но увеличивает цену "
        "draft и вероятность расхождения. Оптимум зависит от acceptance "
        "rate, размера draft и длины ответа."
    )
    note(
        "На этой машине и коротком ответе наш Python-цикл часто проигрывает "
        "greedy: draft-шаги и верификация стоят дороже, чем экономят. "
        "В production draft-шаги батчируют, а target/draft разница шире."
    )

    note(
        "Greedy speculative decoding обязан быть эквивалентен обычному greedy. "
        "Ускорение зависит от acceptance rate и цены draft."
    )
    note(
        "В vllm-mlx/mlx-lm этот же механизм используется с MTP/draft-моделью "
        "и откатом кэша; inference/07 показывает готовый speculative_generate_step."
    )

    section("Вывод")
    note("Draft предлагает k токенов, target проверяет их одним forward.")
    note("Принимаем совпавший префикс, на расхождении берём токен target.")
    note("Rewind KV-кэша — обязательная часть, иначе кэш разъедется с текстом.")
    note("Корректность проверяется сравнением с обычным greedy-декодом.")


if __name__ == "__main__":
    main()
