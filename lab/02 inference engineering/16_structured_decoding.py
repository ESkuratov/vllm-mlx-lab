# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 16: structured / constrained decoding.

Идея
----
Обычная генерация свободна: модель может выдать что угодно. Constrained
decoding вмешивается в логиты и разрешает только те токены, которые продолжают
допустимый по грамматике текст. Так получают гарантированно валидный JSON,
строгий enum, числа и т.п.

Технически это «logits processor»: функция, которая берёт текущие логиты и
возвращает изменённые (например, запрещённым токенам ставит -inf). Во vllm-mlx
это `response_format` / constrained decoding в `server.py`.

Что показываем:
  1. что выдаёт модель без ограничений;
  2. как logits processor ограничивает ответ конечным языком (enum);
  3. как выглядит маскирование под «схему» (набор допустимых JSON);
  4. мягкий logits processor (штраф за повторы) для контраста.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/16_structured_decoding.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import numpy as np
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from common import MODEL_ID, load_model, note, section, subsection

_END = object()  # маркер конца допустимой строки в trie


def build_trie(strings, tok) -> dict:
    """Префиксное дерево из token-id всех допустимых строк."""
    root: dict = {}
    for s in strings:
        node = root
        for tid in tok.encode(s):
            node = node.setdefault(tid, {})
        node[_END] = True
    return root


def _nodes_after(root: dict, ids: list[int]) -> list[dict]:
    nodes = [root]
    for t in ids:
        nxt = [nd[t] for nd in nodes if t in nd]
        if not nxt:
            return []
        nodes = nxt
    return nodes


class ConstrainedProcessor:
    """Разрешает только токены, продолжающие одну из строк конечного языка."""

    def __init__(self, root, tok, prompt_len: int, eos_id: int, vocab: int):
        self.root = root
        self.prompt_len = prompt_len
        self.eos_id = eos_id
        self.vocab = vocab
        self.blocked_ever = 0

    def __call__(self, tokens, logits):
        gen = [int(t) for t in np.array(tokens).tolist()[self.prompt_len :]]
        nodes = _nodes_after(self.root, gen)

        allowed = set()
        terminal = False
        for nd in nodes:
            for k in nd:
                if k is not _END:
                    allowed.add(k)
            if _END in nd:
                terminal = True
        if terminal or not nodes:
            allowed.add(self.eos_id)
        if not allowed:
            allowed = {self.eos_id}

        self.blocked_ever += self.vocab - len(allowed)
        mask = np.full(self.vocab, -np.inf, dtype=np.float32)
        mask[list(allowed)] = 0.0
        return logits + mx.array(mask, dtype=logits.dtype)[None, :]


def make_repetition_penalty(penalty: float = 1.15):
    """Мягкий процессор: штрафует уже встречавшиеся токены."""

    def proc(tokens, logits):
        seen = set(int(t) for t in np.array(tokens).tolist())
        if not seen:
            return logits
        row = logits[0]
        idx = mx.array(sorted(seen))
        vals = row[idx]
        row = row.at[idx].add(mx.where(vals > 0, -vals * penalty, -vals / penalty))
        return row[None, :]

    return proc


def gen(model, prompt, tok, max_tokens, processors=None, sampler=None):
    eos = set(getattr(tok, "eos_token_ids", {tok.eos_token_id}))
    out = []
    for t, _ in generate_step(
        mx.array(prompt), model, max_tokens=max_tokens,
        sampler=sampler or make_sampler(temp=0.0),
        logits_processors=processors,
    ):
        t = int(t)
        if t in eos:
            break
        out.append(t)
    return tok.decode(out)


def main() -> None:
    section("16. Structured / constrained decoding")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()
    vocab = model.args.vocab_size
    eos_id = tok.eos_token_id if isinstance(tok.eos_token_id, int) else list(tok.eos_token_id)[0]

    # ----------------------------------------------------- без ограничений
    question = tok.encode("Is 2+2=5? Answer yes or no:")
    subsection("A. Обычная генерация (что получится)")
    free = gen(model, question, tok, 24)
    print(f"  {free!r}")
    note("Модель может отказаться отвечать ровно одним словом — её никто не ограничивал.")

    # ------------------------------------------------------------- enum
    subsection("B. Constrained: ответ из конечного набора {yes, no}")
    targets = [" yes", " no"]
    root = build_trie(targets, tok)
    proc = ConstrainedProcessor(root, tok, len(question), eos_id, vocab)
    constrained = gen(model, question, tok, 8, processors=[proc])
    print(f"  допустимые: {targets}")
    print(f"  ответ: {constrained!r}")
    note("Что бы модель ни «хотела», на выходе гарантированно один из двух вариантов.")

    # ------------------------------------------------------------- JSON
    subsection("C. Constrained под «схему»: конечный набор JSON")
    jsons = [
        '{"answer": "yes", "confidence": "high"}',
        '{"answer": "yes", "confidence": "low"}',
        '{"answer": "no", "confidence": "high"}',
        '{"answer": "no", "confidence": "low"}',
    ]
    jroot = build_trie(jsons, tok)
    jproc = ConstrainedProcessor(jroot, tok, len(question), eos_id, vocab)
    jout = gen(model, question, tok, 40, processors=[jproc])
    print(f"  вывод: {jout!r}")
    try:
        import json

        print(f"  json.loads → {json.loads(jout)}")
        note("Ответ парсится как JSON всегда — схема гарантирует валидность.")
    except Exception as exc:
        print(f"  json.loads упал: {exc}")

    # ------------------------------------------------------ мягкий штраф
    subsection("D. Мягкий logits processor: штраф за повтор")
    loop_prompt = tok.encode("Repeat after me: ha")
    plain = gen(model, loop_prompt, tok, 16)
    pen = gen(model, loop_prompt, tok, 16, processors=[make_repetition_penalty(1.2)])
    print(f"  без штрафа: {plain!r}")
    print(f"  со штрафом: {pen!r}")
    note(
        "Маскирование — это жёсткий вариант (нельзя вообще), а temperature/top-p/"
        "repetition penalty — мягкий (можно, но невыгодно)."
    )

    section("Вывод")
    note("Logits processor = точка вмешательства в выбор токена.")
    note("Constrained decoding делает вывод структурно валидным by construction.")
    note("Во vllm-mlx: response_format, грамматики; сэмплинг — --default-top-p и др.")


if __name__ == "__main__":
    main()
