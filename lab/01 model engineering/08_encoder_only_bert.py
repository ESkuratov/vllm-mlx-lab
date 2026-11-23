# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 08: encoder-only (BERT) — двунаправленное внимание.

Идея
----
BERT — это стек transformer-слоёв без causal-маски: токен видит и левый, и
правый контекст. За счёт этого он решает masked language modeling (MLM):
по предложению с пропуском предсказывает пропущенное слово.

Собираем маленький BERT с нуля на MLX и на игрушечной задаче показываем, зачем
нужна двунаправленность:

    [CLS] start  [MASK]  apple  end [SEP]
                  ↑
        цвет определяется существительным СПРАВА

Causal-модель (как decoder-only LLM) видит только «start» и не может угадать
цвет. Bidirectional-модель видит «apple» и предсказывает «red».

Что показываем:
  1. устройство BERT: эмбеддинги, post-LN блоки, MLM-голова;
  2. обучение двух одинаковых моделей — causal и bidirectional;
  3. карту внимания: куда смотрит [MASK] в обоих случаях;
  4. чем encoder-only отличается от decoder-only LLM.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/08_encoder_only_bert.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from common import note, section, subsection

# --------------------------------------------------------------------------- #
# Игрушечный корпус: цвет однозначно задаётся существительным.
# --------------------------------------------------------------------------- #

COLORS = ["red", "blue", "green", "yellow", "black", "white", "pink", "brown"]
NOUNS = ["apple", "sky", "grass", "sun", "night", "snow", "rose", "wood"]

SPECIALS = ["[PAD]", "[CLS]", "[SEP]", "[MASK]", "[UNK]"]
VOCAB = SPECIALS + ["start", "end"] + COLORS + NOUNS
STOI = {token: i for i, token in enumerate(VOCAB)}

DIM = 64
HEADS = 4
FFN = 256
LAYERS = 2
MAX_POS = 16

CLS_POS = 0
COLOR_POS = 2
NOUN_POS = 3
SEQ_LEN = 6


# --------------------------------------------------------------------------- #
# BERT: эмбеддинги, слой, MLM-голова
# --------------------------------------------------------------------------- #


class BertEmbeddings(nn.Module):
    """Word + learned position + token type, затем LayerNorm."""

    def __init__(self, vocab: int, dim: int, max_pos: int):
        super().__init__()
        self.word = nn.Embedding(vocab, dim)
        self.position = nn.Embedding(max_pos, dim)
        self.token_type = nn.Embedding(2, dim)
        self.norm = nn.LayerNorm(dim, eps=1e-12)

    def __call__(self, ids: mx.array) -> mx.array:
        L = ids.shape[1]
        positions = mx.arange(L)[None, :]
        token_types = mx.zeros_like(ids)
        x = self.word(ids) + self.position(positions) + self.token_type(token_types)
        return self.norm(x)


class BertLayer(nn.Module):
    """Классический post-LN слой BERT: Add&Norm после каждого подслоя."""

    def __init__(self, dim: int, heads: int, ffn: int):
        super().__init__()
        self.attn = nn.MultiHeadAttention(dim, heads, bias=True)
        self.ln1 = nn.LayerNorm(dim, eps=1e-12)
        self.ln2 = nn.LayerNorm(dim, eps=1e-12)
        self.fc1 = nn.Linear(dim, ffn)
        self.fc2 = nn.Linear(ffn, dim)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = self.ln1(x + self.attn(x, x, x, mask))
        x = self.ln2(x + self.fc2(nn.gelu(self.fc1(x))))
        return x


class TinyBertMLM(nn.Module):
    """Encoder-only BERT + MLM-голова с весами, связанными с эмбеддингами."""

    def __init__(self, vocab: int, dim: int, layers: int, heads: int, ffn: int,
                 max_pos: int, causal: bool):
        super().__init__()
        self.embeddings = BertEmbeddings(vocab, dim, max_pos)
        self.layers = [BertLayer(dim, heads, ffn) for _ in range(layers)]
        self.mlm_dense = nn.Linear(dim, dim)
        self.mlm_norm = nn.LayerNorm(dim, eps=1e-12)
        self.decoder_bias = mx.zeros((vocab,))
        self.causal = causal

    def __call__(self, ids: mx.array) -> mx.array:
        L = ids.shape[1]
        mask = (
            nn.MultiHeadAttention.create_additive_causal_mask(L)
            if self.causal
            else None
        )

        x = self.embeddings(ids)
        for layer in self.layers:
            x = layer(x, mask)

        h = self.mlm_norm(nn.gelu(self.mlm_dense(x)))
        return self.embeddings.word.as_linear(h) + self.decoder_bias


# --------------------------------------------------------------------------- #
# Данные и обучение
# --------------------------------------------------------------------------- #


def make_batch() -> tuple[mx.array, mx.array, mx.array]:
    """Два примера на предложение: маскируем цвет и существительное."""
    rows, labels, positions = [], [], []
    for color, noun in zip(COLORS, NOUNS):
        base = ["[CLS]", "start", color, noun, "end", "[SEP]"]
        for pos, target in ((COLOR_POS, color), (NOUN_POS, noun)):
            tokens = base.copy()
            tokens[pos] = "[MASK]"
            rows.append([STOI[t] for t in tokens])
            labels.append(STOI[target])
            positions.append(pos)
    return mx.array(rows), mx.array(labels), mx.array(positions)


def loss_fn(model, ids, labels, positions):
    logits = model(ids)
    rows = mx.arange(ids.shape[0])
    logits = logits[rows, positions, :]
    return nn.losses.cross_entropy(logits, labels, reduction="mean")


def accuracy(model, ids, labels, positions) -> tuple[float, float]:
    logits = model(ids)
    rows = mx.arange(ids.shape[0])
    pred = mx.argmax(logits[rows, positions, :], axis=-1)
    ok = (pred == labels).tolist()
    pos = positions.tolist()

    def acc_at(target_pos: int) -> float:
        idx = [i for i, p in enumerate(pos) if p == target_pos]
        return sum(ok[i] for i in idx) / len(idx)

    return acc_at(COLOR_POS), acc_at(NOUN_POS)


def train(causal: bool, *, steps: int = 800, lr: float = 1e-3,
          seed: int = 0) -> tuple[TinyBertMLM, list[float], float, float]:
    mx.random.seed(seed)
    model = TinyBertMLM(len(VOCAB), DIM, LAYERS, HEADS, FFN, MAX_POS, causal)
    ids, labels, positions = make_batch()

    optimizer = optim.Adam(learning_rate=lr)
    loss_and_grad = nn.value_and_grad(model, loss_fn)
    losses: list[float] = []

    for step in range(steps):
        loss, grads = loss_and_grad(model, ids, labels, positions)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        losses.append(float(loss))
        if step % 200 == 0:
            print(f"    шаг {step:>4}: loss = {float(loss):.4f}")

    acc_color, acc_noun = accuracy(model, ids, labels, positions)
    return model, losses, acc_color, acc_noun


# --------------------------------------------------------------------------- #
# Визуализация внимания
# --------------------------------------------------------------------------- #


def attention_to_noun(model: TinyBertMLM, ids: mx.array) -> mx.array:
    """Веса внимания [MASK]-позиции в первом слое: (heads, seq)."""
    x = model.embeddings(ids)
    layer = model.layers[0]

    q = layer.attn.query_proj(x)
    k = layer.attn.key_proj(x)
    B, L, D = x.shape
    H = layer.attn.num_heads
    head_dim = D // H

    q = q.reshape(B, L, H, head_dim).transpose(0, 2, 1, 3)
    k = k.reshape(B, L, H, head_dim).transpose(0, 2, 1, 3)
    scores = (q @ k.transpose(0, 1, 3, 2)) / mx.sqrt(head_dim)

    if model.causal:
        scores = scores + nn.MultiHeadAttention.create_additive_causal_mask(L)

    probs = mx.softmax(scores, axis=-1)
    return probs[0, :, COLOR_POS, :]  # (heads, seq)


# --------------------------------------------------------------------------- #
# Эксперименты
# --------------------------------------------------------------------------- #


def main() -> None:
    section("08. Encoder-only (BERT): двунаправленное внимание")

    # ------------------------------------------------ A. архитектура
    subsection("A. Устройство BERT")
    print("  [CLS] start  red  apple  end [SEP]")
    print("    │")
    print("    ├─ word + positional + token-type embeddings → LayerNorm")
    print(f"    ├─ {LAYERS} × [ bidirectional MHA → Add&Norm → FFN(GELU) → Add&Norm ]")
    print("    └─ MLM head: Linear → GELU → LayerNorm → tied decoder")
    print()
    print(f"  hidden={DIM}, heads={HEADS}, layers={LAYERS}, ffn={FFN}, "
          f"vocab={len(VOCAB)}")
    model = TinyBertMLM(len(VOCAB), DIM, LAYERS, HEADS, FFN, MAX_POS, causal=False)
    total = sum(a.size for _, a in tree_flatten(model.parameters()))
    print(f"  параметров: {total:,}")
    note("Bias в проекциях и FFN, LayerNorm, GELU — как в оригинальном BERT.")
    note("MLM-голова делит веса с входными эмбеддингами (tied embeddings).")

    # ------------------------------------------------ B. causal vs bidirectional
    subsection("B. Одна задача, два типа внимания")
    print("  Пример:  [CLS] start  [MASK]  apple  end [SEP]")
    print("  Цель:    red       ← цвет задаётся существительным справа")
    print("  Causal-модель видит только [CLS] start [MASK] и не знает ответ.")
    print()

    print("  Bidirectional BERT:")
    bidir, bidir_losses, b_color, b_noun = train(causal=False)
    print("  Causal-модель (как decoder-only LLM):")
    causal, causal_losses, c_color, c_noun = train(causal=True)

    print()
    print(f"  {'модель':<18}{'loss':>10}{'цвет (правый контекст)':>26}"
          f"{'существительное (левый)':>27}")
    print(f"  {'bidirectional':<18}{bidir_losses[-1]:>10.4f}"
          f"{b_color:>25.1%}{b_noun:>26.1%}")
    print(f"  {'causal':<18}{causal_losses[-1]:>10.4f}"
          f"{c_color:>25.1%}{c_noun:>26.1%}")
    note(
        f"Случайное угадывание цвета — 1/{len(COLORS)} = "
        f"{1 / len(COLORS):.1%}. Causal-модель именно к этому и приходит."
    )
    note(
        "Зато существительное causal-модель предсказывает идеально: цвет стоит "
        "слева. Дело не в ёмкости, а в направлении видимости."
    )
    note(
        "Loss causal-модели ≈ (0 + ln 8) / 2 ≈ 1.04 — среднее между идеальным "
        "левый-контекст и равномерным правый."
    )

    # ------------------------------------------------ C. карта внимания
    subsection("C. Куда смотрит [MASK] в первом слое")
    ids, _, _ = make_batch()
    sample = ids[0:1]  # цвет замаскирован, существительное — apple
    tokens = ["[CLS]", "start", "[MASK]", "apple", "end", "[SEP]"]

    bidir_attn = attention_to_noun(bidir, sample)   # (heads, seq)
    causal_attn = attention_to_noun(causal, sample)
    bidir_mean = bidir_attn.mean(axis=0).tolist()
    causal_mean = causal_attn.mean(axis=0).tolist()

    print(f"  среднее по {HEADS} головам первого слоя:")
    print(f"  {'позиция':<8}{'bidir':>8}  {'':<20}{'causal':>8}")
    for i, token in enumerate(tokens):
        bar_b = "#" * round(bidir_mean[i] * 20)
        bar_c = "#" * round(causal_mean[i] * 20)
        print(
            f"  {token:<8}{bidir_mean[i]:>8.3f}  {bar_b:<20}"
            f"{causal_mean[i]:>8.3f}  {bar_c}"
        )
    b_noun = float(bidir_attn[:, NOUN_POS].mean())
    c_noun = float(causal_attn[:, NOUN_POS].mean())
    print(f"\n  внимание к 'apple': bidirectional {b_noun:.2f}, causal {c_noun:.2f}")
    note(
        "BERT смотрит на существительное справа — именно оттуда берётся цвет. "
        "Causal-маска физически не даёт это сделать: её масса остаётся слева."
    )

    # ------------------------------------------------ D. сравнение
    subsection("D. Encoder-only vs decoder-only")
    print(f"  {'':<24}{'BERT (encoder-only)':>24}{'LLM (decoder-only)':>24}")
    print(f"  {'маска внимания':<24}{'двунаправленная':>24}{'causal':>24}")
    print(f"  {'задача обучения':<24}{'MLM / контраст':>24}{'next-token':>24}")
    print(f"  {'позиции':<24}{'обучаемые absolute':>24}{'RoPE / absolute':>24}")
    print(f"  {'нормализация':<24}{'post-LN (классика)':>24}{'pre-LN / RMSNorm':>24}")
    print(f"  {'голова':<24}{'MLM / pooler':>24}{'lm_head':>24}")
    print(f"  {'типичное применение':<24}{'поиск, классификация':>24}{'генерация':>24}")

    note(
        "BERT не генерирует текст авторегрессионно: он один раз кодирует весь "
        "вход. Поэтому эмбеддинги и rerank (`model/05`) — его естественная ниша."
    )
    note(
        "Настоящий BERT можно загрузить через mlx_embeddings: "
        "`from mlx_embeddings import load` (модели семейства BERT/RoBERTa)."
    )

    section("Вывод")
    note("BERT = encoder-only transformer без causal-маски.")
    note("Двунаправленность даёт правый контекст — критичный для MLM.")
    note("Causal-модель не «глупее»: она просто не видит будущие токены.")
    note("MLM-голова + tied embeddings — дешёвый способ учить представления.")


if __name__ == "__main__":
    main()
