# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 07: полный transformer block — собираем с нуля.

Идея
----
Современный decoder-only слой — это не «attention + MLP», а конкретная
конструкция:

    x ──► RMSNorm ──► causal attention (RoPE, GQA) ──► + x
      ──► RMSNorm ──► SwiGLU FFN                    ──► + x

Здесь собираем такой блок руками на MLX: проверяем формы, считаем параметры,
смотрим на residual stream и сравниваем с классическим
`nn.TransformerEncoderLayer` и с реальным слоем Qwen2.5.

Что показываем:
  1. из чего состоит блок и сколько параметров в каждой части;
  2. как течёт residual stream и какой вклад дают attention и MLP;
  3. чем современный блок отличается от «канонического» transformer-слоя;
  4. где параметры живут в реальной модели (Qwen2.5-3B).

Запуск:
    source ./vllm-env/bin/activate
    python "lab/01 model engineering/07_full_transformer_block.py"
"""
from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from common import gb, load_model, model_nbytes, note, section, subsection

# Небольшая конфигурация, чтобы всё считалось мгновенно.
DIM = 256
HEADS = 8
KV_HEADS = 2          # GQA: ключей/значений в 4 раза меньше, чем query
HEAD_DIM = DIM // HEADS
FFN_DIM = 688         # ~8/3 * DIM — типичный размер для SwiGLU
VOCAB = 512
LAYERS = 4


# --------------------------------------------------------------------------- #
# Слои блока
# --------------------------------------------------------------------------- #


class RMSNorm(nn.Module):
    """Нормализация по RMS без вычитания среднего (как в LLaMA/Qwen)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = mx.ones((dim,))
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + self.eps)
        return x * rms * self.weight


class CausalAttention(nn.Module):
    """Multi-head attention с RoPE и GQA."""

    def __init__(self, dim: int = DIM, heads: int = HEADS, kv_heads: int = KV_HEADS):
        super().__init__()
        self.heads = heads
        self.kv_heads = kv_heads
        self.head_dim = dim // heads

        self.q_proj = nn.Linear(dim, heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(dim, kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(heads * self.head_dim, dim, bias=False)
        self.rope = nn.RoPE(self.head_dim, base=10000)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        B, L, _ = x.shape

        q = self.q_proj(x).reshape(B, L, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.kv_heads, self.head_dim)

        # (B, L, H, D) → (B, H, L, D): головы становятся отдельной осью.
        q = self.rope(q.transpose(0, 2, 1, 3))
        k = self.rope(k.transpose(0, 2, 1, 3))
        v = v.transpose(0, 2, 1, 3)

        if mask is None:
            mask = nn.MultiHeadAttention.create_additive_causal_mask(L)

        # SDPA сам «размножает» KV-головы до числа query-голов (GQA).
        out = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.head_dim**-0.5, mask=mask
        )
        return self.o_proj(out.transpose(0, 2, 1, 3).reshape(B, L, -1))


class SwiGLU(nn.Module):
    """FFN с gated-активацией: down(silu(gate(x)) * up(x))."""

    def __init__(self, dim: int = DIM, hidden: int = FFN_DIM):
        super().__init__()
        self.gate = nn.Linear(dim, hidden, bias=False)
        self.up = nn.Linear(dim, hidden, bias=False)
        self.down = nn.Linear(hidden, dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down(nn.silu(self.gate(x)) * self.up(x))


class DecoderBlock(nn.Module):
    """Pre-norm residual block: x + attn(norm(x)), затем x + mlp(norm(x))."""

    def __init__(self, dim: int = DIM, heads: int = HEADS, kv_heads: int = KV_HEADS,
                 ffn: int = FFN_DIM):
        super().__init__()
        self.ln1 = RMSNorm(dim)
        self.attn = CausalAttention(dim, heads, kv_heads)
        self.ln2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, ffn)

    def __call__(self, x: mx.array, mask: mx.array | None = None) -> mx.array:
        x = x + self.attn(self.ln1(x), mask)
        return x + self.mlp(self.ln2(x))


class TinyTransformer(nn.Module):
    """Полный decoder-only стек: embeddings → N блоков → norm → lm_head."""

    def __init__(self, vocab: int = VOCAB, dim: int = DIM, layers: int = LAYERS,
                 heads: int = HEADS, kv_heads: int = KV_HEADS, ffn: int = FFN_DIM):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        self.layers = [DecoderBlock(dim, heads, kv_heads, ffn) for _ in range(layers)]
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.dim = dim

    def __call__(self, ids: mx.array) -> mx.array:
        x = self.embed(ids)
        mask = nn.MultiHeadAttention.create_additive_causal_mask(ids.shape[1])
        for layer in self.layers:
            x = layer(x, mask)
        return self.lm_head(self.norm(x))


# --------------------------------------------------------------------------- #
# Утилиты
# --------------------------------------------------------------------------- #


def nparams(module: nn.Module) -> int:
    return sum(a.size for _, a in tree_flatten(module.parameters()))


def bench(fn, repeats: int = 7) -> float:
    mx.eval(fn())
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


# --------------------------------------------------------------------------- #
# Эксперименты
# --------------------------------------------------------------------------- #


def main() -> None:
    section("07. Полный transformer block: собираем с нуля")

    # ------------------------------------------------ A. формы и параметры
    subsection("A. Один блок: формы и распределение параметров")
    block = DecoderBlock()
    x = mx.random.normal((2, 16, DIM))
    y = block(x)
    print(f"  вход:  {x.shape}  (batch, seq, hidden)")
    print(f"  выход: {y.shape}")

    p_attn = nparams(block.attn)
    p_mlp = nparams(block.mlp)
    p_norm = nparams(block.ln1) + nparams(block.ln2)
    p_total = nparams(block)
    print(f"\n  {'часть':<14}{'параметров':>14}{'доля':>9}")
    for name, value in (("attention", p_attn), ("SwiGLU FFN", p_mlp), ("RMSNorm", p_norm)):
        print(f"  {name:<14}{value:>14,}{value / p_total:>8.1%}")
    print(f"  {'итого':<14}{p_total:>14,}{1.0:>8.0%}")
    note("Большая часть параметров блока — в FFN, а не во внимании.")
    note(f"GQA: {HEADS} query-голов и {KV_HEADS} KV-голов → KV-проекции в "
         f"{HEADS // KV_HEADS}× меньше.")

    # ------------------------------------------------ B. residual stream
    subsection("B. Residual stream: что добавляют attention и MLP")
    model = TinyTransformer()
    ids = mx.random.randint(0, VOCAB, (2, 16))
    mask = nn.MultiHeadAttention.create_additive_causal_mask(16)

    h = model.embed(ids)
    print(f"  {'слой':>5}{'‖x‖':>10}{'‖attn‖/‖x‖':>14}{'‖mlp‖/‖x_after_attn‖':>24}")
    for i, layer in enumerate(model.layers):
        attn_out = layer.attn(layer.ln1(h), mask)
        h_after_attn = h + attn_out
        mlp_out = layer.mlp(layer.ln2(h_after_attn))
        print(
            f"  {i:>5}{float(mx.linalg.norm(h)):>10.2f}"
            f"{float(mx.linalg.norm(attn_out) / mx.linalg.norm(h)):>14.2f}"
            f"{float(mx.linalg.norm(mlp_out) / mx.linalg.norm(h_after_attn)):>24.2f}"
        )
        h = h_after_attn + mlp_out
    note(
        "Pre-norm + residual: каждый подслой добавляет поправку к residual stream, "
        "а не переписывает его целиком. Поэтому глубокие стеки обучаются стабильно."
    )
    note(
        "На случайных весах норма residual stream растёт с глубиной; в обученной "
        "модели она стабилизируется, но соотношение вкладов сохраняется."
    )

    # ------------------------------------------------ C. классический слой
    subsection("C. Современный блок vs классический TransformerEncoderLayer")
    classic = nn.TransformerEncoderLayer(
        DIM, HEADS, mlp_dims=FFN_DIM, norm_first=True
    )
    print(f"  {'':<20}{'наш блок':>18}{'классический':>18}")
    print(f"  {'параметров':<20}{p_total:>18,}{nparams(classic):>18,}")
    print(f"  {'norm':<20}{'RMSNorm':>18}{'LayerNorm':>18}")
    print(f"  {'FFN':<20}{'SwiGLU (3 матрицы)':>18}{'ReLU (2 матрицы)':>18}")
    print(f"  {'позиции':<20}{'RoPE':>18}{'нет':>18}")
    print(f"  {'KV-голов':<20}{str(KV_HEADS):>18}{str(HEADS):>18}")

    print(f"\n  {'S':>6}{'наш блок':>12}{'классический':>16}")
    for seq in (16, 128, 512, 2048):
        x = mx.random.normal((1, seq, DIM))
        m = nn.MultiHeadAttention.create_additive_causal_mask(seq)
        t_ours = bench(lambda: block(x, m))
        t_classic = bench(lambda: classic(x, m))
        print(f"  {seq:>6}{t_ours * 1e3:>10.2f} ms{t_classic * 1e3:>14.2f} ms")
    note(
        "Время похоже: оба варианта упираются в матричные умножения. Разница — "
        "в устройстве: RMSNorm дешевле LayerNorm, RoPE не требует обучаемых "
        "позиционных эмбеддингов, GQA экономит память KV-кэша, SwiGLU даёт "
        "лучшее качество при том же числе параметров."
    )

    # ------------------------------------------------ D. реальная модель
    subsection("D. Реальный слой Qwen2.5: где живут параметры")
    model_real, _ = load_model()
    args = model_real.args
    head_dim = args.hidden_size // args.num_attention_heads

    attn = (
        args.hidden_size * args.hidden_size
        + 2 * args.hidden_size * (args.num_key_value_heads * head_dim)
        + args.hidden_size * args.hidden_size
    )
    mlp = 3 * args.hidden_size * args.intermediate_size
    per_block = attn + mlp
    embed = args.vocab_size * args.hidden_size
    lm_head = 0 if getattr(args, "tie_word_embeddings", False) else embed
    total = embed + args.num_hidden_layers * per_block + lm_head

    print(f"  модель: {args.model_type}, hidden={args.hidden_size}, "
          f"layers={args.num_hidden_layers}")
    print(f"  heads={args.num_attention_heads}, kv_heads={args.num_key_value_heads}, "
          f"intermediate={args.intermediate_size}")
    print(f"\n  {'часть':<18}{'параметров':>16}{'доля':>9}")
    print(f"  {'embedding':<18}{embed:>16,}{embed / total:>8.1%}")
    print(f"  {'один блок':<18}{per_block:>16,}{per_block / total:>8.1%}")
    print(f"  {'  attention':<18}{attn:>16,}{attn / per_block:>8.1%}")
    print(f"  {'  MLP':<18}{mlp:>16,}{mlp / per_block:>8.1%}")
    print(f"  {'все блоки':<18}{args.num_hidden_layers * per_block:>16,}"
          f"{args.num_hidden_layers * per_block / total:>8.1%}")
    print(f"  {'lm_head':<18}{lm_head:>16,}{lm_head / total:>8.1%}")
    print(f"  {'итого':<18}{total:>16,}{1.0:>8.0%}")

    actual = model_nbytes(model_real)
    print(f"\n  веса в памяти: {gb(actual)} "
          f"(4-bit квантизация, поэтому меньше {total * 2 / 1024**3:.2f} GB в fp16)")
    note("Внутри блока ~80% параметров — FFN; embedding — заметная доля модели.")
    note("Именно поэтому квантизация весов (`inference/10`) и MoE (`model/03`) дают "
         "такой большой выигрыш по памяти/вычислениям.")

    section("Вывод")
    note("Transformer block = pre-norm residual + attention + FFN.")
    note("Современный вариант: RMSNorm, RoPE, GQA, SwiGLU.")
    note("Residual stream — «магистраль», подслои лишь добавляют поправки.")
    note("Основные параметры и вычисления — в FFN, а не во внимании.")


if __name__ == "__main__":
    main()
