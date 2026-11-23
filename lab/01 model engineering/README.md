# model — модель и её сборка

Из чего состоит модель, как её собрать и как её считать быстро: decoder block,
внимание и позиции, FFN, MoE, мультимодальность, сборка чекпоинта — плюс свои
Metal-ядра (W4-матмулы, плиточное внимание, fused-операции).

Номера нумеруются внутри папки: `01`…`21`.

Запуск — из корня репозитория, как и остальные лабы:

```bash
source ./vllm-env/bin/activate
python "lab/01 model engineering/07_full_transformer_block.py"
```

Модель по умолчанию — `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`
(переопределяется `LAB_MODEL`). Лабы `13`, `20` и `21` грузят реальный чекпоинт.

## Эксперименты

### Устройство слоя

| Файл | О чём | Ключевой вывод |
| --- | --- | --- |
| `07_full_transformer_block.py` | полный decoder-block с нуля | pre-norm residual + RMSNorm + RoPE + GQA + SwiGLU; ~80% параметров блока — в FFN |
| `01_attention_scaling.py` | O(S²) и GQA | время attention растёт квадратично; GQA сокращает KV-кэш, но не время оператора |
| `02_rope_long_context.py` | RoPE и длинный контекст | внимание становится относительным; дальние позиции декогерируют |
| `08_encoder_only_bert.py` | encoder-only BERT и MLM | двунаправленное внимание использует правый контекст, causal-модель — нет |

### Варианты архитектур и выходов

| Файл | О чём | Ключевой вывод |
| --- | --- | --- |
| `03_moe.py` | MoE: роутинг, активные параметры | «скорость маленькой, параметры большой»; важен баланс экспертов |
| `05_embeddings_vs_generation.py` | эмбеддинги vs генерация | один forward против авторегрессии; дешёвый путь для поиска |
| `06_multimodal_prefill.py` | картинка как токены, vision-проекция | цена картинки = цена её токенов |
| `17_qwen3_moe.py` | MoE: router, top-k, grouped experts | grouped `gather_qmm` быстрее Python-цикла; активных параметров top_k/E |
| `19_qwen3_moe_model.py` | уменьшенная Qwen3-MoE модель | полный decoder block с router/top-k/grouped experts |

### Сборка реальной модели

| Файл | О чём | Ключевой вывод |
| --- | --- | --- |
| `13_load_qwen3.py` | сборка Qwen из checkpoint + `metal_ops` | top-5 logits совпадают с mlx_lm; packed W4 не разворачивается целиком |
| `20_selected_model.py` | единая Week 2 `selected`-модель | checkpoint-флаги и cumulative feature set на реальном Qwen |
| `21_real_qwen3_moe.py` | реальный Qwen3-MoE 30B-A3B | конфиг и `switch_mlp` подтверждают sparse MoE; тяжёлая, нужна память |

## Как устроен decoder block (07)

```
x ──► RMSNorm ──► causal attention ──► + x
  ──► RMSNorm ──► SwiGLU FFN       ──► + x
        │              │
        │              └─ gate/up/down, активация SiLU
        └─ Q/K/V-проекции, RoPE, GQA, SDPA
```

Лаба собирает этот блок руками, считает параметры по частям, показывает вклад
подслоёв в residual stream и сравнивает с `nn.TransformerEncoderLayer` и
реальным слоем Qwen2.5.

## Как устроен BERT (08)

```
[CLS] start [MASK] apple end [SEP]
  │
  ├─ word + position + token-type embeddings → LayerNorm
  ├─ N × [ bidirectional MHA → Add&Norm → FFN(GELU) → Add&Norm ]
  └─ MLM head: Linear → GELU → LayerNorm → tied decoder
```

На игрушечной задаче цвет определяется существительным **справа**. Обе модели
учатся на одних данных: bidirectional BERT предсказывает и цвет, и
существительное; causal-модель предсказывает существительное по левому
контексту, а цвет — на уровне случайного угадывания. Карта внимания показывает
причину: BERT смотрит на существительное справа, causal-маска оставляет всю
массу слева.

## Ядра и быстрый счёт на GPU

Лабы про то, как эти же операторы считаются руками на Metal. Ядра
компилируются в рантайме через `mx.fast.metal_kernel` — CMake и nanobind нужны
только для `14`. Здесь меряется не «быстрее ли MLX» (обычно нет), а **режим**:
matvec для decode против matmul для prefill, один query против плитки, split-K
по форме.

| Файл | О чём | Ключевой вывод |
| --- | --- | --- |
| `09_quantized_matvec_metal.py` | W4 matvec, распаковка веса в регистре | matvec — decode-путь; для prefill нужен tiled matmul с reuse весов |
| `10_fused_ops_metal.py` | fused RMSNorm, RoPE, SwiGLU | один проход вместо цепочки ops; на микро-бенче выигрыш непостоянен |
| `11_tiled_w4_matmul.py` | tiled W4 matmul с `simdgroup_matrix` | плитка весов читается один раз на 8 строк активаций; частичные плитки — нулями |
| `12_paged_flash_attention_metal.py` | page-walk + online-softmax в Metal | K/V читаются по `block_table`, без dense-реконструкции |
| `15_quantized_embedding_splitk.py` | quantized embedding + split-K | embedding распаковывает только выбранные строки; split-K — occupancy против reduction |
| `16_tiled_dense_prefill_attention.py` | tiled dense prefill attention | 8 query-строк на плитку, QK через `simdgroup_matrix`, P·V пока скалярный |
| `18_tiled_paged_flash_attention.py` | tiled paged FlashAttention | page-walk + MMA для QK **и** P·V |
| `14_mlx_extension_build.py` | CMake + nanobind + MLX extension | как упаковать C++/Metal в Python-модуль; без инструментов — чек-лист |
| `04_kernel_fusion.py` | `mx.compile`, слияние ядер | помогает длинным цепочкам мелких ops, на большом forward — почти нет |

Плюс `extensions/demo/` — минимальное настоящее расширение:
`CMakeLists.txt`, `bindings.cpp`, `build.py`, `demo_ext/`. Его собирает `14`:

```bash
python "lab/01 model engineering/14_mlx_extension_build.py" --build
```

Общее у этих лаб: один инструмент (`mx.fast.metal_kernel`, формы через
`template=[...]`, threadgroup memory для плиток), эталон рядом
(`dense_reference`/`mlx_reference` + `assert_close`) и честный вердикт против
MLX в каждом выводе. Лабы с проверкой `mx.metal.is_available()` на машине без
Metal печатают «лаборатория пропущена» и выходят.

Чего пока нет: плиток 32×32 и double-buffering, `P·V` через MMA в `16`,
fused-эпилога attention (RoPE внутри ядра), MoE-ядра и внимания над
квантизованным KV. Общие ядра RMSNorm/RoPE/SwiGLU лежат в `lab/metal_ops.py` —
их же использует `13` и `20`.

## Связь с остальными лабами

- KV-кэш и paged attention — `lab/02 inference engineering/02_kv_cache.py`,
  `lab/02 inference engineering/04_paged_attention.py`.
- Квантизация весов и KV — `lab/02 inference engineering/10_weight_quantization.py`,
  `lab/02 inference engineering/05_kv_quantization.py`.
- Ядра внимания на Metal — `16_tiled_dense_prefill_attention.py`,
  `12_paged_flash_attention_metal.py`.
- `mx.compile` — `04_kernel_fusion.py`.
- Эмбеддинги и rerank в сервисе — `05_embeddings_vs_generation.py`
  и `vllm_mlx/embedding.py`, `vllm_mlx/rerank.py`.

## Техническая деталь

Скрипты в этой папке добавляют `lab/` в `sys.path`, поэтому `from common import
...` работает при запуске из любой директории. Запускать всё равно удобнее из
корня репозитория.
