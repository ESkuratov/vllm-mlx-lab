# lab — пять модулей о том, как устроена LLM и её сервис

Мини-лаборатория на Apple Silicon (MLX): из чего состоит модель, как её
считает инференс, как ведёт себя агент, как мерить качество и эксплуатацию —
и что именно оптимизирует `vllm-mlx`. Каждый скрипт самостоятельный, с замерами
и печатью выводов. Модули идут в порядке `01` → `05`; инференс — второй из них,
а не весь курс.

Лаборатории разложены по темам и пронумерованы **внутри своей папки**: `01`, `02`,
… Имена папок длинные, поэтому в тексте лабы называются коротким алиасом из
третьей колонки: `inference/07`, `model/16`, `context/03`, `eval/01`,
`observability/01`.

| Папка | Алиас | О чём | Файлов |
| --- | --- | --- | --- |
| [`01 model engineering/`](01%20model%20engineering/README.md) | `model/` | из чего состоит модель, как её собрать и как считать на GPU: block, RoPE, MoE, BERT, VLM, Metal-ядра | 21 |
| [`02 inference engineering/`](02%20inference%20engineering/README.md) | `inference/` | механика и обслуживание: prefill/decode, KV-кэш, память, сэмплинг, ограничения, слоты, страницы, политики | 24 + `logits.py` |
| [`03 context engineering/`](03%20context%20engineering/README.md) | `context/` | цикл агента, контекст, bounded evidence, оценка прогонов | 4 |
| [`04 eval engineering/`](04%20eval%20engineering/README.md) | `eval/` | качество как измерение: набор, схема прогона, интервалы, судьи, гейты | 1 |
| [`05 observability engineering/`](05%20observability%20engineering/README.md) | `observability/` | латентность и нагрузка: TTFT/TPOT, p95, очередь, кэши, память | 1 |

Интуиция и метафоры к ключевым понятиям — в [`CONCEPTS.md`](CONCEPTS.md):
одна страница с аналогиями, у каждой указано, что она предсказывает и где
ломается. У каждой лабы там шесть блоков: концепция, проблема, плюсы, минусы,
**замер** (какая метрика смотрится в выводе лабы) и три аналогии.

3D-раскадровки концепций (предложение) — в [`3D_ATLAS.md`](3D_ATLAS.md):
девять залов, 50 сцен, первая волна из 10.

Общие модули лежат в корне `lab/`:

| Файл | Роль |
| --- | --- |
| `common.py` | загрузка модели, тайминги, размеры кэша, формат вывода |
| `metal_ops.py` | общие ядра RMSNorm / RoPE / SwiGLU |
| `agent_utils.py` | протокол, компакция, evaluation, evidence |
| `_loader.py` | поиск и загрузка лаб по имени независимо от папки |

## Как запускать

```bash
source ./vllm-env/bin/activate
python "lab/01 model engineering/07_full_transformer_block.py"
python "lab/02 inference engineering/01_prefill_vs_decode.py"
```

Или всё подряд (займёт несколько минут):

```bash
./lab/run_all.sh
```

Модель по умолчанию — `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`.
Переопределяется переменной `LAB_MODEL`:

```bash
LAB_MODEL=mlx-community/Llama-3.2-3B-Instruct-4bit python "lab/02 inference engineering/01_prefill_vs_decode.py"
```

Другие переменные: `LAB_DRAFT` (draft-модель для speculative decoding),
`LAB_REAL=1` (калибровка синтетических констант по реальной модели),
`LAB_MOE_MODEL` (какая MoE-модель грузится в `21`).

## Требования

- Apple Silicon (Metal), окружение `./vllm-env` уже настроено.
- Скачанные модели (см. корневой `readme.md`, раздел «Установка модели»).
- Реальную модель грузят `model/13`, `model/20`,
  `model/21` и `inference/23`; последняя использует приватные
  `_merge_caches` / `_extend_cache` из mlx-lm.
- `model/14` требует CMake, nanobind и полный Xcode; без них печатает
  чек-лист и пропускает сборку.
- Metal нужен `model/09`–`12`, `model/15`, `model/16`,
  `model/18` и `inference/19`; без Metal они пропускаются.
- Лабы в `context/` не грузят LLM и работают на чистом Python.
- Лабы в `04 eval engineering/` и `05 observability engineering/` грузят
  реальную модель: там замеряются ответы и тайминги, а не симуляции.
- `model/19` не грузит 30B: это синтетическая уменьшённая Qwen3-MoE.

Первый запуск каждого скрипта дольше: модель грузится и компилируются ядра Metal.

## Соответствие флагам vllm-mlx

| Концепция | Где во vllm-mlx |
| --- | --- |
| Continuous batching | `--continuous-batching`, `--max-num-seqs`, `--prefill-batch-size`, `--completion-batch-size` |
| Prefix caching | `--enable-prefix-cache`, `--prefix-cache-size`, `--cache-memory-mb`, `--cache-memory-percent` |
| Paged attention | `--use-paged-cache`, `--paged-cache-block-size`, `--max-cache-blocks` |
| Квантизация KV | `--kv-cache-quantization`, `--kv-cache-quantization-bits {4,8}`, `--kv-cache-quantization-group-size`, `--kv-cache-min-quantize-tokens` |
| Prefill-чанки | `--prefill-step-size`, `--chunked-prefill-tokens`, `--prefill-batch-size` |
| Скользящее окно KV | `--max-kv-size` |
| Speculative / MTP | `--enable-mtp`, `--mtp-num-draft-tokens`, `--mllm-draft-model`, `--specprefill*` |
| SSD-тир | `--ssd-cache-dir`, `--ssd-cache-max-gb` |
| Детокенизация | `vllm-mlx bench-detok`, `--stream-interval` |
| Constrained decoding | `response_format` в `server.py` |
| Tool / reasoning | `--enable-auto-tool-choice`, `--tool-call-parser`, `--reasoning-parser` |
| Эмбеддинги / rerank | `--embedding-model`, `--rerank-model` |
| Мультимодальность | `--mllm`, `--mllm-prefill-step-size`, `--max-audio-upload-mb` |
| Планировщик | `--continuous-batching`, `--max-num-seqs`, `--gpu-memory-utilization` |

Реальный код управления кэшем лежит в установленном пакете:
`vllm-env/lib/python3.12/site-packages/vllm_mlx/` —
`paged_cache.py`, `prefix_cache.py`, `memory_cache.py`, `ssd_cache.py`,
`scheduler.py`, `attention.py`.

## Идеи для дальнейших экспериментов

1. **LoRA / адаптеры** — как адаптер меняет веса «на лету» и уживается с базовой
   моделью в одном батче → `model/`.
2. **Prefill/decode disaggregation** — разнести prefill и decode по разным
   пулам и посмотреть на latency/throughput → `inference/`.
3. **KV-квантизация на длинном контексте** — накопление ошибки логитов при
   `kv_bits=4` с ростом контекста → `inference/`.
4. **Взаимодействие окна и prefix caching** — когда общий префикс длиннее окна
   → `inference/`.
5. **Vision-энкодер по-настоящему** — прогнать реальный VLM (`mlx_vlm`) и
   сравнить с симуляцией из `model/06` → `model/`.
6. **SpecPrefill** — сжатие длинных промптов draft-моделью (`specprefill.py`)
   → `inference/`.
7. **Multimodal scheduling** — как мультимодальные запросы влияют на батчинг
   (`mllm_scheduler.py`) → `inference/`.
8. **Наблюдаемость под нагрузкой** — очереди, occupancy и хвост латентности
   на живом сервере (`bench_serve.py`, `metrics.py`) → `05 observability engineering/`.
9. **P·V через MMA** — добавить в `model/16` и `model/18` `simdgroup_matrix` для второй матрицы и
   плитку 32×32 → `model/`.
10. **Реальный Qwen3-MoE** — прогнать `model/19` на 30B-A3B чекпоинте
    → `model/`.
11. **Production tuning** — occupancy, double-buffering, split-K одним grid и
    сравнение с MLX на 8K/32K → `model/`.
12. **Судья без эталона** — перенести verbalizer-классификацию в eval-судью с
    калибровкой и замером согласия с человеком → `04 eval engineering/`.
