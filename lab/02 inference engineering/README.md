# inference — механика и обслуживание

Как считается один запрос и как обслуживается поток запросов: prefill против
decode, KV-кэш и память, сэмплинг и ограничения на выход, а дальше — слоты,
страницы, кэши и политики очереди.

Архитектура модели и Metal-ядра — в [`../model/`](../01%20model%20engineering/README.md).

Запуск — из корня репозитория:

```bash
source ./vllm-env/bin/activate
python "lab/02 inference engineering/01_prefill_vs_decode.py"
```

Номера нумеруются внутри папки: `01`, `02`, … `24` плюс `logits.py` без номера.
Ссылки в тексте даются с папкой — `inference/19`, `model/16`.

## Механика запроса

| Файл | О чём | Ключевой вывод |
| --- | --- | --- |
| `01_prefill_vs_decode.py` | prefill vs decode | prefill = compute-bound (все токены разом), decode = memory-bound (по токену, время шага ~константа) |
| `02_kv_cache.py` | устройство KV-кэша | убирает повторный prefill → генерация O(n); плата — память O(контекст) |
| `03_continuous_batching.py` | зачем нужен continuous batching | батч амортизирует чтение весов; continuous выжимает простой слотов |
| `04_paged_attention.py` | paged attention и префиксы | блоки = страницы памяти → разделение префиксов, COW, LRU-вытеснение |
| `05_kv_quantization.py` | квантизация KV-кэша | 8/4 бита экономят >50% памяти, 8 бит почти без потери качества |
| `06_sampling.py` | сэмплинг: T, top-k, top-p | как логиты превращаются в выбор токена; энтропия и разнообразие |
| `logits.py` | логиты следующего токена | базовый пример: что вообще выдаёт модель |
| `07_speculative_decoding.py` | draft-модель + проверка одним проходом | ускорение растёт с «долей из draft»; лишние draft-проходы не окупаются |
| `08_chunked_prefill.py` | prefill по чанкам | время то же, пик RAM падает |
| `09_prefix_caching.py` | переиспользование общего префикса | TTFT падает в разы: пересчитывается только хвост |
| `10_weight_quantization.py` | веса 2/4/8/16 бит | память ∝ битам; 8 бит ≈ без потери, 4 бита заметно шумят |
| `11_roofline.py` | memory- vs compute-bound | батч поднимает memory-roofline, но рост упирается в compute |
| `12_sliding_window_quality.py` | окно KV vs качество | маленькое окно «забывает» факт из начала → другой ответ |
| `13_ssd_kv_offload.py` | KV-кэш на диске | сериализация в safetensors; загрузка дешевле повторного prefill |
| `14_detokenization.py` | наивный vs инкрементальный detok | наивно O(n²) и в разы дороже инкрементального |
| `15_batch_determinism.py` | детерминизм greedy в батче | на этой конфигурации токены совпадают; полагаться нельзя |
| `16_structured_decoding.py` | constrained decoding, logits processors | маскирование логитов делает вывод валидным by construction |
| `17_tool_reasoning_parsing.py` | tool-call и reasoning парсеры | структура появляется из текста только при парсере под формат модели |
| `21_speculative_from_scratch.py` | draft → verify → accept/reject → KV rewind | greedy speculative обязан совпасть с greedy target |
| `22_verbalizer_classification.py` | классификация по логитам меток (Jev) | один prefill вместо генерации; порядок вариантов и prior классов меняют ответ сильнее, чем содержание |

## Обслуживание и память сервиса

| Файл | О чём | Ключевой вывод |
| --- | --- | --- |
| `18_scheduler.py` | политики FCFS/SJF/priority/fair, справедливость, starvation | батчинг даёт throughput; политика решает, кому какие задержки |
| `19_paged_attention_from_scratch.py` | page pool, block_table, gather и direct page-walk | paged KV убирает dense-реконструкцию; gather проще, direct не копирует историю |
| `20_continuous_batching_from_scratch.py` | слоты, prefill/decode, chunked admission, политики | continuous снижает TTFT; размер чанка меняет gap против overhead |
| `23_real_continuous_batching.py` | реальные слоты и `BatchKVCache` | настоящий batched forward: ускорение и совпадение greedy на 3B |
| `24_capacity_cache.py` | dense KV-cache: capacity, rewind, release | capacity убирает growth-копии; rewind нужен speculative decoding |

`20` умеет калибровать синтетическую модель стоимости по реальной модели:

```bash
LAB_REAL=1 python "lab/02 inference engineering/20_continuous_batching_from_scratch.py"
```

## Как это связано

```
prefill ──► decode ──► decode ──► ...
   │           │
 compute    memory-bound: веса читаются на каждый токен
   │           ├──► batching (03) и roofline (11)
   │           ├──► speculative (07, 21)
   │           └──► детокенизация (14), детерминизм (15)
   │
   └──► KV-кэш (02) экономит повторный prefill
             ├──► paged attention (04) и prefix caching (09)
             ├──► квантизация KV (05), окно (12), SSD-тир (13)
             ├──► страницы своими руками (19), capacity-кэш (24)
             └──► O(S²) внимания и GQA (model/01)

   слоты и очередь: 20 (симулятор) → 23 (реальные кэши) → 18 (политика)
```

## Границы

- `20` и `24` — симуляторы: показывают политику и метрики, но не делают
  настоящий batched forward;
- `23` использует приватные `_merge_caches` / `_extend_cache` из mlx-lm — при
  обновлении библиотеки сигнатуры могут измениться;
- decode-часть (capacity-cache в живом цикле) пока не собрана в одну лабу — это
  открытый пункт в [`../README.md`](../README.md).
