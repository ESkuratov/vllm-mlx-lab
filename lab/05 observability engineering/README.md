# observability engineering — что сервер реально делает

Латентность, хвосты, очередь, кэши и память: числа, по которым видно, что
происходит с сервисом под нагрузкой. Всё считается из двух отметок времени
(первый токен и последний) и счётчиков `vllm_mlx/metrics.py`.

Модель нужна настоящая: TTFT и TPOT — свойства модели и железа, а не
симуляции. Общие замеры — `lab/common.py`.

Номера нумеруются внутри папки: `01`, `02`, …

Запуск — из корня репозитория:

```bash
source ./vllm-env/bin/activate
python "lab/05 observability engineering/01_ttft_tpot.py"
```

## Эксперименты

| Файл | О чём | Ключевой вывод |
| --- | --- | --- |
| `01_ttft_tpot.py` | TTFT vs длина промпта, TPOT, цена синхронизации, холодный старт, p50/p95 | латентность — это пара (TTFT, TPOT): TTFT = префилл, TPOT = decode-шаг; холодный прогон дороже в 1.3× |

## Как это складывается

```
01  две отметки времени: TTFT (prefill) и TPOT (decode), плюс p95 и прогрев
     │
     ├── хвост: распределение, а не среднее — p50/p95/p99 и что делает очередь
     ├── загрузка: occupancy слотов, глубина очереди, закон Литтла
     ├── кэши и память: hit rate prefix/paged-кэша, вытеснения, peak memory
     └── нагрузочный тест: closed vs open loop, coordinated omission
```

Связь с остальными папками:

- `inference/01_prefill_vs_decode.py` — почему TTFT и TPOT
  масштабируются по-разному; здесь это же, но как метрика.
- `inference/03_continuous_batching.py` — откуда берётся хвост
  под нагрузкой: слоты и очередь.
- `inference/09_prefix_caching.py` — TTFT, который падает в
  разы; в observability это hit rate кэша.
- `inference/24_capacity_cache.py` — память и вытеснение как
  метрика, а не как механика.

## План

| Тема | Что должно появиться |
| --- | --- |
| `02_percentiles_tail.py` | распределение латентности, p50/p95/p99, почему mean врёт |
| `03_queue_and_saturation.py` | слоты, очередь, закон Литтла, utilization |
| `04_cache_hit_and_memory.py` | hit rate prefix/paged-кэша, вытеснения, peak memory |
| `05_load_test.py` | closed vs open loop, coordinated omission, workload из `bench_serve.py` |
