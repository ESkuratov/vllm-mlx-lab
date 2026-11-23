# vllm-mlx-lab — пять модулей о LLM на Apple Silicon

`01` модель → `02` инференс → `03` контекст и агент → `04` оценка →
`05` наблюдаемость: 51 самостоятельный скрипт с замерами на MLX плюс конспект по
`vllm-mlx` — OpenAI/Anthropic-совместимому inference-серверу для Apple Silicon
(он же плагин платформы `mlx` для vLLM).

Модель по умолчанию во всех примерах: `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`.

## Содержание

- [1. Мониторинг ресурсов (asitop)](#1-мониторинг-ресурсов-asitop)
- [2. Установка модели](#2-установка-модели)
- [3. Запуск сервера](#3-запуск-сервера)
- [4. Проверка сервера и метрики](#4-проверка-сервера-и-метрики)
- [5. Чат через API](#5-чат-через-api)
- [6. Gradio-клиент](#6-gradio-клиент)
- [7. Курс в lab/](#7-курс-в-lab)

---

## 1. Мониторинг ресурсов (asitop)

```bash
source ./vllm-env/bin/activate
pip install asitop          # в окружение vllm-env не входит
sudo asitop                 # нужен доступ к powermetrics
```

## 2. Установка модели

```bash
source ./vllm-env/bin/activate
vllm-mlx download mlx-community/Qwen2.5-Coder-3B-Instruct-4bit
```

## 3. Запуск сервера

```bash
source ./vllm-env/bin/activate
vllm-mlx serve mlx-community/Qwen2.5-Coder-3B-Instruct-4bit --port 8000
```

> **Важно:** у `vllm-mlx` первой всегда идёт подкоманда (`serve`, `download`, …),
> а модель — позиционный аргумент после неё. Опция `--served-model-name`
> принадлежит `serve` и задаёт имя модели в API (по умолчанию — путь к модели).

## 4. Проверка сервера и метрики

```bash
curl -s localhost:8000/health       # health check
curl -s localhost:8000/v1/status    # status
curl -s localhost:8000/v1/models    # список моделей и их имён
curl -s localhost:8000/metrics      # Prometheus-метрики
```

`/metrics` по умолчанию выключен и отдаётся без авторизации: включи флагом
`vllm-mlx serve … --enable-metrics`. Оттуда берутся TTFT, длительности
запросов и счётчики токенов — то, чем занимается лаба
`lab/05 observability engineering/`.

Другие подкоманды, полезные вне сервера: `vllm-mlx bench` (бенчмарк модели),
`vllm-mlx bench-detok` (стоимость потоковой детокенизации), `vllm-mlx
bench-kv-cache` (экономия памяти от квантизации KV), `vllm-mlx bench-serve
--url http://127.0.0.1:8000` (нагрузка на живой сервер; по умолчанию ждёт
`8080`), `vllm-mlx model inspect|acquire|convert|register|qualify`.

## 5. Чат через API

Отдельная команда не нужна: модель обслуживает запущенный сервер,
обращайся к OpenAI-совместимому эндпоинту.

```bash
curl -s localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit",
    "messages": [{"role": "user", "content": "2+2 equals"}]
  }'
```

## 6. Gradio-клиент

Клиент запускается **отдельным процессом** и подключается к уже работающему
серверу (сначала выполни `vllm-mlx serve` выше). Значение `--served-model-name`
у клиента должно совпадать с именем модели из `curl -s localhost:8000/v1/models`.

**Текстовый клиент** — для LLM, порт по умолчанию **7861**:

```bash
source ./vllm-env/bin/activate
vllm-mlx-text-chat --served-model-name mlx-community/Qwen2.5-Coder-3B-Instruct-4bit
```

**Мультимодальный клиент** — картинки/видео, порт по умолчанию **7860**:

```bash
# для обычной текстовой LLM добавь --text-only
vllm-mlx-chat --text-only --served-model-name mlx-community/Qwen2.5-Coder-3B-Instruct-4bit
```

### Ссылки

| Что | Адрес |
| --- | --- |
| Текстовый чат | http://127.0.0.1:7861 |
| Мультимодальный чат | http://127.0.0.1:7860 |
| Статус сервера | http://localhost:8000/v1/status |

## 7. Курс в lab/

`lab/` — учебная лаборатория, а не набор демок: каждый скрипт самостоятельный,
печатает замеры и вывод. Пять тем идут по порядку — от устройства модели до
эксплуатации; номера нумеруются **внутри темы**, поэтому ссылаться на лабу лучше
с папкой (`inference/07`, `model/16`).

| № | Тема | О чём | Лаб |
| --- | --- | --- | --- |
| 01 | `lab/01 model engineering/` | из чего состоит модель и как считать её на GPU: block, RoPE, MoE, BERT, VLM, Metal-ядра (W4-матмулы, плиточное внимание) | 21 |
| 02 | `lab/02 inference engineering/` | механика и обслуживание: prefill/decode, KV-кэш, батчинг, сэмплинг, ограничения, страницы, политики | 24 + `logits.py` |
| 03 | `lab/03 context engineering/` | цикл агента: протокол, approval, receipts, компакция контекста, bounded evidence, оценка прогонов | 4 |
| 04 | `lab/04 eval engineering/` | качество как измерение: интервалы, шум схемы прогона, судьи, A/B-гейты | 1 |
| 05 | `lab/05 observability engineering/` | эксплуатация: TTFT/TPOT, хвосты латентности, очередь, кэши, память | 1 |

Темы 04 и 05 пока содержат по одной лабе из запланированных пяти — остальные
расписаны в README соответствующих папок.

Модули идут в порядке `01` → `05`, лабы внутри модуля самостоятельные: можно
проходить подряд или прыгать по интересу. Вход в каждый модуль —
`model/01`, `inference/01`, `context/01`, `eval/01`, `observability/01`;
внутри модуля стоит начать с `model/07` (transformer block с нуля) и
`inference/02` (KV-кэш).

Таблицы по каждому файлу (файл → о чём → ключевой вывод), соответствие флагам
`vllm-mlx` и идеи для дальнейших экспериментов — в
[`lab/README.md`](lab/README.md) и README каждой папки.

Запуск:

```bash
source ./vllm-env/bin/activate
./lab/run_all.sh          # всё подряд, несколько минут
python "lab/02 inference engineering/01_prefill_vs_decode.py"   # или по одному
```

Модель — та же, что и в примерах выше. Переопределяется переменными окружения:
`LAB_MODEL` (любая лаба), `LAB_DRAFT` (speculative decoding), `LAB_MOE_MODEL`
(`model/21`), `LAB_REAL=1` (калибровка симуляторов по реальной модели).

Требования: Apple Silicon и скачанная модель; Metal — для `model/09`–`12`, `15`,
`16`, `18` и `inference/19`; CMake, nanobind и Xcode — для `model/14`; реальный
чекпоинт — для `model/13`, `20`, `21` и `inference/23`. Лабы `context/` не грузят
LLM вовсе, `eval/` и `observability/` — грузят и меряют ответы и тайминги. Первый
запуск каждого скрипта дольше: модель грузится и компилируются ядра Metal.
