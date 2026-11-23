# vllm-mlx-lab — лаборатория по LLM на Apple Silicon

`lab/` — пять модулей (51 лаба) о том, как устроена LLM и её сервис: от сборки
модели до эксплуатации. Этот файл — про окружение и запуск
`vllm-mlx` — OpenAI/Anthropic-совместимого inference-сервера для Apple Silicon
(он же плагин платформы `mlx` для vLLM). Описание самих лаб — в
[`lab/README.md`](lab/README.md).

Модель по умолчанию: `mlx-community/Qwen2.5-Coder-3B-Instruct-4bit`.

## Содержание

- [1. Мониторинг ресурсов (asitop)](#1-мониторинг-ресурсов-asitop)
- [2. Установка модели](#2-установка-модели)
- [3. Запуск сервера](#3-запуск-сервера)
- [4. Проверка сервера и метрики](#4-проверка-сервера-и-метрики)
- [5. Чат через API](#5-чат-через-api)
- [6. Gradio-клиент](#6-gradio-клиент)
- [7. Лаборатория в lab/](#7-лаборатория-в-lab)

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
| Prometheus-метрики | http://localhost:8000/metrics |

## 7. Лаборатория в lab/

Пять модулей идут по порядку — модель → инференс → контекст и агент → оценка →
наблюдаемость; лабы внутри модуля самостоятельные, нумеруются внутри своей
папки (в тексте — с папкой: `model/07`, `inference/19`).

Всё про курс — таблицы по каждому файлу, соответствие флагам `vllm-mlx`,
требования к железу и пакетам, идеи экспериментов — в
[`lab/README.md`](lab/README.md). Там же [`CONCEPTS.md`](lab/CONCEPTS.md)
(интуиция и метафоры к понятиям) и [`3D_ATLAS.md`](lab/3D_ATLAS.md)
(3D-раскадровки).

```bash
source ./vllm-env/bin/activate
./lab/run_all.sh                                              # всё подряд, несколько минут
python "lab/02 inference engineering/01_prefill_vs_decode.py" # или по одному
```

Модель — та же, что и в примерах выше; переопределяется `LAB_MODEL`. Остальные
переменные (`LAB_DRAFT`, `LAB_MOE_MODEL`, `LAB_REAL`) — в `lab/README.md`.
