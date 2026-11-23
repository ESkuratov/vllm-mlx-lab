# context — дизайн агентов

Цикл агента и всё, что нужно, чтобы он не разваливался: строгий протокол одного
действия на шаг, approval на side effects, receipts, компакция контекста,
bounded evidence, оценка прогонов и ветвление от checkpoint.

Модель ни в одной из этих лаб не нужна: цикл проверяется на scripted-модели, а
остальное — на уровне transcript'а. Общий модуль — `lab/agent_utils.py`.

Номера нумеруются внутри папки: `01`…`04`.

Запуск — из корня репозитория:

```bash
source ./vllm-env/bin/activate
python "lab/03 context engineering/01_agent_loop.py"
```

## Эксперименты

| Файл | О чём | Ключевой вывод |
| --- | --- | --- |
| `01_agent_loop.py` | JSON-протокол, tools, approval, receipts, checkpoint | минимальное ядро агента: модель scripted, проверяется harness |
| `02_agent_compaction_steering.py` | компакция контекста и steering | старые tool results → receipts; checkpoint + инструкция оператора |
| `03_agent_evaluation_branching.py` | evaluation и fork/steer/select | оцениваются observable facts, а не скрытые reasoning |
| `04_agent_bounded_evidence.py` | bounded evidence и range retrieval | большие bytes вне prompt; digest + head/tail + явный диапазон |

## Как они складываются

```
01  протокол и цикл: одно действие на шаг, approval, receipts, лимиты
     │
02  контекст: компакция старых результатов, checkpoint, steering
     │
04  большие результаты: bounded observation + запрос диапазона байт
     │
03  оценка и выбор: критерии по фактам, fork → evaluate → select
```

Связь с сервисом: `inference/17_tool_reasoning_parsing.py` показывает, как
tool-call извлекается из сырого текста парсерами `vllm_mlx` — строгий JSON
протокола `01` — это контракт самого цикла, а не формат модели.
