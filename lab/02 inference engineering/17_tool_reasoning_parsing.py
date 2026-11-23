# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 17: tool-calling и reasoning parsing.

Идея
----
Модель отдаёт обычный текст. Чтобы получить из него структуру (вызов
инструмента, «размышления» отдельно от ответа), сервер:
  1) формирует промпт по шаблону чата — туда попадают описания инструментов;
  2) генерирует текст;
  3) парсит его специализированным парсером под формат конкретной модели.

Здесь используем настоящие парсеры из `vllm_mlx.tool_parsers` и
`vllm_mlx.reasoning` — те же, что применяет сервер.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/02 inference engineering/17_tool_reasoning_parsing.py"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlx.core as mx
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

from common import MODEL_ID, load_model, note, section, subsection


def gen(model, tok, prompt_ids, n: int = 96) -> str:
    eos = set(getattr(tok, "eos_token_ids", {tok.eos_token_id}))
    out = []
    for t, _ in generate_step(
        mx.array(prompt_ids), model, max_tokens=n, sampler=make_sampler(temp=0.0)
    ):
        t = int(t)
        if t in eos:
            break
        out.append(t)
    return tok.decode(out)


def main() -> None:
    section("17. Tool-calling и reasoning parsing")
    print(f"  модель: {MODEL_ID}")
    model, tok = load_model()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]

    # ---------------------------------------------------- шаблон инструментов
    subsection("A. Как инструменты попадают в промпт")
    try:
        prompt_ids = tok.apply_chat_template(
            messages, tools=tools, add_generation_prompt=True
        )
        shown = tok.decode(prompt_ids)
        head = "\n".join(shown.strip().splitlines()[:12])
        print(head)
        print("  ...")
        note(
            "Сервер вставляет JSON-описания инструментов в системный промпт и "
            "указывает формат ответа-вызова. Это делает chat template модели."
        )
    except Exception as exc:
        print(f"  chat template недоступен: {exc}")
        prompt_ids = tok.encode(
            "Call get_weather(city='Paris'). Return a JSON tool call."
        )

    # ------------------------------------------------------- генерация
    subsection("B. Генерация и парсинг tool-call")
    raw = gen(model, tok, prompt_ids)
    print(f"  сырой вывод: {raw[:220]!r}")

    from vllm_mlx.tool_parsers import ToolParserManager

    parser = ToolParserManager.get_tool_parser("qwen")(tok)
    result = parser.extract_tool_calls(raw)
    print(f"  qwen-парсер: tools_called={result.tools_called}")
    for tc in result.tool_calls or []:
        print(f"    → {tc['name']}({tc['arguments']})")
    if not result.tools_called:
        note("Слабая модель не всегда соблюдает формат — парсер честно вернул «вызовов нет».")

    subsection("C. Что парсер делает с каноничным выводом")
    canon = {
        "qwen": '<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call>',
        "mistral": '[TOOL_CALLS] [{"name": "get_weather", "arguments": {"city": "Paris"}}]',
    }
    for name, text in canon.items():
        p = ToolParserManager.get_tool_parser(name)(tok)
        r = p.extract_tool_calls(text)
        calls = [(tc["name"], tc["arguments"]) for tc in (r.tool_calls or [])]
        print(f"  {name:<8}: {calls}")
    note(
        "Каждая модель пишет вызов по-своему (Qwen — XML-теги, Mistral — "
        "[TOOL_CALLS] и т.д.), поэтому парсер выбирают под модель: "
        "во vllm-mlx это --tool-call-parser (30+ форматов)."
    )

    # -------------------------------------------------------- reasoning
    subsection("D. Reasoning: отделяем «размышления» от ответа")
    think_prompt = tok.encode(
        "Answer step by step, then give the final result.\nQuestion: 17*3?\n"
    )
    reasoned = gen(model, tok, think_prompt, n=80)
    print(f"  сырой вывод: {reasoned[:200]!r}")

    from vllm_mlx.reasoning import get_parser

    rp = get_parser("qwen3")()
    r_reason, r_content = rp.extract_reasoning(reasoned)
    print(f"  разбор реального вывода: reasoning={r_reason!r}, content={r_content!r}")

    canon_think = "Let me multiply 17 by 3. 17*3 = 51.</think>The answer is 51."
    c_reason, c_content = rp.extract_reasoning(canon_think)
    print(f"  разбор каноничного:      reasoning={c_reason!r}, content={c_content!r}")
    note(
        "Reasoning-парсер вырезает содержимое  thinking...</think>: клиенту уходит "
        "чистый ответ, а размышления — отдельным полем. Во vllm-mlx: "
        "--reasoning-parser (qwen3, deepseek_r1, gpt_oss, ...)."
    )

    # ------------------------------------------------------- соберём API-ответ
    subsection("E. Как это выглядит в API")
    if result.tools_called:
        payload = {
            "finish_reason": "tool_calls",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]},
                }
                for tc in result.tool_calls
            ],
        }
        print("  " + json.dumps(payload, ensure_ascii=False))
    else:
        print("  (в этом прогоне модель не вызвала инструмент — структуры нет)")

    section("Вывод")
    note("Парсинг — это слой между «сырым текстом» и структурой API.")
    note("Формат задаёт модель; парсер и chat template выбирают под неё.")
    note("Без парсера tool-call останется просто текстом в ответе.")


if __name__ == "__main__":
    main()
