# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 02: компакция контекста и steering с checkpoint.

Идея
----
Когда агент делает много tool calls, полные результаты быстро съедают
контекст. Week 4 tiny-llm предлагает:

* **компакцию**: старые завершённые tool results заменяются короткой
  ссылкой на receipt, полный текст остаётся вне prompt;
* **checkpoint**: состояние (messages, step, receipts) сохраняется на диск;
* **steering**: оператор добавляет инструкцию в paused checkpoint, и
  свежая модель продолжает с учётом неё.

Здесь мы делаем это на уровне transcript, без LLM: важна механика.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/03 context engineering/02_agent_compaction_steering.py"
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import note, section, subsection

from agent_utils import (
    branch_checkpoint,
    compact_messages,
    load_checkpoint,
    messages_chars,
    save_checkpoint,
)


def make_transcript() -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": "You are a coding agent."},
        {"role": "user", "content": "Inspect the repo, fix the test, and verify."},
    ]
    results = [
        ("list_files", "src\nnotes.txt\ntests\nREADME.md"),
        ("read_file", "notes.txt: TODO: replace naive matmul with tiled one.\n" * 40),
        ("edit_file", "edited notes.txt: replaced 1 occurrence"),
        ("run_command", "exit=0\npytest -q\n4 passed in 0.12s"),
        ("read_file", "README.md\n" + ("Long documentation line.\n" * 120)),
    ]
    for i, (tool, result) in enumerate(results, start=1):
        messages.append(
            {
                "role": "assistant",
                "content": f'{{"tool":"{tool}","step":{i}}}',
            }
        )
        messages.append({"role": "user", "content": "Tool result:\n" + result})
    return messages


def main() -> None:
    section("02. Компакция контекста и steering")
    messages = make_transcript()
    print(
        f"  исходный transcript: {len(messages)} сообщений, "
        f"{messages_chars(messages)} символов"
    )

    subsection("A. Компакция старых tool results")
    compacted, report = compact_messages(messages, keep_recent=2)
    print(
        f"  после компакции: {messages_chars(compacted)} символов, "
        f"сэкономлено {report['saved_chars']}"
    )
    print(f"  сжато результатов: {report['compacted']}")
    assert report["compacted"] == 3, report
    assert report["after_chars"] < report["before_chars"]
    # Последние два результата не тронуты.
    assert "Long documentation line." in compacted[-1]["content"]
    assert "pytest -q" in compacted[-3]["content"]
    # Полный текст сохранён в receipts.
    receipt = max(report["receipts"].values(), key=lambda item: item["chars"])
    assert receipt["chars"] > 1000
    print(
        f"  receipt: {receipt['chars']} chars, "
        f"sha256={receipt['sha256'][:16]}... (полный текст вне prompt)"
    )
    note(
        "Компакция не удаляет данные: она заменяет их в prompt на receipt "
        "id + digest, а полный текст остаётся в receipt store."
    )

    subsection("B. Checkpoint и resume")
    with tempfile.TemporaryDirectory(prefix="agent-ckpt-") as tmp:
        path = Path(tmp) / "checkpoint.json"
        save_checkpoint(
            path,
            step=5,
            messages=compacted,
            receipts=report["receipts"],
        )
        state = load_checkpoint(path)
        print(
            f"  checkpoint: step={state['step']}, "
            f"messages={len(state['messages'])}, "
            f"receipts={len(state['receipts'])}"
        )
        assert state["step"] == 5 and len(state["receipts"]) == 3

        subsection("C. Steering paused agent")
        steer = branch_checkpoint(
            state,
            "Prefer a minimal edit. Explain why the test passes in one sentence.",
        )
        print(f"  operator instruction: {steer['instruction']}")
        # Свежая модель видит checkpoint + инструкцию.
        final = (
            "Minimal edit applied; the test passes because the tiled matmul "
            "matches the reference on partial tiles."
        )
        steer["messages"].append({"role": "assistant", "content": f'{{"final":"{final}"}}'})
        print(f"  resumed final: {final}")
        assert "operator instruction" in steer["messages"][-2]["content"].lower()
        assert "minimal edit" in steer["instruction"].lower()
        note(
            "Steering не переигрывает завершённые side effects: он добавляет "
            "инструкцию в paused transcript и продолжает с checkpoint."
        )

    subsection("D. Что осталось за кадром")
    note(
        "В полном tiny-llm checkpoint содержит tool receipts и точные "
        "артефакты; здесь мы храним messages + receipts, чтобы показать "
        "механику."
    )
    note(
        "Компакция должна запускаться до лимита контекста и не трогать "
        "текущий незавершённый tool call."
    )

    section("Вывод")
    note("Компакция меняет prompt, но не receipts.")
    note("Checkpoint позволяет продолжить с того же шага.")
    note("Steering добавляет инструкцию, не откатывая side effects.")
    note("Всё это — Week 4 Days 5–6.")


if __name__ == "__main__":
    main()
