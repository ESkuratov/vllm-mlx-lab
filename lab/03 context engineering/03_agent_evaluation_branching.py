# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 03: evaluation и fork/steer/select.

Идея
----
После steering можно получить несколько веток. Week 4 tiny-llm предлагает:

* оценивать **наблюдаемые** факты: final, содержимое файлов, код возврата
  команды, receipts;
* не оценивать скрытые рассуждения и точную форму transcript;
* выбрать passing-ветку явно, не притворяясь, что side effects откатились.

Здесь две ветки от одного checkpoint: одна исправляет файл и проходит тест,
другая только пишет документацию. Evaluation выбирает прошедшую.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/03 context engineering/03_agent_evaluation_branching.py"
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import note, section, subsection

from agent_utils import (
    Criterion,
    branch_checkpoint,
    evaluate_run,
    select_branch,
)


def base_checkpoint() -> dict:
    return {
        "step": 4,
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Fix the failing matmul test."},
            {"role": "assistant", "content": '{"tool":"read_file","path":"notes.txt"}'},
            {"role": "user", "content": "Tool result:\nTODO: implement tiled matmul"},
        ],
        "receipts": {"read": {"path": "notes.txt", "chars": 28}},
    }


def criteria() -> list[Criterion]:
    return [
        Criterion(
            "final is non-empty",
            lambda facts: bool(facts.get("final", "").strip()),
        ),
        Criterion(
            "file contains DONE",
            lambda facts: "DONE" in facts["files"].get("notes.txt", ""),
        ),
        Criterion(
            "test command exited 0",
            lambda facts: facts["commands"].get("pytest") == 0,
        ),
        Criterion(
            "edit receipt exists",
            lambda facts: "edit" in facts.get("receipts", {}),
        ),
    ]


def evaluate_branch(name: str, facts: dict) -> dict:
    result = evaluate_run(facts, criteria())
    print(f"\n  branch {name}: {'PASS' if result['passed'] else 'FAIL'}")
    for item in result["results"]:
        mark = "✓" if item["passed"] else "✗"
        print(f"    {mark} {item['name']}")
    return result


def main() -> None:
    section("03. Evaluation и fork/steer/select")
    base = base_checkpoint()
    print(
        f"  base checkpoint: step={base['step']}, "
        f"messages={len(base['messages'])}, receipts={len(base['receipts'])}"
    )

    subsection("A. Две ветки от одного checkpoint")
    fix = branch_checkpoint(base, "Prefer a minimal code edit over docs.")
    docs = branch_checkpoint(base, "Write documentation instead of code.")
    print(f"  branch fix:  {fix['instruction']}")
    print(f"  branch docs: {docs['instruction']}")

    # Наблюдаемые факты после выполнения веток.
    fix_facts = {
        "final": "Fixed the tiled matmul path; the test passes.",
        "files": {"notes.txt": "DONE: tiled matmul implemented\n"},
        "commands": {"pytest": 0},
        "receipts": {"edit": {"chars": 32}, "read": {"chars": 28}},
        "reasoning": "I thought about padding and split-K for a while.",
    }
    docs_facts = {
        "final": "Documented paged attention in README.",
        "files": {"notes.txt": "TODO: implement tiled matmul"},
        "commands": {"pytest": 1},
        "receipts": {"read": {"chars": 28}},
        "reasoning": "I decided documentation is enough.",
    }

    subsection("B. Evaluation по observable facts")
    results = {
        "fix": evaluate_branch("fix", fix_facts),
        "docs": evaluate_branch("docs", docs_facts),
    }
    note(
        "Поле reasoning не входит в criteria: оцениваются final, файлы, "
        "код возврата и receipts, а не «правильные мысли»."
    )

    subsection("C. Явный выбор ветки")
    selected, passed = select_branch(results)
    print(f"  passing branches: {passed}")
    print(f"  selected: {selected}")
    assert selected == "fix", (selected, passed)
    assert passed == ["fix"]

    note(
        "Fork/steer/select не откатывает уже сделанные side effects. "
        "Выбор фиксирует, какая ветка принята, и какие receipts к ней относятся."
    )
    note(
        "Если бы обе ветки прошли, политика выбора должна быть явной: "
        "первая, лучшая по метрике или ручное решение оператора."
    )

    section("Вывод")
    note("Evaluation смотрит только на наблюдаемые артефакты.")
    note("Скрытые reasoning и точная форма transcript не оцениваются.")
    note("Fork создаёт ветки от checkpoint; select фиксирует выбор.")
    note("Это Week 4 Days 7–8.")


if __name__ == "__main__":
    main()
