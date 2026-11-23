# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 01: минимальный coding agent — протокол, инструменты, receipts.

Идея
----
Week 4 tiny-llm начинается с валидированного agent loop:

* модель возвращает **один** JSON-объект — tool call или final;
* аргументы проверяются по схеме;
* side effects (запись, команда) требуют approval;
* каждое действие и результат пишутся в receipts;
* loop ограничен шагами, ошибками и повторяющимися действиями;
* состояние можно сохранить в checkpoint и продолжить с него.

Здесь мы пишем такой loop с нуля. Модель — детерминированный скрипт,
поэтому лаборатория не грузит LLM и проверяет именно механику harness.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/03 context engineering/01_agent_loop.py"
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import note, section, subsection


# --------------------------------------------------------------------------- #
# Протокол
# --------------------------------------------------------------------------- #


class AgentError(ValueError):
    """Ошибка, которую можно вернуть модели и продолжить loop."""


@dataclass(frozen=True)
class FinalAction:
    final: str


@dataclass(frozen=True)
class ToolAction:
    tool: str
    arguments: dict[str, Any]


Action = FinalAction | ToolAction

TOOL_SCHEMA = {
    "list_files": (set(), {"path"}),
    "read_file": ({"path"}, set()),
    "write_file": ({"path", "content"}, set()),
    "edit_file": ({"path", "old", "new"}, set()),
    "run_command": ({"argv"}, set()),
}


def parse_action(text: str, available: set[str]) -> Action:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as error:
        raise AgentError(f"response is not valid JSON: {error.msg}") from error
    if not isinstance(raw, dict):
        raise AgentError("response must be a JSON object")

    if "final" in raw:
        if set(raw) != {"final"} or not isinstance(raw["final"], str):
            raise AgentError("final must contain exactly one string field")
        if not raw["final"].strip():
            raise AgentError("final must not be empty")
        return FinalAction(raw["final"])

    tool = raw.get("tool")
    if not isinstance(tool, str):
        raise AgentError("tool action requires a string 'tool' field")
    if tool not in TOOL_SCHEMA:
        raise AgentError(f"unknown tool: {tool}")
    if tool not in available:
        raise AgentError(f"tool is not enabled: {tool}")

    required, optional = TOOL_SCHEMA[tool]
    supplied = set(raw) - {"tool"}
    missing = required - supplied
    extra = supplied - required - optional
    if missing:
        raise AgentError(f"missing fields: {', '.join(sorted(missing))}")
    if extra:
        raise AgentError(f"unexpected fields: {', '.join(sorted(extra))}")

    arguments = {key: raw[key] for key in supplied}
    for key in ("path", "content", "old", "new"):
        if key in arguments and not isinstance(arguments[key], str):
            raise AgentError(f"{key} must be a string")
    if "argv" in arguments and (
        not isinstance(arguments["argv"], list)
        or not arguments["argv"]
        or any(not isinstance(part, str) or not part for part in arguments["argv"])
    ):
        raise AgentError("argv must be a non-empty array of non-empty strings")
    return ToolAction(tool, arguments)


# --------------------------------------------------------------------------- #
# Workspace и receipts
# --------------------------------------------------------------------------- #


class Workspace:
    def __init__(
        self,
        root: Path,
        approve: Callable[[ToolAction], bool],
        allow_writes: bool = True,
        allowed_commands: tuple[tuple[str, ...], ...] = (),
        receipt_path: Path | None = None,
    ):
        self.root = root.resolve()
        if not self.root.is_dir():
            raise ValueError("workspace root must be an existing directory")
        self.approve = approve
        self.allow_writes = allow_writes
        self.allowed_commands = set(allowed_commands)
        self.receipts: list[dict[str, Any]] = []
        self.receipt_path = receipt_path

    @property
    def available_tools(self) -> set[str]:
        tools = {"list_files", "read_file", "run_command"}
        if self.allow_writes:
            tools |= {"write_file", "edit_file"}
        return tools

    def _resolve(self, raw: str) -> Path:
        if not raw or Path(raw).is_absolute() or ".." in Path(raw).parts:
            raise AgentError(f"path must be workspace-relative: {raw!r}")
        path = (self.root / raw).resolve()
        if path != self.root and self.root not in path.parents:
            raise AgentError(f"path escapes workspace: {raw!r}")
        return path

    def _record(self, step: int, action: ToolAction, result: str, ok: bool) -> None:
        receipt = {
            "step": step,
            "tool": action.tool,
            "arguments": action.arguments,
            "ok": ok,
            "result": result[:500],
        }
        self.receipts.append(receipt)
        if self.receipt_path is not None:
            with self.receipt_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(receipt, ensure_ascii=False) + "\n")

    def execute(self, step: int, action: ToolAction) -> str:
        if action.tool in {"write_file", "edit_file", "run_command"}:
            if not self.approve(action):
                result = "error: action denied by operator"
                self._record(step, action, result, False)
                return result
        try:
            result = self._execute(action)
        except Exception as error:
            result = f"error: {error}"
            self._record(step, action, result, False)
            return result
        self._record(step, action, result, True)
        return result

    def _execute(self, action: ToolAction) -> str:
        if action.tool == "list_files":
            path = self._resolve(action.arguments.get("path", "."))
            if not path.is_dir():
                raise AgentError(f"not a directory: {action.arguments.get('path')}")
            names = sorted(p.name for p in path.iterdir())
            return "\n".join(names) if names else "(empty directory)"
        if action.tool == "read_file":
            path = self._resolve(action.arguments["path"])
            if not path.is_file():
                raise AgentError(f"not a file: {action.arguments['path']}")
            text = path.read_text(encoding="utf-8", errors="replace")
            return text[:4000] + ("\n...[truncated]" if len(text) > 4000 else "")
        if action.tool == "write_file":
            if not self.allow_writes:
                raise AgentError("writes are disabled")
            path = self._resolve(action.arguments["path"])
            path.write_text(action.arguments["content"], encoding="utf-8")
            return f"wrote {len(action.arguments['content'])} chars to {path.name}"
        if action.tool == "edit_file":
            if not self.allow_writes:
                raise AgentError("writes are disabled")
            path = self._resolve(action.arguments["path"])
            text = path.read_text(encoding="utf-8")
            old = action.arguments["old"]
            new = action.arguments["new"]
            count = text.count(old)
            if count != 1:
                raise AgentError(f"old text must occur exactly once, found {count}")
            path.write_text(text.replace(old, new), encoding="utf-8")
            return f"edited {path.name}: replaced 1 occurrence"
        if action.tool == "run_command":
            argv = tuple(action.arguments["argv"])
            if argv not in self.allowed_commands:
                raise AgentError(f"command is not allowed: {list(argv)}")
            proc = subprocess.run(
                list(argv),
                cwd=self.root,
                capture_output=True,
                text=True,
                timeout=5,
            )
            output = (proc.stdout + proc.stderr).strip()
            return f"exit={proc.returncode}\n{output[:2000]}"
        raise AgentError(f"unhandled tool: {action.tool}")


# --------------------------------------------------------------------------- #
# Agent loop
# --------------------------------------------------------------------------- #


@dataclass
class AgentEvent:
    step: int
    response: str
    action: Action | None
    result: str | None


@dataclass
class AgentResult:
    completed: bool
    final: str | None
    events: list[AgentEvent] = field(default_factory=list)
    checkpoint: dict[str, Any] | None = None


def system_prompt(workspace: Workspace) -> str:
    return "\n".join(
        [
            "You are a coding agent. Reply with one JSON object and no markdown.",
            'Finish with {"final":"..."}.',
            'Inspect with {"tool":"list_files","path":"."}.',
            'Read with {"tool":"read_file","path":"notes.txt"}.',
            'Edit with {"tool":"edit_file","path":"notes.txt","old":"TODO","new":"DONE"}.',
            'Run with {"tool":"run_command","argv":["cat","notes.txt"]}.',
            "Paths are workspace-relative. Never invent file contents.",
        ]
    )


def run_agent(
    task: str,
    generate: Callable[[list[dict[str, str]]], str],
    workspace: Workspace,
    *,
    max_steps: int = 6,
    checkpoint_path: Path | None = None,
    resume: dict[str, Any] | None = None,
) -> AgentResult:
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if resume is None:
        messages = [
            {"role": "system", "content": system_prompt(workspace)},
            {"role": "user", "content": task},
        ]
        start_step = 1
        events: list[AgentEvent] = []
    else:
        messages = list(resume["messages"])
        start_step = int(resume["step"]) + 1
        events = []

    invalid = 0
    previous_signature: str | None = None
    identical = 0

    for step in range(start_step, max_steps + 1):
        if sum(len(m["content"]) for m in messages) > 48_000:
            return AgentResult(False, None, events, _checkpoint(step - 1, messages))

        response = generate(messages)
        try:
            action = parse_action(response, workspace.available_tools)
        except AgentError as error:
            invalid += 1
            previous_signature = None
            identical = 0
            result = f"error: {error}"
            events.append(AgentEvent(step, response, None, result))
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content": f"Tool result:\n{result}"},
            ]
            if checkpoint_path is not None:
                checkpoint_path.write_text(
                    json.dumps(_checkpoint(step, messages), ensure_ascii=False)
                )
            if invalid >= 3:
                return AgentResult(False, None, events)
            continue

        if isinstance(action, FinalAction):
            events.append(AgentEvent(step, response, action, None))
            return AgentResult(True, action.final, events)

        signature = json.dumps(
            {"tool": action.tool, **action.arguments}, sort_keys=True
        )
        if signature == previous_signature:
            identical += 1
        else:
            previous_signature = signature
            identical = 1
        if identical > 2:
            result = "error: identical action loop detected"
            events.append(AgentEvent(step, response, action, result))
            messages += [
                {"role": "assistant", "content": response},
                {"role": "user", "content": f"Tool result:\n{result}"},
            ]
            return AgentResult(False, None, events)

        result = workspace.execute(step, action)
        events.append(AgentEvent(step, response, action, result))
        messages += [
            {"role": "assistant", "content": response},
            {"role": "user", "content": f"Tool result:\n{result}"},
        ]
        if checkpoint_path is not None:
            checkpoint_path.write_text(
                json.dumps(_checkpoint(step, messages), ensure_ascii=False)
            )

    return AgentResult(False, None, events, _checkpoint(max_steps, messages))


def _checkpoint(step: int, messages: list[dict[str, str]]) -> dict[str, Any]:
    return {"step": step, "messages": messages}


# --------------------------------------------------------------------------- #
# Scripted model
# --------------------------------------------------------------------------- #

SCRIPT = [
    '{"tool":"list_files","path":"."}',
    '{"tool":"read_file","path":"notes.txt"}',
    '{"tool":"edit_file","path":"notes.txt","old":"TODO","new":"DONE"}',
    '{"tool":"run_command","argv":["cat","notes.txt"]}',
    '{"final":"Updated notes.txt and verified it with cat."}',
]


def scripted_model(messages: list[dict[str, str]]) -> str:
    """Выбрать следующий шаг по числу уже сделанных assistant-сообщений."""
    done = sum(1 for message in messages if message["role"] == "assistant")
    return SCRIPT[min(done, len(SCRIPT) - 1)]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def show_events(events: list[AgentEvent]) -> None:
    for event in events:
        action = event.action
        if isinstance(action, ToolAction):
            label = action.tool + " " + json.dumps(action.arguments, ensure_ascii=False)
        elif isinstance(action, FinalAction):
            label = "final"
        else:
            label = "invalid"
        print(f"  [{event.step}] {label}")
        if event.result is not None:
            first = event.result.splitlines()[0]
            print(f"       → {first}")


def main() -> None:
    section("01. Минимальный coding agent")
    with tempfile.TemporaryDirectory(prefix="tiny-agent-") as tmp:
        root = Path(tmp)
        (root / "notes.txt").write_text(
            "TODO: document paged attention\n", encoding="utf-8"
        )
        checkpoint = root / "checkpoint.json"
        receipts = root / "receipts.jsonl"

        approvals = []
        workspace = Workspace(
            root,
            approve=lambda action: approvals.append(action.tool) or True,
            allow_writes=True,
            allowed_commands=(("cat", "notes.txt"),),
            receipt_path=receipts,
        )

        subsection("A. Первый запуск до checkpoint")
        first = run_agent(
            "Update notes.txt and verify the result.",
            scripted_model,
            workspace,
            max_steps=3,
            checkpoint_path=checkpoint,
        )
        show_events(first.events)
        print(f"  completed={first.completed}, step={first.checkpoint['step']}")
        print(f"  checkpoint: {checkpoint.name}, "
              f"messages={len(first.checkpoint['messages'])}")

        subsection("B. Resume с checkpoint")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        resumed = run_agent(
            "Update notes.txt and verify the result.",
            scripted_model,
            workspace,
            max_steps=6,
            checkpoint_path=checkpoint,
            resume=state,
        )
        show_events(resumed.events)
        print(f"  completed={resumed.completed}")
        print(f"  final: {resumed.final}")

        subsection("C. Проверки")
        text = (root / "notes.txt").read_text(encoding="utf-8")
        assert "DONE" in text and "TODO" not in text, text
        print("  ✓ notes.txt изменён")
        assert resumed.completed and resumed.final
        print("  ✓ loop завершился final-ответом")
        assert approvals == ["edit_file", "run_command"], approvals
        print(f"  ✓ approval получен для: {approvals}")
        lines = receipts.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 4, len(lines)
        print(f"  ✓ receipts: {len(lines)} записей")

        # Path traversal должен отклоняться.
        try:
            workspace._resolve("../outside.txt")
        except AgentError:
            print("  ✓ path traversal отклонён")
        else:
            raise AssertionError("path traversal не был отклонён")

        # Команда вне allowlist отклоняется.
        bad = ToolAction("run_command", {"argv": ["rm", "-rf", "/"]})
        result = workspace.execute(99, bad)
        assert result.startswith("error:"), result
        print("  ✓ команда вне allowlist отклонена")

    note(
        "Это ядро Week 4: строгий JSON-протокол, проверка аргументов, "
        "approval на side effects, receipts, лимиты и checkpoint/resume."
    )
    note(
        "В полном tiny-llm дальше добавляются компакция контекста, steering, "
        "evaluation, fork/select и bounded evidence для больших tool results."
    )
    note(
        "Модель здесь scripted — так лаборатория проверяет harness, а не "
        "качество генерации. Тот же loop можно подключить к Qwen через "
        "agent.generate(messages)."
    )

    section("Вывод")
    note("Один JSON-объект на шаг; final — отдельное действие.")
    note("Инструменты валидируют пути и аргументы до side effect.")
    note("Receipts и checkpoint делают прогон воспроизводимым.")
    note("Лимиты шагов/ошибок/повторов гарантируют остановку.")


if __name__ == "__main__":
    main()
