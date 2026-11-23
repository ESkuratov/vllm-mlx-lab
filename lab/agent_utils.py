# SPDX-License-Identifier: Apache-2.0
"""Общие утилиты для Week 4-лабораторий: протокол, компакция, evaluation, evidence."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


# --------------------------------------------------------------------------- #
# Протокол
# --------------------------------------------------------------------------- #


class AgentError(ValueError):
    pass


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


def parse_action(text: str, available: set[str] | None = None) -> Action:
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
    if available is not None and tool not in available:
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
# Общие утилиты
# --------------------------------------------------------------------------- #


def digest(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def messages_chars(messages: list[dict[str, str]]) -> int:
    return sum(len(message["content"]) for message in messages)


def _is_tool_result(message: dict[str, str]) -> bool:
    return message["role"] == "user" and message["content"].startswith(
        "Tool result:\n"
    )


def compact_messages(
    messages: list[dict[str, str]],
    *,
    keep_recent: int = 2,
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Заменить старые tool results на bounded-ссылку, сохранив полный текст.

    Возвращает (new_messages, report). Полные результаты лежат в
    ``report["receipts"]``.
    """
    tool_indices = [i for i, m in enumerate(messages) if _is_tool_result(m)]
    keep = set(tool_indices[-keep_recent:]) if keep_recent > 0 else set()
    compacted: list[dict[str, str]] = []
    receipts: dict[str, Any] = {}
    for index, message in enumerate(messages):
        if index in keep or not _is_tool_result(message):
            compacted.append(dict(message))
            continue
        original = message["content"]
        receipt_id = f"r{index}"
        receipts[receipt_id] = {
            "chars": len(original),
            "sha256": digest(original),
            "content": original,
        }
        bounded = (
            f"Tool result:\n[compacted {receipt_id}: "
            f"{len(original)} chars, sha256={digest(original)[:16]}...]"
        )
        compacted.append({"role": "user", "content": bounded})
    before = messages_chars(messages)
    after = messages_chars(compacted)
    return compacted, {
        "before_chars": before,
        "after_chars": after,
        "saved_chars": before - after,
        "compacted": len(receipts),
        "receipts": receipts,
    }


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #


@dataclass
class Criterion:
    name: str
    check: Callable[[dict[str, Any]], bool]


def evaluate_run(facts: dict[str, Any], criteria: list[Criterion]) -> dict[str, Any]:
    results = []
    for criterion in criteria:
        try:
            passed = bool(criterion.check(facts))
        except Exception as error:  # noqa: BLE001 - evaluation shows the reason
            results.append(
                {"name": criterion.name, "passed": False, "reason": str(error)}
            )
            continue
        results.append({"name": criterion.name, "passed": passed})
    passed = all(r["passed"] for r in results)
    return {"passed": passed, "results": results}


# --------------------------------------------------------------------------- #
# Checkpoint / branching
# --------------------------------------------------------------------------- #


def save_checkpoint(
    path: Path,
    *,
    step: int,
    messages: list[dict[str, str]],
    receipts: dict[str, Any] | None = None,
) -> None:
    payload = {
        "step": step,
        "messages": messages,
        "receipts": receipts or {},
    }
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_checkpoint(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def branch_checkpoint(base: dict[str, Any], instruction: str) -> dict[str, Any]:
    messages = [dict(message) for message in base["messages"]]
    messages.append({"role": "user", "content": f"Operator instruction:\n{instruction}"})
    return {
        "step": base.get("step", 0),
        "messages": messages,
        "receipts": dict(base.get("receipts", {})),
        "instruction": instruction,
    }


def select_branch(
    branches: dict[str, dict[str, Any]]
) -> tuple[str | None, list[str]]:
    """Вернуть первый прошедший branch и список всех прошедших."""
    passed = [name for name, result in branches.items() if result.get("passed")]
    return (passed[0] if passed else None, passed)


# --------------------------------------------------------------------------- #
# Bounded evidence
# --------------------------------------------------------------------------- #


class EvidenceStore:
    """Хранит полные bytes вне prompt, отдаёт bounded observation и range."""

    def __init__(self, directory: Path | None = None):
        self.directory = directory
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
        self._data: dict[str, bytes] = {}

    def put(self, name: str, data: bytes) -> dict[str, Any]:
        self._data[name] = data
        if self.directory is not None:
            (self.directory / name).write_bytes(data)
        return {
            "name": name,
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    def observe(self, name: str, head: int = 80, tail: int = 80) -> str:
        data = self._data[name]
        text = data.decode("utf-8", errors="replace")
        prefix = text[:head]
        suffix = text[-tail:] if len(text) > head + tail else ""
        return (
            f"evidence {name}: {len(data)} bytes, "
            f"sha256={hashlib.sha256(data).hexdigest()[:16]}...\n"
            f"HEAD:\n{prefix}\n"
            f"TAIL:\n{suffix}"
        )

    def retrieve(self, name: str, start: int, end: int) -> bytes:
        data = self._data[name]
        if start < 0 or end < start or end > len(data):
            raise ValueError(
                f"range [{start}, {end}) outside evidence size {len(data)}"
            )
        return data[start:end]
