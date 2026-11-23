# SPDX-License-Identifier: Apache-2.0
"""
Эксперимент 04: bounded tool evidence и range retrieval.

Идея
----
Огромный tool result нельзя целиком класть в prompt. Week 4 tiny-llm
предлагает:

* хранить точные bytes вне prompt;
* показывать модели bounded observation: identity, digest, размер,
  head и tail;
* дать модели возможность запросить один явный диапазон bytes;
* проверять диапазон и возвращать ровно запрошенное.

Здесь мы делаем это без LLM: transcript + scripted range request.

Запуск:
    source ./vllm-env/bin/activate
    python "lab/03 context engineering/04_agent_bounded_evidence.py"
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import note, section, subsection

from agent_utils import EvidenceStore, digest, messages_chars


def make_large_evidence() -> bytes:
    lines = []
    for i in range(4000):
        lines.append(
            f'{{"id":{i},"topic":"paged attention","value":{i * 3}}}'
        )
    return ("\n".join(lines) + "\n").encode("utf-8")


def main() -> None:
    section("04. Bounded tool evidence")
    data = make_large_evidence()
    print(f"  исходный tool result: {len(data)} bytes")

    with tempfile.TemporaryDirectory(prefix="evidence-") as tmp:
        store = EvidenceStore(Path(tmp))
        meta = store.put("tool_result.jsonl", data)
        print(
            f"  evidence id: {meta['name']}, bytes={meta['bytes']}, "
            f"sha256={meta['sha256'][:16]}..."
        )
        assert meta["bytes"] == len(data)

        subsection("A. Bounded observation")
        observation = store.observe("tool_result.jsonl", head=80, tail=80)
        print(observation)
        assert len(observation) < 500, len(observation)
        assert "sha256=" in observation and "HEAD:" in observation
        assert len(observation) < len(data)
        note(
            "В prompt попадает только identity + digest + размер + "
            "head/tail. Полные bytes остаются в EvidenceStore."
        )

        subsection("B. Prompt не содержит полный результат")
        messages = [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "Inspect the huge tool result."},
            {"role": "user", "content": "Tool result:\n" + observation},
        ]
        prompt = "\n".join(m["content"] for m in messages)
        print(f"  prompt: {messages_chars(messages)} chars")
        assert messages_chars(messages) < 1000
        assert data.decode("utf-8")[:200] not in prompt
        # head/tail всё же видны.
        assert data.decode("utf-8")[:40] in prompt
        note("Модель видит начало и конец, но не весь массив.")

        subsection("C. Range retrieval")
        # Scripted model просит ровно один диапазон.
        request = json.loads(
            '{"tool":"read_evidence","name":"tool_result.jsonl",'
            '"start":1000,"end":1200}'
        )
        start, end = request["start"], request["end"]
        chunk = store.retrieve(request["name"], start, end)
        print(
            f"  requested [{start}, {end}) -> {len(chunk)} bytes, "
            f"sha256={digest(chunk)[:16]}..."
        )
        assert chunk == data[start:end]
        assert len(chunk) == 200
        print(f"  first 60 chars: {chunk[:60].decode('utf-8')!r}")
        note(
            "Диапазон явный и проверяемый: модель не может попросить "
            "«весь файл», а получает ровно [start, end)."
        )

        subsection("D. Некорректный диапазон отклоняется")
        for bad_start, bad_end in ((-1, 10), (10, 5), (0, len(data) + 1)):
            try:
                store.retrieve("tool_result.jsonl", bad_start, bad_end)
            except ValueError:
                print(f"  ✓ отклонён диапазон [{bad_start}, {bad_end})")
            else:
                raise AssertionError(
                    f"диапазон [{bad_start}, {bad_end}) не был отклонён"
                )

    note(
        "В полном агенте этот механизм встраивается в tool protocol: "
        "модель получает bounded observation и может вызвать read_evidence."
    )
    note(
        "Digest позволяет проверить, что retrieved bytes относятся к тому "
        "же объекту, а receipt — восстановить полный артефакт."
    )

    section("Вывод")
    note("Большие результаты хранятся вне prompt.")
    note("Наблюдение bounded: identity, digest, размер, head/tail.")
    note("Range retrieval отдаёт ровно запрошенные bytes.")
    note("Некорректные диапазоны отклоняются до чтения.")


if __name__ == "__main__":
    main()
