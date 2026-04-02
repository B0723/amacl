from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from amacl.schemas import SeedCase


def load_seed_cases(path: str | Path) -> list[SeedCase]:
    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        return []

    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        payload = _load_concatenated_json_objects(content)

    if isinstance(payload, dict):
        return [SeedCase.from_dict(payload)]

    if isinstance(payload, list):
        return [SeedCase.from_dict(item) for item in payload]

    raise ValueError("Seed file must contain a JSON object, a JSON array, or concatenated JSON objects.")


def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON object found in model output.")

    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                payload = text[start : index + 1]
                return json.loads(payload)
    raise ValueError("Incomplete JSON object in model output.")


def _load_concatenated_json_objects(content: str) -> list[dict[str, Any]]:
    decoder = json.JSONDecoder()
    cursor = 0
    payloads: list[dict[str, Any]] = []

    while cursor < len(content):
        while cursor < len(content) and content[cursor].isspace():
            cursor += 1

        if cursor >= len(content):
            break

        item, next_cursor = decoder.raw_decode(content, cursor)
        if not isinstance(item, dict):
            raise ValueError("Concatenated JSON input must contain JSON objects.")
        payloads.append(item)
        cursor = next_cursor

    return payloads