from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import SeedCase


def load_seed_cases(path: str | Path) -> list[SeedCase]:
    cases: list[SeedCase] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            cases.append(SeedCase.from_dict(json.loads(line)))
    return cases


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
