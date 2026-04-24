from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from amacl.schemas import SeedCase


# ---------------------------------------------------------------------------
# 预处理后 JSONL 文件加载（{query, doc} 格式，由 preprocess_csv_to_jsonl.py 生成）
# ---------------------------------------------------------------------------

def load_seed_pool_jsonl(
    path: str | Path,
    sample_n: int | None = None,
    rng: random.Random | None = None,
) -> list[SeedCase]:
    """从预处理后的 JSONL 文件加载 SeedCase 列表。

    文件格式：每行一个 JSON 对象，包含 query 和 doc 字段。
    由 scripts/preprocess_csv_to_jsonl.py 生成。

    Args:
        path:     JSONL 文件路径。
        sample_n: 若指定，则从全量数据中随机抽取 sample_n 条；None 表示全量返回。
        rng:      随机数生成器；为 None 时使用全局 random。

    Returns:
        SeedCase 列表。
    """
    rows: list[dict] = []
    with Path(path).open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL 解析失败（第 {line_no} 行）：{e}") from e

    if not rows:
        return []

    cases: list[SeedCase] = []
    for idx, row in enumerate(rows):
        query = str(row.get("query", "")).strip()
        doc = str(row.get("doc", "")).strip()
        if not query or not doc:
            continue
        cases.append(
            SeedCase(
                seed_id=str(idx),
                query=query,
                doc=doc,
                label=row.get("label", 0),
            )
        )

    if sample_n is not None and sample_n < len(cases):
        _rng = rng if rng is not None else random.Random()
        cases = _rng.sample(cases, sample_n)

    return cases


# ---------------------------------------------------------------------------
# 美团格式 JSON 文件加载（高频/长尾/边界等 seed pool）
# ---------------------------------------------------------------------------

def load_meituan_seed_pool(
    path: str | Path,
    sample_n: int | None = None,
    rng: random.Random | None = None,
) -> list[SeedCase]:
    """从美团格式 JSON 文件加载 SeedCase 列表。

    文件格式：JSON 数组，每个元素为一行 CSV 对应的字段字典，包含
    globalid / query / poiid / feature / relevancegrade 等字段。

    Args:
        path:     JSON 文件路径。
        sample_n: 若指定，则从全量数据中随机抽取 sample_n 条；None 表示全量返回。
        rng:      随机数生成器；为 None 时使用全局 random。

    Returns:
        SeedCase 列表。
    """
    content = Path(path).read_text(encoding="utf-8").strip()
    if not content:
        return []

    payload = json.loads(content)
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        raise ValueError(f"JSON 文件格式不符：{path}")

    cases = [SeedCase.from_meituan_dict(row) for row in rows]

    if sample_n is not None and sample_n < len(cases):
        _rng = rng if rng is not None else random.Random()
        cases = _rng.sample(cases, sample_n)

    return cases


# ---------------------------------------------------------------------------
# 旧版 seed_pool.jsonl 加载（保留兼容性）
# ---------------------------------------------------------------------------

def load_seed_cases(path: str | Path) -> list[SeedCase]:
    """加载旧版 seed_pool.jsonl 文件（每行一个 JSON 对象，或单个 JSON 对象/数组）。"""
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


# ---------------------------------------------------------------------------
# 写出工具
# ---------------------------------------------------------------------------

def dump_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """将字典列表写为 JSONL 文件（每行一个 JSON 对象）。"""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# JSON 提取工具（供 planner 解析 LLM 输出）
# ---------------------------------------------------------------------------

def extract_json_object(text: str) -> dict[str, Any]:
    """从 LLM 输出文本中提取第一个完整 JSON 对象。"""
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


# ---------------------------------------------------------------------------
# 私有：拼接多对象 JSONL 解析
# ---------------------------------------------------------------------------

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
