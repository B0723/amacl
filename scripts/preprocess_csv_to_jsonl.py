"""preprocess_csv_to_jsonl.py

将三种 CSV 格式的 seed pool 转换为统一的 JSONL 格式。

每条 JSONL 样本结构：
    {"query": "...", "doc": "POI名称 | 一级类目 | 二级类目 | 三级类目 | 团单1 | 团单2 | 团单3"}

三级类目只在有效时加入（thirdBackCateId > 0 且 thirdBackCateName 非空）。
团单取 top_3（高频/中低频取 POI2TOPKDEALSMAP 的 dealInfo；模型不确定性取 ranked_deals 前3）。

用法：
    python scripts/preprocess_csv_to_jsonl.py \
        --high-freq   data/high_frequency_seed_pool.csv \
        --medlow      data/medlow_frequency_seed_pool.csv \
        --uncertainty data/model_uncertainty_seed_pool.csv \
        --output-dir  data/
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# 解析辅助
# ---------------------------------------------------------------------------

def _extract_deal_infos_from_feature(feature_raw: str) -> list[str]:
    """从 POI2TOPKDEALSMAP 嵌套字符串中提取 dealInfo 列表（按出现顺序，最多3条）。"""
    deals: list[str] = []
    for m in re.finditer(r'"dealInfo"\s*:\s*"([^"]+)"', feature_raw):
        info = m.group(1).strip()
        if info:
            deals.append(info)
        if len(deals) == 3:
            break
    return deals


def _build_doc(
    poi_name: str,
    first_cate: str,
    second_cate: str,
    third_cate_id: int,
    third_cate_name: str,
    top3_deals: list[str],
) -> str:
    """拼接 doc 字段：POI名称 | 各级类目 | top3团单。"""
    parts: list[str] = []
    if poi_name:
        parts.append(poi_name)
    if first_cate:
        parts.append(first_cate)
    if second_cate:
        parts.append(second_cate)
    # 三级类目：thirdBackCateId > 0 且名称非空才加入
    if third_cate_id > 0 and third_cate_name:
        parts.append(third_cate_name)
    parts.extend(top3_deals)
    return " | ".join(parts)


# ---------------------------------------------------------------------------
# 高频 / 中低频 CSV（feature 字段为嵌套非标准 JSON）
# ---------------------------------------------------------------------------

def _parse_feature_field(feature_raw: str) -> dict:
    """从 feature 字段正则提取 POIINFO 和 POI2TOPKDEALSMAP 信息。"""
    result = {
        "poi_name": "",
        "first_cate": "",
        "second_cate": "",
        "third_cate_id": -1,
        "third_cate_name": "",
        "top3_deals": [],
    }

    # 提取 poiName
    m = re.search(r'"poiName"\s*:\s*"([^"]*)"', feature_raw)
    if m:
        result["poi_name"] = m.group(1).strip()

    # 提取 firstBackCateName
    m = re.search(r'"firstBackCateName"\s*:\s*"([^"]*)"', feature_raw)
    if m:
        result["first_cate"] = m.group(1).strip()

    # 提取 secondBackCateName
    m = re.search(r'"secondBackCateName"\s*:\s*"([^"]*)"', feature_raw)
    if m:
        result["second_cate"] = m.group(1).strip()

    # 提取 thirdBackCateId
    m = re.search(r'"thirdBackCateId"\s*:\s*(-?\d+)', feature_raw)
    if m:
        result["third_cate_id"] = int(m.group(1))

    # 提取 thirdBackCateName
    m = re.search(r'"thirdBackCateName"\s*:\s*"([^"]*)"', feature_raw)
    if m:
        result["third_cate_name"] = m.group(1).strip()

    # 提取 top3 团单
    result["top3_deals"] = _extract_deal_infos_from_feature(feature_raw)

    return result


def iter_feature_csv(path: Path) -> Iterator[dict]:
    """逐行读取高频/中低频 CSV，每行 yield {query, doc}。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = (row.get("query") or "").strip()
            if not query:
                continue

            feature_raw = row.get("feature") or ""
            info = _parse_feature_field(feature_raw)

            doc = _build_doc(
                poi_name=info["poi_name"],
                first_cate=info["first_cate"],
                second_cate=info["second_cate"],
                third_cate_id=info["third_cate_id"],
                third_cate_name=info["third_cate_name"],
                top3_deals=info["top3_deals"],
            )
            if not doc:
                continue

            yield {"query": query, "doc": doc}


# ---------------------------------------------------------------------------
# 模型不确定性 CSV（结构化列，deals 为分号分隔字符串）
# ---------------------------------------------------------------------------

def iter_uncertainty_csv(path: Path) -> Iterator[dict]:
    """逐行读取 model_uncertainty CSV，每行 yield {query, doc}。"""
    with path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = (row.get("query") or "").strip()
            if not query:
                continue

            poi_name = (row.get("mt_poi_name") or "").strip()
            first_cate = (row.get("mt_poi_first_cate_name") or "").strip()
            second_cate = (row.get("mt_poi_second_cate_name") or "").strip()
            third_cate_name = (row.get("mt_poi_third_cate_name") or "").strip()
            # model_uncertainty 没有 thirdBackCateId，用名称是否非空判断
            third_cate_id = 1 if third_cate_name else -1

            # ranked_deals 为分号分隔，取前3
            ranked_deals_raw = row.get("ranked_deals") or ""
            top3_deals = [d.strip() for d in ranked_deals_raw.split(";") if d.strip()][:3]

            doc = _build_doc(
                poi_name=poi_name,
                first_cate=first_cate,
                second_cate=second_cate,
                third_cate_id=third_cate_id,
                third_cate_name=third_cate_name,
                top3_deals=top3_deals,
            )
            if not doc:
                continue

            yield {"query": query, "doc": doc}


# ---------------------------------------------------------------------------
# 写出 JSONL
# ---------------------------------------------------------------------------

def write_jsonl(records: Iterator[dict], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
            if count % 50000 == 0:
                print(f"  已写出 {count} 条...", flush=True)
    return count


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="将美团 seed pool CSV 转换为 JSONL 格式")
    parser.add_argument("--high-freq",   default=None, help="high_frequency_seed_pool.csv 路径")
    parser.add_argument("--medlow",      default=None, help="medlow_frequency_seed_pool.csv 路径")
    parser.add_argument("--uncertainty", default=None, help="model_uncertainty_seed_pool.csv 路径")
    parser.add_argument("--output-dir",  default="data", help="输出目录（默认：data/）")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    tasks = []

    if args.high_freq:
        tasks.append((Path(args.high_freq), "high_frequency", iter_feature_csv))
    if args.medlow:
        tasks.append((Path(args.medlow), "medlow_frequency", iter_feature_csv))
    if args.uncertainty:
        tasks.append((Path(args.uncertainty), "model_uncertainty", iter_uncertainty_csv))

    if not tasks:
        print("未指定任何输入文件，退出。", file=sys.stderr)
        sys.exit(1)

    for csv_path, name, iter_fn in tasks:
        if not csv_path.exists():
            print(f"[跳过] 文件不存在：{csv_path}", file=sys.stderr)
            continue
        output_path = output_dir / f"{name}_seed_pool.jsonl"
        print(f"[{name}] 开始处理 {csv_path} → {output_path}")
        count = write_jsonl(iter_fn(csv_path), output_path)
        print(f"[{name}] 完成，共写出 {count} 条\n")


if __name__ == "__main__":
    main()
