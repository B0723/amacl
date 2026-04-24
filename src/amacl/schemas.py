from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SeedCase:
    seed_id: str
    query: str
    doc: str
    label: int | str
    # 美团 O2O 业务字段
    poi_id: str | None = None
    poi_name: str | None = None
    first_cate: str | None = None
    second_cate: str | None = None
    top_deals: list[str] = field(default_factory=list)   # 团单标题列表
    top_dishes: str | None = None                         # POITOPKDISHES 原始字符串
    # 通用字段（保留兼容性）
    query_frequency: int | None = None
    exposure: int | None = None
    click: int | None = None
    ctr: float | None = None
    student_confidence: float | None = None
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 从美团 CSV→JSON 格式解析
    # 字段来源：globalid / query / poiid / feature / relevancegrade
    # ------------------------------------------------------------------
    @classmethod
    def from_meituan_dict(cls, payload: dict[str, Any]) -> "SeedCase":
        seed_id = str(payload.get("globalid", payload.get("seed_id", ""))).strip() or "unknown_seed"
        query = str(payload.get("query", "")).strip()
        poi_id = str(payload.get("poiid", "")).strip()

        # 解析 feature 字段（可能是 JSON 字符串，也可能已经是 dict）
        feature_raw = payload.get("feature", {})
        if isinstance(feature_raw, str):
            try:
                feature = json.loads(feature_raw)
            except json.JSONDecodeError:
                feature = {}
        else:
            feature = feature_raw if isinstance(feature_raw, dict) else {}

        # 解析 POIINFO（可能是嵌套 JSON 字符串）
        poi_info_raw = feature.get("POIINFO", {})
        if isinstance(poi_info_raw, str):
            try:
                poi_info = json.loads(poi_info_raw)
            except json.JSONDecodeError:
                poi_info = {}
        else:
            poi_info = poi_info_raw if isinstance(poi_info_raw, dict) else {}

        poi_name = str(poi_info.get("poiName", "")).strip()
        first_cate = str(poi_info.get("firstBackCateName", "")).strip() or None
        second_cate = str(poi_info.get("secondBackCateName", "")).strip() or None

        # 解析 POI2TOPKDEALSMAP，提取团单标题列表
        top_deals: list[str] = []
        deals_raw = feature.get("POI2TOPKDEALSMAP", "")
        if isinstance(deals_raw, str) and deals_raw:
            # 格式：多个 JSON 对象拼接，每个含 dealInfo
            _extract_deal_infos(deals_raw, top_deals)
        elif isinstance(deals_raw, list):
            for item in deals_raw:
                if isinstance(item, dict) and item.get("dealInfo"):
                    top_deals.append(str(item["dealInfo"]).strip())

        # 构建 doc：POI 名称 + 最多 2 条团单标题
        doc_parts = [poi_name] if poi_name else []
        doc_parts.extend(top_deals[:2])
        doc = " | ".join(p for p in doc_parts if p) or poi_name or poi_id

        top_dishes = str(feature.get("POITOPKDISHES", "")).strip() or None

        # 相关性标签：优先使用 relevancegrade，退而使用 label
        label_raw = payload.get("relevancegrade", payload.get("label", 0))
        label = _maybe_int(label_raw) or 0

        return cls(
            seed_id=seed_id,
            query=query,
            doc=doc,
            label=label,
            poi_id=poi_id or None,
            poi_name=poi_name or None,
            first_cate=first_cate,
            second_cate=second_cate,
            top_deals=top_deals,
            top_dishes=top_dishes,
            tags=list(payload.get("tags", [])),
            meta={
                "biz_scene": "search",
                "requestid": str(payload.get("requestid", "")),
                "strategy": str(payload.get("strategy", "")),
                "cityid": str(payload.get("cityid", "")),
                "raw_relevance": str(payload.get("relevance", "")),
                "adrelevancegrade": str(payload.get("adrelevancegrade", "")),
            },
        )

    # ------------------------------------------------------------------
    # 从旧版 seed_pool.jsonl 格式解析（保留兼容性）
    # ------------------------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SeedCase":
        return cls(
            seed_id=str(payload.get("seed_id", payload.get("id", ""))).strip() or "unknown_seed",
            query=str(payload["query"]),
            doc=str(payload["doc"]),
            label=payload.get("label", 0),
            query_frequency=_maybe_int(payload.get("query_frequency")),
            exposure=_maybe_int(payload.get("exposure")),
            click=_maybe_int(payload.get("click")),
            ctr=_maybe_float(payload.get("ctr")),
            student_confidence=_maybe_float(payload.get("student_confidence")),
            tags=list(payload.get("tags", [])),
            meta=dict(payload.get("meta", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class MutationDraft:
    operator_name: str
    target_field: str
    expected_effect: str
    rationale: str
    mutated_query: str
    mutated_doc: str
    raw_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class GeneratedCase:
    seed_id: str
    seed_pool: str                  # 来源 seed pool 名称，例如 "high_frequency"
    operator_name: str              # 变异算子名称（mutation type）
    target_field: str               # 被变异的字段：query / doc / both
    expected_effect: str            # 预期效果：keep_label / decrease_relevance / increase_relevance
    rationale: str                  # Agent 的变异理由
    original_query: str
    original_doc: str
    mutated_query: str
    mutated_doc: str
    label: int | str                # 原始标签（Oracle 打标前保留原值）
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# 旧版 SamplerDecision 保留供测试兼容
# ---------------------------------------------------------------------------
@dataclass(slots=True)
class SamplerDecision:
    sampler_name: str
    reason: str
    raw_response: str = ""


# ---------------------------------------------------------------------------
# 私有辅助函数
# ---------------------------------------------------------------------------

def _extract_deal_infos(raw: str, out: list[str]) -> None:
    """从 POI2TOPKDEALSMAP 的拼接 JSON 字符串中提取 dealInfo 列表。"""
    import re
    # 用正则抓取所有 "dealInfo":"..." 的值
    for m in re.finditer(r'"dealInfo"\s*:\s*"([^"]+)"', raw):
        info = m.group(1).strip()
        if info:
            out.append(info)


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None
