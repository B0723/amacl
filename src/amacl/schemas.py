from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class SeedCase:
    seed_id: str
    query: str
    doc: str
    label: int | str
    query_frequency: int | None = None
    exposure: int | None = None
    click: int | None = None
    ctr: float | None = None
    student_confidence: float | None = None
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

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
class SamplerDecision:
    sampler_name: str
    reason: str
    raw_response: str = ""


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
    sampler_name: str
    operator_name: str
    expected_effect: str
    rationale: str
    original_query: str
    original_doc: str
    mutated_query: str
    mutated_doc: str
    label: int | str
    tags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _maybe_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
