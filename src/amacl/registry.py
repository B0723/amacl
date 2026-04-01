from __future__ import annotations

from typing import TypeVar


SamplerType = TypeVar("SamplerType")
OperatorType = TypeVar("OperatorType")

SAMPLER_REGISTRY: dict[str, type] = {}
OPERATOR_REGISTRY: dict[str, type] = {}


def register_sampler(cls: SamplerType) -> SamplerType:
    SAMPLER_REGISTRY[cls.name] = cls
    return cls


def register_operator(cls: OperatorType) -> OperatorType:
    OPERATOR_REGISTRY[cls.name] = cls
    return cls
