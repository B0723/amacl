from __future__ import annotations

import random
from abc import ABC, abstractmethod

from amacl.config import RuntimeConfig
from amacl.registry import SAMPLER_REGISTRY, register_sampler
from amacl.schemas import SeedCase


class SeedSampler(ABC):
    name = "base"
    description = "Base seed sampler"

    @abstractmethod
    def is_match(self, case: SeedCase, config: RuntimeConfig) -> bool:
        raise NotImplementedError

    def candidates(self, cases: list[SeedCase], config: RuntimeConfig) -> list[SeedCase]:
        return [case for case in cases if self.is_match(case, config)]

    def sample(
        self,
        cases: list[SeedCase],
        config: RuntimeConfig,
        rng: random.Random,
    ) -> SeedCase:
        matched = self.candidates(cases, config)
        population = matched or cases
        if not population:
            raise ValueError("Seed pool is empty.")
        return rng.choice(population)


@register_sampler
class HighFreqQuerySampler(SeedSampler):
    name = "high_freq_query"
    description = "采样高频头部 query，主要用于防退化和稳态训练。"

    def is_match(self, case: SeedCase, config: RuntimeConfig) -> bool:
        return case.query_frequency is not None and case.query_frequency >= config.high_freq_threshold


@register_sampler
class TailQuerySampler(SeedSampler):
    name = "tail_query"
    description = "采样长尾或低频 query，主要用于考察泛化能力。"

    def is_match(self, case: SeedCase, config: RuntimeConfig) -> bool:
        return case.query_frequency is not None and case.query_frequency <= config.tail_freq_threshold


@register_sampler
class OnlineHardcaseSampler(SeedSampler):
    name = "online_hardcase"
    description = "采样线上高曝光低点击或已标记 hardcase 的样本。"

    def is_match(self, case: SeedCase, config: RuntimeConfig) -> bool:
        if "online_hardcase" in case.tags or bool(case.meta.get("is_online_hardcase")):
            return True

        if case.exposure is None:
            return False

        ctr = case.ctr
        if ctr is None and case.click is not None and case.exposure > 0:
            ctr = case.click / case.exposure

        return (
            ctr is not None
            and case.exposure >= config.online_hardcase_min_exposure
            and ctr <= config.online_hardcase_max_ctr
        )


@register_sampler
class BoundaryConfidenceSampler(SeedSampler):
    name = "boundary_confidence"
    description = "采样 student 置信度位于边界区间的样本。"

    def is_match(self, case: SeedCase, config: RuntimeConfig) -> bool:
        if case.student_confidence is None:
            return False
        return config.boundary_low <= case.student_confidence <= config.boundary_high


def build_sampler_instances() -> dict[str, SeedSampler]:
    return {name: sampler_cls() for name, sampler_cls in SAMPLER_REGISTRY.items()}


def build_sampler_summaries(cases: list[SeedCase], config: RuntimeConfig) -> list[dict[str, str | int]]:
    summaries: list[dict[str, str | int]] = []
    for sampler in build_sampler_instances().values():
        summaries.append(
            {
                "name": sampler.name,
                "description": sampler.description,
                "candidate_count": len(sampler.candidates(cases, config)),
            }
        )
    return summaries
