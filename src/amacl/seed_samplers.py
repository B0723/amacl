"""seed_samplers.py

冷启动阶段不再需要 Agent 动态选择 seed 策略。
Seed pool 的划分（高频/长尾/边界/在线难例）由离线数据处理流程决定，
直接对应 data/ 目录下的不同 JSON 文件（high_frequency.json 等）。

本模块保留 SeedSampler 基类和各采样器实现，供后续引入 ZPD Reward 的
交替迭代训练阶段使用（届时可能需要在一个 pool 内部按置信度二次过滤）。
"""

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
    description = "高频头部 query 采样器，对应 high_frequency.json。"

    def is_match(self, case: SeedCase, config: RuntimeConfig) -> bool:
        # 冷启动阶段：数据文件本身已经是高频 pool，直接全量匹配
        return True


@register_sampler
class BoundaryConfidenceSampler(SeedSampler):
    name = "boundary_confidence"
    description = "Student 置信度位于边界区间的样本（用于后续迭代阶段）。"

    def is_match(self, case: SeedCase, config: RuntimeConfig) -> bool:
        if case.student_confidence is None:
            return False
        return config.boundary_low <= case.student_confidence <= config.boundary_high


def build_sampler_instances() -> dict[str, SeedSampler]:
    return {name: sampler_cls() for name, sampler_cls in SAMPLER_REGISTRY.items()}
