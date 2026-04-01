from __future__ import annotations

import random

from .config import RuntimeConfig
from .operators import build_operator_catalogue
from .planner import MutationPlanner, StrategyPlanner
from .schemas import GeneratedCase, SeedCase
from .seed_samplers import build_sampler_instances


class QuestionGenerationAgent:
    def __init__(self, strategy_planner: StrategyPlanner, mutation_planner: MutationPlanner, config: RuntimeConfig):
        self.strategy_planner = strategy_planner
        self.mutation_planner = mutation_planner
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.samplers = build_sampler_instances()
        self.operators = build_operator_catalogue()

    def generate_one(self, cases: list[SeedCase]) -> GeneratedCase:
        if not cases:
            raise ValueError("Seed pool is empty.")

        sampler_decision = self.strategy_planner.choose(cases)
        sampler = self.samplers[sampler_decision.sampler_name]
        seed = sampler.sample(cases, self.config, self.rng)
        draft = self.mutation_planner.generate(seed, self.operators)

        return GeneratedCase(
            seed_id=seed.seed_id,
            sampler_name=sampler_decision.sampler_name,
            operator_name=draft.operator_name,
            expected_effect=draft.expected_effect,
            rationale=draft.rationale or sampler_decision.reason,
            original_query=seed.query,
            original_doc=seed.doc,
            mutated_query=draft.mutated_query,
            mutated_doc=draft.mutated_doc,
            label=seed.label,
            tags=seed.tags,
            meta={
                **seed.meta,
                "sampler_reason": sampler_decision.reason,
                "sampler_raw_response": sampler_decision.raw_response,
                "mutation_raw_response": draft.raw_response,
                "target_field": draft.target_field,
            },
        )

    def generate_many(self, cases: list[SeedCase], num_samples: int) -> list[GeneratedCase]:
        return [self.generate_one(cases) for _ in range(num_samples)]
