from __future__ import annotations

import unittest

from amacl.agent import QuestionGenerationAgent
from amacl.config import RuntimeConfig
from amacl.modeling import MockGenerator
from amacl.planner import MutationPlanner, StrategyPlanner
from amacl.schemas import SeedCase


class QuestionGenerationAgentTest(unittest.TestCase):
    def test_generate_one(self) -> None:
        generator = MockGenerator(
            responses=[
                '{"sampler_name":"boundary_confidence","reason":"当前边界样本更适合出题"}',
                '{"operator_name":"brand_confusion","target_field":"doc","expected_effect":"decrease_relevance","rationale":"制造品牌混淆负例","mutated_query":"苹果手机","mutated_doc":"华为 Mate 60 Pro"}',
            ]
        )
        config = RuntimeConfig(random_seed=7)
        strategy_planner = StrategyPlanner(generator, config)
        mutation_planner = MutationPlanner(generator)
        agent = QuestionGenerationAgent(strategy_planner, mutation_planner, config)

        cases = [
            SeedCase(seed_id="a", query="苹果手机", doc="iPhone 15 Pro", label=1, student_confidence=0.51),
            SeedCase(seed_id="b", query="火锅", doc="重庆火锅双人餐", label=1, query_frequency=5000),
        ]
        generated = agent.generate_one(cases)

        self.assertEqual(generated.sampler_name, "boundary_confidence")
        self.assertEqual(generated.operator_name, "brand_confusion")
        self.assertEqual(generated.mutated_doc, "华为 Mate 60 Pro")


if __name__ == "__main__":
    unittest.main()
