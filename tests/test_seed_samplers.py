from __future__ import annotations

import unittest

from amacl.config import RuntimeConfig
from amacl.schemas import SeedCase
from amacl.seed_samplers import build_sampler_instances


class SeedSamplerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = RuntimeConfig(
            high_freq_threshold=1000,
            tail_freq_threshold=10,
            boundary_low=0.4,
            boundary_high=0.6,
            online_hardcase_min_exposure=100,
            online_hardcase_max_ctr=0.05,
        )
        self.samplers = build_sampler_instances()
        self.cases = [
            SeedCase(seed_id="head", query="苹果手机", doc="iPhone 15 Pro", label=1, query_frequency=2000),
            SeedCase(seed_id="tail", query="便宜团建烧烤去哪", doc="某某烧烤店", label=1, query_frequency=5),
            SeedCase(seed_id="hard", query="奶茶", doc="奶茶第二杯半价", label=1, exposure=200, click=6),
            SeedCase(seed_id="boundary", query="火锅", doc="重庆老火锅双人餐", label=1, student_confidence=0.55),
        ]

    def test_high_freq_sampler(self) -> None:
        matched = self.samplers["high_freq_query"].candidates(self.cases, self.config)
        self.assertEqual([case.seed_id for case in matched], ["head"])

    def test_tail_sampler(self) -> None:
        matched = self.samplers["tail_query"].candidates(self.cases, self.config)
        self.assertEqual([case.seed_id for case in matched], ["tail"])

    def test_online_hardcase_sampler(self) -> None:
        matched = self.samplers["online_hardcase"].candidates(self.cases, self.config)
        self.assertEqual([case.seed_id for case in matched], ["hard"])

    def test_boundary_sampler(self) -> None:
        matched = self.samplers["boundary_confidence"].candidates(self.cases, self.config)
        self.assertEqual([case.seed_id for case in matched], ["boundary"])


if __name__ == "__main__":
    unittest.main()
