from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from amacl.agent import ColdStartAgent
from amacl.config import RuntimeConfig
from amacl.modeling import MockGenerator
from amacl.planner import MutationPlanner


# 最小化的美团格式 JSON seed pool（2 条数据）
MOCK_SEED_POOL = json.dumps([
    {
        "globalid": "seed_001",
        "requestid": "req_001",
        "query": "铁锅炖",
        "poiid": "1397295770",
        "feature": json.dumps({
            "POIINFO": json.dumps({
                "poiId": 1397295770,
                "poiName": "回真楼（南门店）",
                "firstBackCateName": "餐饮",
                "secondBackCateName": "东北菜",
            }),
            "POI2TOPKDEALSMAP": '{"dealId":1624193762,"dealInfo":"3-4人餐木火铁锅炖杂鱼套餐","simScore":0.64}',
            "POITOPKDISHES": "公鸡蛋,土鸡,牛栏山",
        }),
        "relevancegrade": 3,
    },
    {
        "globalid": "seed_002",
        "requestid": "req_002",
        "query": "火锅",
        "poiid": "9876543210",
        "feature": json.dumps({
            "POIINFO": json.dumps({
                "poiId": 9876543210,
                "poiName": "海底捞火锅（望京店）",
                "firstBackCateName": "餐饮",
                "secondBackCateName": "火锅",
            }),
            "POI2TOPKDEALSMAP": '{"dealId":11111,"dealInfo":"双人套餐（锅底+牛肉+虾滑）","simScore":0.81}',
            "POITOPKDISHES": "毛肚,鸭血,宽粉",
        }),
        "relevancegrade": 3,
    },
], ensure_ascii=False)

MOCK_MUTATION_RESPONSE = json.dumps({
    "operator_name": "brand_confusion",
    "target_field": "doc",
    "expected_effect": "decrease_relevance",
    "rationale": "将东北菜商家替换为烤串商家，制造品牌混淆负例",
    "mutated_query": "铁锅炖",
    "mutated_doc": "烤天下烤串（望京店）| 双人烤串套餐（烤羊排+牛肉）",
}, ensure_ascii=False)


class ColdStartAgentTest(unittest.TestCase):
    def _make_agent(self, responses: list[str]) -> tuple[ColdStartAgent, RuntimeConfig]:
        generator = MockGenerator(responses=responses)
        config = RuntimeConfig(random_seed=42, cold_start_sample_n=2)
        mutation_planner = MutationPlanner(generator)
        agent = ColdStartAgent(mutation_planner, config)
        return agent, config

    def test_run_produces_correct_output(self) -> None:
        """end-to-end: 从临时 JSON pool 采样 2 条，生成 2 条变异样本，写出到临时目录。"""
        # MockGenerator 需要为每条 seed 返回一次 mutation 响应
        agent, config = self._make_agent([MOCK_MUTATION_RESPONSE, MOCK_MUTATION_RESPONSE])

        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = Path(tmpdir) / "high_frequency.json"
            pool_path.write_text(MOCK_SEED_POOL, encoding="utf-8")
            output_path = Path(tmpdir) / "mutation_output.jsonl"

            # 覆盖 data_mutation_dir 使用临时目录
            config.data_mutation_dir = tmpdir

            results = agent.run(
                seed_pool_file=str(pool_path),
                seed_pool_name="high_frequency",
                output_file=str(output_path),
            )

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].seed_pool, "high_frequency")
        self.assertEqual(results[0].operator_name, "brand_confusion")
        self.assertIn(results[0].original_query, ("铁锅炖", "火锅"))

    def test_generated_case_has_mutation_type(self) -> None:
        """生成的样本必须记录 operator_name（变异类型）字段。"""
        agent, config = self._make_agent([MOCK_MUTATION_RESPONSE])

        with tempfile.TemporaryDirectory() as tmpdir:
            pool_path = Path(tmpdir) / "high_frequency.json"
            # 只写 1 条 seed
            pool_path.write_text(
                json.dumps([json.loads(MOCK_SEED_POOL)[0]], ensure_ascii=False),
                encoding="utf-8",
            )
            config.cold_start_sample_n = 1
            config.data_mutation_dir = tmpdir

            results = agent.run(
                seed_pool_file=str(pool_path),
                seed_pool_name="high_frequency",
            )

        case = results[0]
        self.assertIsNotNone(case.operator_name)
        self.assertNotEqual(case.operator_name, "")
        # to_dict 序列化后应包含 seed_pool 和 operator_name
        d = case.to_dict()
        self.assertIn("seed_pool", d)
        self.assertIn("operator_name", d)
        self.assertIn("target_field", d)
        self.assertIn("expected_effect", d)

    def test_from_meituan_dict_parses_poi_fields(self) -> None:
        """SeedCase.from_meituan_dict 应正确解析美团格式 JSON。"""
        from amacl.schemas import SeedCase

        row = json.loads(MOCK_SEED_POOL)[0]
        case = SeedCase.from_meituan_dict(row)

        self.assertEqual(case.seed_id, "seed_001")
        self.assertEqual(case.query, "铁锅炖")
        self.assertEqual(case.poi_name, "回真楼（南门店）")
        self.assertEqual(case.second_cate, "东北菜")
        self.assertEqual(case.label, 3)
        self.assertIn("3-4人餐木火铁锅炖杂鱼套餐", case.doc)


if __name__ == "__main__":
    unittest.main()
