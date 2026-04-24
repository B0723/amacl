from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from amacl.config import RuntimeConfig
from amacl.schemas import SeedCase
from amacl.utils import load_meituan_seed_pool, load_seed_cases


# ---------------------------------------------------------------------------
# 美团格式测试数据
# ---------------------------------------------------------------------------
MEITUAN_POOL = [
    {
        "globalid": "g001",
        "query": "铁锅炖",
        "poiid": "111",
        "feature": json.dumps({
            "POIINFO": json.dumps({
                "poiId": 111,
                "poiName": "回真楼",
                "firstBackCateName": "餐饮",
                "secondBackCateName": "东北菜",
            }),
            "POI2TOPKDEALSMAP": '{"dealId":1,"dealInfo":"铁锅炖大鹅3-4人餐","simScore":0.7}',
            "POITOPKDISHES": "公鸡蛋,花卷",
        }),
        "relevancegrade": 3,
    },
    {
        "globalid": "g002",
        "query": "火锅",
        "poiid": "222",
        "feature": json.dumps({
            "POIINFO": json.dumps({
                "poiId": 222,
                "poiName": "海底捞",
                "firstBackCateName": "餐饮",
                "secondBackCateName": "火锅",
            }),
            "POI2TOPKDEALSMAP": '{"dealId":2,"dealInfo":"双人套餐","simScore":0.9}',
            "POITOPKDISHES": "毛肚,鸭血",
        }),
        "relevancegrade": 3,
    },
]


class LoadMeituanSeedPoolTest(unittest.TestCase):
    def test_load_all(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "high_frequency.json"
            path.write_text(json.dumps(MEITUAN_POOL, ensure_ascii=False), encoding="utf-8")
            cases = load_meituan_seed_pool(path)

        self.assertEqual(len(cases), 2)
        self.assertEqual(cases[0].seed_id, "g001")
        self.assertEqual(cases[0].query, "铁锅炖")
        self.assertEqual(cases[0].poi_name, "回真楼")
        self.assertEqual(cases[0].second_cate, "东北菜")
        self.assertIn("铁锅炖大鹅3-4人餐", cases[0].doc)
        self.assertEqual(cases[0].label, 3)

    def test_sample_n(self) -> None:
        import random as _random
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "high_frequency.json"
            path.write_text(json.dumps(MEITUAN_POOL, ensure_ascii=False), encoding="utf-8")
            rng = _random.Random(42)
            cases = load_meituan_seed_pool(path, sample_n=1, rng=rng)

        self.assertEqual(len(cases), 1)

    def test_sample_n_larger_than_pool(self) -> None:
        """sample_n >= pool 大小时返回全量数据，不报错。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "high_frequency.json"
            path.write_text(json.dumps(MEITUAN_POOL, ensure_ascii=False), encoding="utf-8")
            cases = load_meituan_seed_pool(path, sample_n=100)

        self.assertEqual(len(cases), 2)

    def test_empty_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.json"
            path.write_text("", encoding="utf-8")
            cases = load_meituan_seed_pool(path)
        self.assertEqual(cases, [])


class LegacySeedCaseLoadTest(unittest.TestCase):
    """旧版 load_seed_cases 兼容性测试。"""

    def test_load_multiline_json_object(self) -> None:
        content = """
{
  "seed_id": "case_001",
  "query": "苹果手机",
  "doc": "Apple iPhone 15 Pro",
  "label": 1,
  "meta": {"biz_scene": "search"}
}
""".strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seed.jsonl"
            path.write_text(content, encoding="utf-8")
            cases = load_seed_cases(path)

        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].seed_id, "case_001")
        self.assertEqual(cases[0].meta["biz_scene"], "search")

    def test_load_concatenated_multiline_json_objects(self) -> None:
        content = """
{
  "seed_id": "case_001",
  "query": "苹果手机",
  "doc": "Apple iPhone 15 Pro",
  "label": 1
}
{
  "seed_id": "case_002",
  "query": "火锅",
  "doc": "重庆火锅双人餐",
  "label": 1
}
""".strip()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "seed.jsonl"
            path.write_text(content, encoding="utf-8")
            cases = load_seed_cases(path)

        self.assertEqual([case.seed_id for case in cases], ["case_001", "case_002"])


class SeedCaseFromMeituanDictTest(unittest.TestCase):
    def test_doc_constructed_from_poi_and_deals(self) -> None:
        """doc 应由 poi_name + top_deals 拼接构成。"""
        row = MEITUAN_POOL[0]
        case = SeedCase.from_meituan_dict(row)
        self.assertIn("回真楼", case.doc)
        self.assertIn("铁锅炖大鹅3-4人餐", case.doc)

    def test_boundary_confidence_sampler_still_works(self) -> None:
        """BoundaryConfidenceSampler 应正确过滤 student_confidence 字段。"""
        from amacl.seed_samplers import build_sampler_instances

        config = RuntimeConfig(boundary_low=0.4, boundary_high=0.6)
        samplers = build_sampler_instances()

        cases = [
            SeedCase(seed_id="a", query="q", doc="d", label=1, student_confidence=0.50),
            SeedCase(seed_id="b", query="q", doc="d", label=1, student_confidence=0.80),
        ]
        matched = samplers["boundary_confidence"].candidates(cases, config)
        self.assertEqual([c.seed_id for c in matched], ["a"])


if __name__ == "__main__":
    unittest.main()
