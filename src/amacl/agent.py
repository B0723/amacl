from __future__ import annotations

import random
from pathlib import Path

from amacl.config import RuntimeConfig
from amacl.operators import MutationOperator, build_operator_catalogue
from amacl.planner import MutationPlanner
from amacl.schemas import GeneratedCase, SeedCase
from amacl.utils import dump_jsonl, load_seed_pool_jsonl


class ColdStartAgent:
    """冷启动版出题 Agent。

    无需外部反馈（无 Oracle 打标、无 Reward 计算）。
    流程：
      1. 从高频 seed pool JSONL 随机采样 N 条
      2. 对每条 SeedCase 用全部 8 种算子各改写一次，共生成 N×8 条变异样本
      3. 将原始样本和变异结果写到 data_mutation 目录
    """

    def __init__(self, mutation_planner: MutationPlanner, config: RuntimeConfig):
        self.mutation_planner = mutation_planner
        self.config = config
        self.rng = random.Random(config.random_seed)
        self.operators = build_operator_catalogue()

    # ------------------------------------------------------------------
    # 主入口：加载 pool → 采样 → 批量 mutation → 写出
    # ------------------------------------------------------------------

    def run(
        self,
        seed_pool_file: str | None = None,
        seed_pool_name: str = "high_frequency",
        output_file: str | None = None,
    ) -> list[GeneratedCase]:
        """执行冷启动出题流程。

        Args:
            seed_pool_file: 预处理后的 JSONL 文件路径；默认使用 config.high_freq_seed_file。
            seed_pool_name: pool 名称标识，记录到输出字段 seed_pool。
            output_file:    输出 JSONL 文件路径；默认在 data_mutation_dir 下生成。

        Returns:
            生成的 GeneratedCase 列表（每条 seed 对应 8 条，共 N×8 条）。
        """
        pool_path = seed_pool_file or self.config.high_freq_seed_file

        # Step 1: 从 seed pool 随机采样
        cases = load_seed_pool_jsonl(
            path=pool_path,
            sample_n=self.config.cold_start_sample_n,
            rng=self.rng,
        )
        if not cases:
            raise ValueError(f"Seed pool 为空：{pool_path}")

        print(f"[ColdStartAgent] 从 {pool_path} 采样 {len(cases)} 条 seed，"
              f"每条使用 {len(self.operators)} 种算子，共生成 {len(cases) * len(self.operators)} 条变异样本")

        # Step 2: 对每条 seed 用全部算子各做一次 mutation
        results: list[GeneratedCase] = []
        total = len(cases) * len(self.operators)
        done = 0
        for seed_idx, seed in enumerate(cases, 1):
            for op in self.operators:
                done += 1
                print(
                    f"[ColdStartAgent] [{done}/{total}] "
                    f"seed {seed_idx}/{len(cases)} query={seed.query!r} "
                    f"算子={op.name}"
                )
                generated = self._mutate_with_operator(seed, op, seed_pool_name)
                results.append(generated)

        # Step 3: 写出到 data_mutation 目录
        if output_file is None:
            output_dir = Path(self.config.data_mutation_dir)
            output_file = str(output_dir / f"{seed_pool_name}_cold_start.jsonl")

        dump_jsonl(output_file, [item.to_dict() for item in results])
        print(f"[ColdStartAgent] 已生成 {len(results)} 条变异样本 → {output_file}")

        return results

    # ------------------------------------------------------------------
    # 单条 seed × 单个算子 → 一条 GeneratedCase
    # ------------------------------------------------------------------

    def _mutate_with_operator(
        self,
        seed: SeedCase,
        operator: MutationOperator,
        seed_pool_name: str,
    ) -> GeneratedCase:
        """用指定算子对 seed 进行变异，返回一条 GeneratedCase。"""
        draft = self.mutation_planner.generate_with_operator(seed, operator, self.operators)

        return GeneratedCase(
            seed_id=seed.seed_id,
            seed_pool=seed_pool_name,
            operator_name=draft.operator_name,
            target_field=draft.target_field,
            expected_effect=draft.expected_effect,
            rationale=draft.rationale,
            original_query=seed.query,
            original_doc=seed.doc,
            mutated_query=draft.mutated_query,
            mutated_doc=draft.mutated_doc,
            label=seed.label,
            tags=list(seed.tags),
            meta={
                **seed.meta,
                "poi_id": seed.poi_id or "",
                "poi_name": seed.poi_name or "",
                "first_cate": seed.first_cate or "",
                "second_cate": seed.second_cate or "",
                "top_dishes": seed.top_dishes or "",
                "mutation_raw_response": draft.raw_response,
            },
        )
