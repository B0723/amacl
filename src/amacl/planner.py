from __future__ import annotations

import json

from amacl.config import RuntimeConfig
from amacl.modeling import BaseTextGenerator
from amacl.operators import MutationOperator
from amacl.schemas import MutationDraft, SamplerDecision, SeedCase
from amacl.seed_samplers import build_sampler_summaries
from amacl.utils import extract_json_object


SAMPLER_SYSTEM_PROMPT = """你是 AMACL 的 seed 采样策略规划器。
你的任务是根据当前 seed pool 的统计信息，先从候选策略中选择最合适的一种。
你只能输出 JSON，不要输出解释性散文。"""


MUTATION_SYSTEM_PROMPT = """你是 AMACL 的出题 agent。
你的任务是为给定的 query-doc seed 选择一个 mutation operator，并直接生成一个变异后的样本。
目标是得到真实、自然、适合搜索相关性训练的样本。
你只能输出 JSON，不要输出解释性散文。"""


class StrategyPlanner:
    def __init__(self, generator: BaseTextGenerator, config: RuntimeConfig):
        self.generator = generator
        self.config = config

    def choose(self, cases: list[SeedCase]) -> SamplerDecision:
        summaries = build_sampler_summaries(cases, self.config)
        prompt = (
            "请从以下 seed sampler 中选择一个作为本轮宏观采样策略。\n"
            f"候选策略:\n{json.dumps(summaries, ensure_ascii=False, indent=2)}\n\n"
            "输出 JSON，格式为：\n"
            '{"sampler_name":"...", "reason":"..."}'
        )
        raw = self.generator.generate(SAMPLER_SYSTEM_PROMPT, prompt)
        payload = extract_json_object(raw)
        sampler_name = str(payload.get("sampler_name", "")).strip()

        valid_names = {item["name"] for item in summaries if int(item["candidate_count"]) > 0}
        if sampler_name not in valid_names:
            sampler_name = _fallback_sampler_name(summaries)

        return SamplerDecision(
            sampler_name=sampler_name,
            reason=str(payload.get("reason", "fallback sampler")),
            raw_response=raw,
        )


class MutationPlanner:
    def __init__(self, generator: BaseTextGenerator):
        self.generator = generator

    def generate(self, seed: SeedCase, operators: list[MutationOperator]) -> MutationDraft:
        prompt = (
            "请基于给定 seed 生成一个更难但仍然自然的训练样本。\n"
            f"seed:\n{json.dumps(seed.to_dict(), ensure_ascii=False, indent=2)}\n\n"
            f"可用 operators:\n{json.dumps([item.to_dict() for item in operators], ensure_ascii=False, indent=2)}\n\n"
            "请严格输出 JSON，格式为：\n"
            '{'
            '"operator_name":"...",'
            '"target_field":"query|doc|both",'
            '"expected_effect":"keep_label|decrease_relevance|increase_relevance",'
            '"rationale":"...",'
            '"mutated_query":"...",'
            '"mutated_doc":"..."'
            '}'
        )
        raw = self.generator.generate(MUTATION_SYSTEM_PROMPT, prompt)
        payload = extract_json_object(raw)
        valid_names = {item.name for item in operators}
        operator_name = str(payload.get("operator_name", "")).strip()
        if operator_name not in valid_names:
            operator_name = operators[0].name

        return MutationDraft(
            operator_name=operator_name,
            target_field=str(payload.get("target_field", "doc")).strip() or "doc",
            expected_effect=str(payload.get("expected_effect", "decrease_relevance")).strip(),
            rationale=str(payload.get("rationale", "")).strip(),
            mutated_query=str(payload.get("mutated_query", seed.query)).strip(),
            mutated_doc=str(payload.get("mutated_doc", seed.doc)).strip(),
            raw_response=raw,
        )


def _fallback_sampler_name(summaries: list[dict[str, str | int]]) -> str:
    non_empty = [item for item in summaries if int(item["candidate_count"]) > 0]
    if not non_empty:
        return str(summaries[0]["name"])
    best = max(non_empty, key=lambda item: int(item["candidate_count"]))
    return str(best["name"])
