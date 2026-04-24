from __future__ import annotations

import json

from amacl.modeling import BaseTextGenerator
from amacl.operators import MutationOperator
from amacl.schemas import MutationDraft, SeedCase
from amacl.utils import extract_json_object


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

MUTATION_SYSTEM_PROMPT = """你是美团搜索相关性训练数据的出题 Agent。
你的任务是：给定一条搜索 query 和对应的 POI（商家）信息，选择一种合适的变异算子（mutation operator），
对该样本进行语义变异，生成一条"更难但仍然真实自然"的训练样本。

变异目标：让 BERT 等小模型更难正确判断相关性，同时保证变异后的文本符合真实业务场景。

要求：
1. 选择最能暴露当前样本"相关性判断弱点"的算子
2. 变异后的 query 或 doc 必须像真实的搜索词或真实的商家/团单文案
3. 输出严格的 JSON，不要输出任何解释性文字"""


# ---------------------------------------------------------------------------
# MutationPlanner
# ---------------------------------------------------------------------------

class MutationPlanner:
    """调用 LLM 为单条 SeedCase 选择算子并生成变异样本。"""

    def __init__(self, generator: BaseTextGenerator):
        self.generator = generator

    def generate(self, seed: SeedCase, operators: list[MutationOperator]) -> MutationDraft:
        """让 LLM 从全部算子中自由选择最合适的一个并生成变异样本。"""
        prompt = self._build_prompt(seed, operators)
        raw = self.generator.generate(MUTATION_SYSTEM_PROMPT, prompt)

        try:
            payload = extract_json_object(raw)
        except (ValueError, Exception):
            payload = {}

        # 校验 operator_name
        valid_names = {op.name for op in operators}
        operator_name = str(payload.get("operator_name", "")).strip()
        if operator_name not in valid_names:
            operator_name = operators[0].name

        return MutationDraft(
            operator_name=operator_name,
            target_field=str(payload.get("target_field", "doc")).strip() or "doc",
            expected_effect=str(payload.get("expected_effect", "decrease_relevance")).strip(),
            rationale=str(payload.get("rationale", "")).strip(),
            mutated_query=str(payload.get("mutated_query", seed.query)).strip() or seed.query,
            mutated_doc=str(payload.get("mutated_doc", seed.doc)).strip() or seed.doc,
            raw_response=raw,
        )

    def generate_with_operator(
        self,
        seed: SeedCase,
        operator: MutationOperator,
        all_operators: list[MutationOperator],
    ) -> MutationDraft:
        """强制使用指定算子对 seed 进行变异，LLM 只需生成变异内容而无需选择算子。"""
        prompt = self._build_prompt_for_operator(seed, operator)
        raw = self.generator.generate(MUTATION_SYSTEM_PROMPT, prompt)

        try:
            payload = extract_json_object(raw)
        except (ValueError, Exception):
            payload = {}

        return MutationDraft(
            operator_name=operator.name,
            target_field=str(payload.get("target_field", "doc")).strip() or "doc",
            expected_effect=str(payload.get("expected_effect", operator.default_effect)).strip(),
            rationale=str(payload.get("rationale", "")).strip(),
            mutated_query=str(payload.get("mutated_query", seed.query)).strip() or seed.query,
            mutated_doc=str(payload.get("mutated_doc", seed.doc)).strip() or seed.doc,
            raw_response=raw,
        )

    # ------------------------------------------------------------------
    def _build_prompt(self, seed: SeedCase, operators: list[MutationOperator]) -> str:
        # 1. 构建原始样本描述
        seed_info: dict = {
            "query": seed.query,
            "poi_name": seed.poi_name or "",
            "doc": seed.doc,
            "relevance_label": seed.label,
        }
        if seed.first_cate:
            seed_info["first_category"] = seed.first_cate
        if seed.second_cate:
            seed_info["second_category"] = seed.second_cate
        if seed.top_deals:
            seed_info["top_deals"] = seed.top_deals[:3]
        if seed.top_dishes:
            seed_info["top_dishes"] = seed.top_dishes

        # 2. 构建算子描述块（每个算子附带 few-shot 示例）
        operator_blocks = _render_operator_blocks(operators)

        # 3. 输出格式说明
        output_format = (
            "{\n"
            '  "operator_name": "算子名称（必须是上方列出的算子之一）",\n'
            '  "target_field": "query 或 doc 或 both",\n'
            '  "expected_effect": "decrease_relevance 或 keep_label 或 increase_relevance",\n'
            '  "rationale": "选择该算子的理由及变异逻辑",\n'
            '  "mutated_query": "变异后的 query（若 target_field 为 doc，则与原 query 相同）",\n'
            '  "mutated_doc": "变异后的 doc"\n'
            "}"
        )

        return (
            "## 原始样本\n\n"
            f"{json.dumps(seed_info, ensure_ascii=False, indent=2)}\n\n"
            "---\n\n"
            "## 可用变异算子（含示例）\n\n"
            f"{operator_blocks}\n"
            "---\n\n"
            "## 任务\n\n"
            "请为上方的原始样本选择一个最合适的算子，生成变异样本。\n"
            "参考对应算子的示例格式，严格输出以下 JSON（不要输出任何其他内容）：\n\n"
            f"{output_format}"
        )

    def _build_prompt_for_operator(self, seed: SeedCase, operator: MutationOperator) -> str:
        """为指定算子构建 prompt，LLM 只需生成变异内容，无需选择算子。"""
        seed_info: dict = {
            "query": seed.query,
            "doc": seed.doc,
            "relevance_label": seed.label,
        }
        if seed.poi_name:
            seed_info["poi_name"] = seed.poi_name
        if seed.first_cate:
            seed_info["first_category"] = seed.first_cate
        if seed.second_cate:
            seed_info["second_category"] = seed.second_cate
        if seed.top_deals:
            seed_info["top_deals"] = seed.top_deals[:3]
        if seed.top_dishes:
            seed_info["top_dishes"] = seed.top_dishes

        # 只展示当前指定算子的描述和示例
        op_block = _render_operator_blocks([operator])

        output_format = (
            "{\n"
            f'  "operator_name": "{operator.name}",\n'
            '  "target_field": "query 或 doc 或 both",\n'
            f'  "expected_effect": "{operator.default_effect}",\n'
            '  "rationale": "变异逻辑说明",\n'
            '  "mutated_query": "变异后的 query（若只改 doc 则与原 query 相同）",\n'
            '  "mutated_doc": "变异后的 doc"\n'
            "}"
        )

        return (
            "## 原始样本\n\n"
            f"{json.dumps(seed_info, ensure_ascii=False, indent=2)}\n\n"
            "---\n\n"
            "## 本次使用的变异算子\n\n"
            f"{op_block}\n"
            "---\n\n"
            "## 任务\n\n"
            f"请使用【{operator.name}】算子对上方原始样本进行变异，生成变异样本。\n"
            "参考上方示例格式，严格输出以下 JSON（不要输出任何其他内容）：\n\n"
            f"{output_format}"
        )


# ---------------------------------------------------------------------------
# 私有：渲染算子描述块
# ---------------------------------------------------------------------------

def _render_operator_blocks(operators: list[MutationOperator]) -> str:
    """将每个算子渲染为：名称 + 说明 + 约束 + few-shot 示例。"""
    blocks: list[str] = []

    for op in operators:
        lines: list[str] = []
        lines.append(f"### 算子：{op.name}")
        lines.append(f"- **说明**：{op.description}")
        lines.append(f"- **约束**：{op.constraints}")
        lines.append(f"- **默认效果**：{op.default_effect}")

        if op.few_shot_examples:
            lines.append("- **示例**：")
            for idx, ex in enumerate(op.few_shot_examples, 1):
                # 渲染为"输入 → 输出"的完整 JSON 对，方便模型类比
                example_output = json.dumps(
                    {
                        "operator_name": op.name,
                        "target_field": ex.target_field,
                        "expected_effect": ex.expected_effect,
                        "rationale": ex.rationale,
                        "mutated_query": ex.mutated_query,
                        "mutated_doc": ex.mutated_doc,
                    },
                    ensure_ascii=False,
                    indent=4,
                )
                example_input = json.dumps(
                    {
                        "original_query": ex.original_query,
                        "original_doc": ex.original_doc,
                    },
                    ensure_ascii=False,
                    indent=4,
                )
                lines.append(
                    f"  示例{idx}（{ex.description}）：\n"
                    f"  输入：\n"
                    f"  ```\n{_indent(example_input, 2)}\n  ```\n"
                    f"  输出：\n"
                    f"  ```\n{_indent(example_output, 2)}\n  ```"
                )

        blocks.append("\n".join(lines))

    return "\n\n".join(blocks) + "\n"


def _indent(text: str, spaces: int) -> str:
    """给多行文本每行前加指定数量的空格缩进。"""
    pad = " " * spaces
    return "\n".join(pad + line for line in text.splitlines())
