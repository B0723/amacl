from __future__ import annotations

from dataclasses import asdict, dataclass

from .registry import OPERATOR_REGISTRY, register_operator


@dataclass(slots=True)
class MutationOperator:
    name: str
    description: str
    constraints: str
    default_effect: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@register_operator
class SynonymSubstitution:
    name = "synonym_substitution"
    description = "使用近义表达改写 query 或 doc，制造 hard positive。"
    constraints = "保持业务语义一致，不要改坏主实体，不要生成生硬口号。"
    default_effect = "keep_label"


@register_operator
class EntityReplacement:
    name = "entity_replacement"
    description = "替换核心实体或关键词，制造局部语义错配。"
    constraints = "替换后的实体必须真实、常见，且与原文本仍存在字面相似性。"
    default_effect = "decrease_relevance"


@register_operator
class BrandConfusion:
    name = "brand_confusion"
    description = "做同类目品牌替换，制造品牌混淆负例。"
    constraints = "品牌要真实存在，优先同类目竞品，避免跨行业乱替换。"
    default_effect = "decrease_relevance"


@register_operator
class CategoryDrift:
    name = "category_drift"
    description = "从相关类目漂移到邻近但不一致的类目。"
    constraints = "保留足够高的字面相似度，但目标类目必须发生变化。"
    default_effect = "decrease_relevance"


@register_operator
class AttributeInjection:
    name = "attribute_injection"
    description = "增加规格、数量、套餐或场景属性，制造细粒度匹配难例。"
    constraints = "注入属性必须自然，优先使用容量、距离、套餐、人数、门店属性。"
    default_effect = "decrease_relevance"


@register_operator
class IntentShift:
    name = "intent_shift"
    description = "把购买意图、了解意图、到店意图等进行转换。"
    constraints = "意图变化要清晰，但文本仍然要像真实搜索或真实商家文案。"
    default_effect = "decrease_relevance"


@register_operator
class NegationInsertion:
    name = "negation_insertion"
    description = "插入否定或排除信息，制造逻辑反转样本。"
    constraints = "否定表达必须自然，例如不含、无需、非、不是，不要制造病句。"
    default_effect = "decrease_relevance"


@register_operator
class ColloquialRewrite:
    name = "colloquial_rewrite"
    description = "做口语化、别名化、俗称化改写，制造鲁棒性 hard positive。"
    constraints = "改写必须符合中文口语习惯，可以用别称，但不要过度网络黑话。"
    default_effect = "keep_label"


def build_operator_catalogue() -> list[MutationOperator]:
    operators: list[MutationOperator] = []
    for operator_cls in OPERATOR_REGISTRY.values():
        operators.append(
            MutationOperator(
                name=operator_cls.name,
                description=operator_cls.description,
                constraints=operator_cls.constraints,
                default_effect=operator_cls.default_effect,
            )
        )
    return operators
