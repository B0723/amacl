from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from amacl.registry import OPERATOR_REGISTRY, register_operator


# ---------------------------------------------------------------------------
# Few-shot 示例格式：每个示例包含输入和输出，展示"在给定 seed 上如何应用该算子"
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class FewShotExample:
    """单条 few-shot 示例，直接对应最终输出 JSON 的结构。"""
    description: str        # 一句话说明本条示例的变异意图
    original_query: str
    original_doc: str
    mutated_query: str
    mutated_doc: str
    target_field: str       # query / doc / both
    expected_effect: str    # keep_label / decrease_relevance / increase_relevance
    rationale: str          # 变异理由（对应输出 JSON 的 rationale 字段）

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# MutationOperator：带 few-shot 示例的算子定义
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MutationOperator:
    name: str
    description: str
    constraints: str
    default_effect: str
    few_shot_examples: list[FewShotExample] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# 算子定义（8 个）
# ---------------------------------------------------------------------------

@register_operator
class SynonymSubstitution:
    """近义改写：保持语义不变，制造词汇多样性 hard positive。"""
    name = "synonym_substitution"
    description = "对 query 或 doc 使用近义词、别称或书面/口语对等表达进行改写，制造词汇多样性 hard positive，相关性标签保持不变。"
    constraints = "保持核心业务语义完全一致，不要改变主实体类目，不要引入无关属性，不要生成生硬口号。"
    default_effect = "keep_label"
    few_shot_examples = [
        FewShotExample(
            description="将 query '美甲' 改写为近义词'指甲彩绘'，doc 同步改写，标签保持不变",
            original_query="美甲",
            original_doc="艾美美甲（万达店）| 美甲单色/渐变套餐",
            mutated_query="指甲彩绘",
            mutated_doc="艾美美甲（万达店）| 指甲彩绘单色/渐变套餐",
            target_field="both",
            expected_effect="keep_label",
            rationale="'美甲'与'指甲彩绘'是同一业务的不同表达，改写后语义完全等价，用于测试 BERT 是否依赖词汇精确匹配。",
        ),
        FewShotExample(
            description="将 doc 中'套餐'改写为'组合'，测试同义词替换鲁棒性",
            original_query="火锅双人餐",
            original_doc="小龙坎火锅（旗舰店）| 双人火锅套餐（锅底+牛肉+虾滑）",
            mutated_query="火锅双人餐",
            mutated_doc="小龙坎火锅（旗舰店）| 双人火锅组合（锅底+牛肉+虾滑）",
            target_field="doc",
            expected_effect="keep_label",
            rationale="'套餐'与'组合'在餐饮场景是同义表达，改写后相关性不变，测试模型对近义词的敏感度。",
        ),
    ]


@register_operator
class EntityReplacement:
    """核心实体替换：将主实体换成同类但不同的实体，制造局部语义错配。"""
    name = "entity_replacement"
    description = "将 query 或 doc 中的核心实体（商品名、服务名、菜品名）替换为同类但不同的实体，使 query-doc 相关性降低但仍保持字面相似。"
    constraints = "替换后的实体必须真实、常见；与原实体属于同一大类目；替换后文本要语义通顺自然。"
    default_effect = "decrease_relevance"
    few_shot_examples = [
        FewShotExample(
            description="将 doc 中的'烤鱼'替换为'烤鸭'，query 搜的是烤鱼但 doc 变成了烤鸭商家",
            original_query="烤鱼",
            original_doc="渝是乎烤鱼（中关村店）| 招牌麻辣烤鱼2-3人套餐",
            mutated_query="烤鱼",
            mutated_doc="渝是乎烤鸭（中关村店）| 招牌麻辣烤鸭2-3人套餐",
            target_field="doc",
            expected_effect="decrease_relevance",
            rationale="将'烤鱼'替换为同属烤制菜品的'烤鸭'，字面高度相似但核心菜品不同，用于测试 BERT 是否会被字面相似性误导。",
        ),
        FewShotExample(
            description="将 query '拿铁' 换成 '卡布奇诺'，doc 保持原来的拿铁商家",
            original_query="拿铁咖啡",
            original_doc="瑞幸咖啡（望京SOHO店）| 生椰拿铁·冷萃拿铁买一赠一",
            mutated_query="卡布奇诺咖啡",
            mutated_doc="瑞幸咖啡（望京SOHO店）| 生椰拿铁·冷萃拿铁买一赠一",
            target_field="query",
            expected_effect="decrease_relevance",
            rationale="将 query 从'拿铁'换为'卡布奇诺'，两者同属咖啡饮品但口味/制作不同，doc 仍是拿铁商家，测试模型是否能区分同类但不同的咖啡品种。",
        ),
    ]


@register_operator
class BrandConfusion:
    """品牌混淆：将商家品牌替换为同类目竞品品牌，制造品牌混淆负例。"""
    name = "brand_confusion"
    description = "将 doc 中的品牌名替换为同类目真实竞品品牌，保持其余信息不变，测试模型是否会因类目相同而误判相关。"
    constraints = "替换品牌必须真实存在且在同一细分类目；替换后的商家名和团单信息要保持格式一致；不要跨大类目替换。"
    default_effect = "decrease_relevance"
    few_shot_examples = [
        FewShotExample(
            description="query 搜'星巴克'，doc 品牌替换为同类咖啡品牌'瑞幸'",
            original_query="星巴克咖啡",
            original_doc="星巴克（三里屯店）| 星冰乐·馥芮白买一赠一",
            mutated_query="星巴克咖啡",
            mutated_doc="瑞幸咖啡（三里屯店）| 生椰拿铁·冷萃美式买一赠一",
            target_field="doc",
            expected_effect="decrease_relevance",
            rationale="query 明确指定'星巴克'品牌，doc 替换为竞品'瑞幸'后品牌不匹配，但同属咖啡类目，字面相似度高，用于测试 BERT 是否能识别品牌层面的不相关。",
        ),
        FewShotExample(
            description="query 搜'海底捞'，doc 替换为同类火锅品牌'呷哺呷哺'",
            original_query="海底捞火锅",
            original_doc="海底捞火锅（望京店）| 双人锅底+牛肉卷+虾滑套餐",
            mutated_query="海底捞火锅",
            mutated_doc="呷哺呷哺火锅（望京店）| 双人锅底+牛肉卷+虾滑套餐",
            target_field="doc",
            expected_effect="decrease_relevance",
            rationale="保留'望京'位置和套餐内容，仅替换品牌为同类火锅竞品，测试模型对品牌名的敏感度。",
        ),
    ]


@register_operator
class CategoryDrift:
    """类目漂移：将 doc 从当前类目漂移到邻近但不一致的类目，保留字面相似度。"""
    name = "category_drift"
    description = "将 doc 的商家/团单类目从 query 对应类目漂移到邻近类目，保留高字面相似度但改变核心类目，使得 query 与 doc 不再真正相关。"
    constraints = "漂移目标类目必须与原类目相邻（如'餐饮→外卖'、'美甲→美睫'、'按摩→足疗'），字面上要有重叠词，但服务性质已发生改变。"
    default_effect = "decrease_relevance"
    few_shot_examples = [
        FewShotExample(
            description="query 搜'手机'，doc 从'手机'类漂移到'手机壳'类",
            original_query="苹果手机",
            original_doc="苹果授权体验店（西单店）| iPhone 15 Pro 128G 现货",
            mutated_query="苹果手机",
            mutated_doc="苹果配件专营店（西单店）| iPhone 15 Pro 硅胶保护壳·磁吸手机壳",
            target_field="doc",
            expected_effect="decrease_relevance",
            rationale="从'手机整机'漂移到'手机配件'类目，保留品牌'苹果'和型号'iPhone 15 Pro'的字面相似度，但 query 意图是买手机而非手机壳，类目已发生漂移。",
        ),
        FewShotExample(
            description="query 搜'按摩'，doc 从'按摩'类漂移到相邻的'足疗'类",
            original_query="全身按摩放松",
            original_doc="禅意养生馆（国贸店）| 90分钟全身经络按摩套餐",
            mutated_query="全身按摩放松",
            mutated_doc="禅意养生馆（国贸店）| 60分钟足底反射区按摩（含足浴）",
            target_field="doc",
            expected_effect="decrease_relevance",
            rationale="query 明确要求'全身按摩'，doc 漂移到'足疗'类目，保留商家名和'按摩'关键词，但服务范围从全身缩小到足部，测试模型对服务范围的细粒度识别。",
        ),
    ]


@register_operator
class AttributeInjection:
    """属性注入：在 query 中注入细粒度限定属性，使原本相关的 doc 因不满足属性而变为不相关。"""
    name = "attribute_injection"
    description = "向 query 中注入规格、数量、套餐人数、地理位置、场景等细粒度限定属性，使 doc 因不满足该属性而与 query 不相关。"
    constraints = "注入的属性必须自然，优先使用容量/人数/距离/套餐/门店/时段等常见限定词；注入后 query 要像真实用户搜索词。"
    default_effect = "decrease_relevance"
    few_shot_examples = [
        FewShotExample(
            description="query '火锅' 注入'4人以上包间'属性，使普通双人餐 doc 不再匹配",
            original_query="火锅",
            original_doc="小龙坎火锅（三里屯店）| 双人火锅套餐（锅底+4荤4素）",
            mutated_query="火锅 4人以上包间",
            mutated_doc="小龙坎火锅（三里屯店）| 双人火锅套餐（锅底+4荤4素）",
            target_field="query",
            expected_effect="decrease_relevance",
            rationale="原 query 与 doc 高度相关，注入'4人以上包间'属性后，doc 是双人套餐且无包间信息，query 的细粒度需求无法被满足，测试 BERT 对属性匹配的识别能力。",
        ),
        FewShotExample(
            description="query '咖啡' 注入'外卖30分钟送达'属性，使到店消费的 doc 不再匹配",
            original_query="咖啡",
            original_doc="MANNER咖啡（建国路店）| 美式咖啡·拿铁单杯",
            mutated_query="咖啡 外卖30分钟送达",
            mutated_doc="MANNER咖啡（建国路店）| 美式咖啡·拿铁单杯",
            target_field="query",
            expected_effect="decrease_relevance",
            rationale="原 doc 是门店信息，注入外卖时效属性后，该门店是否支持外卖及时效未知，query 的配送需求与 doc 可能不匹配，测试模型对场景属性的理解。",
        ),
    ]


@register_operator
class IntentShift:
    """意图转换：将 query 的意图类型进行转换，使其与 doc 的供给侧意图不再对齐。"""
    name = "intent_shift"
    description = "将 query 的意图从购买/到店意图转换为了解/比较意图（或反向），使 query 的用户需求与 doc 提供的商业供给产生错位。"
    constraints = "意图变化要语义清晰，改写后 query 要像真实搜索词；不要改变 query 的核心实体，只改变意图信号词。"
    default_effect = "decrease_relevance"
    few_shot_examples = [
        FewShotExample(
            description="将购买意图'买手机'转换为了解意图'手机推荐/对比'",
            original_query="买苹果手机",
            original_doc="苹果授权专营店（中关村店）| iPhone 15 Pro 256G 现货直降500元",
            mutated_query="苹果手机哪款好 推荐",
            mutated_doc="苹果授权专营店（中关村店）| iPhone 15 Pro 256G 现货直降500元",
            target_field="query",
            expected_effect="decrease_relevance",
            rationale="原 query 是明确的购买意图，doc 是促销商家非常相关；改写后 query 变为信息获取意图（对比推荐），与商家的'立即下单'供给不再对齐，测试模型对意图匹配的识别。",
        ),
        FewShotExample(
            description="将到店用餐意图转换为外卖配送意图",
            original_query="北京烤鸭堂食",
            original_doc="全聚德烤鸭店（前门旗舰店）| 经典北京烤鸭3-4人套餐（含荷叶饼+酱料）",
            mutated_query="北京烤鸭外卖配送",
            mutated_doc="全聚德烤鸭店（前门旗舰店）| 经典北京烤鸭3-4人套餐（含荷叶饼+酱料）",
            target_field="query",
            expected_effect="decrease_relevance",
            rationale="全聚德是经典堂食体验场所，doc 未展示外卖信息；将 query 意图从堂食改为外卖后，供需场景错位，测试模型对到店与外卖意图的区分能力。",
        ),
    ]


@register_operator
class NegationInsertion:
    """否定插入：在 query 中插入否定或排除信息，逻辑上反转相关性。"""
    name = "negation_insertion"
    description = "在 query 中插入否定词或排除条件（不含、无需、非、不是、不要），使原本相关的 doc 因不满足排除条件而变为不相关。"
    constraints = "否定表达必须自然流畅（不含、无需、不要、非XXX均可）；不要制造语法错误的病句；否定范围要指向 doc 的核心属性。"
    default_effect = "decrease_relevance"
    few_shot_examples = [
        FewShotExample(
            description="query '火锅' 插入否定词'不要辣'，使麻辣火锅 doc 不再匹配",
            original_query="火锅",
            original_doc="蜀大侠火锅（望京店）| 招牌麻辣锅底双人套餐",
            mutated_query="火锅 不要辣 清淡口味",
            mutated_doc="蜀大侠火锅（望京店）| 招牌麻辣锅底双人套餐",
            target_field="query",
            expected_effect="decrease_relevance",
            rationale="doc 的核心是麻辣锅底，插入'不要辣'否定条件后，query 的需求与 doc 的供给产生直接冲突，测试 BERT 是否能理解否定语义。",
        ),
        FewShotExample(
            description="query '停车场' 插入'不含地下停车'排除信息",
            original_query="附近停车场",
            original_doc="万象城地下停车场 | 地下4层·可停800辆·24小时开放",
            mutated_query="附近停车场 不含地下 露天停车",
            mutated_doc="万象城地下停车场 | 地下4层·可停800辆·24小时开放",
            target_field="query",
            expected_effect="decrease_relevance",
            rationale="doc 明确是地下停车场，插入'不含地下'排除条件后，与 doc 直接矛盾，测试模型对精确属性排除的识别能力。",
        ),
    ]


@register_operator
class ColloquialRewrite:
    """口语化改写：将书面表达改写为口语/别称/俗名，制造鲁棒性 hard positive。"""
    name = "colloquial_rewrite"
    description = "将 query 或 doc 中的书面词汇、行业术语改写为口语表达、民间别称或常见俗名，标签不变，测试模型对表达多样性的鲁棒性。"
    constraints = "改写后表达必须是真实用户常用的口语/别称（如'金拱门'代指麦当劳），不要生造词汇；不要过度使用网络黑话；改写后语义与原文必须等价。"
    default_effect = "keep_label"
    few_shot_examples = [
        FewShotExample(
            description="将'麦当劳'改写为口语别称'金拱门'",
            original_query="麦当劳汉堡",
            original_doc="麦当劳（国贸店）| 巨无霸套餐·麦辣鸡腿堡套餐",
            mutated_query="金拱门汉堡",
            mutated_doc="金拱门（国贸店）| 巨无霸套餐·麦辣鸡腿堡套餐",
            target_field="both",
            expected_effect="keep_label",
            rationale="'金拱门'是麦当劳在中国的官方中文名，也是流行口语别称，改写后语义完全等价，测试模型是否只依赖品牌名字面匹配。",
        ),
        FewShotExample(
            description="将医疗术语'肠镜检查'改写为口语说法'查肠子'",
            original_query="做肠镜检查哪里好",
            original_doc="北京和睦家医院 | 无痛肠镜检查套餐（含术前评估）",
            mutated_query="查肠子去哪里做比较好",
            mutated_doc="北京和睦家医院 | 无痛肠镜检查套餐（含术前评估）",
            target_field="query",
            expected_effect="keep_label",
            rationale="'查肠子'是'肠镜检查'的口语说法，语义等价，测试模型是否能识别医疗领域的口语-书面语对应关系。",
        ),
    ]


# ---------------------------------------------------------------------------
# 构建算子列表（供 Agent 使用）
# ---------------------------------------------------------------------------

def build_operator_catalogue() -> list[MutationOperator]:
    operators: list[MutationOperator] = []
    for operator_cls in OPERATOR_REGISTRY.values():
        operators.append(
            MutationOperator(
                name=operator_cls.name,
                description=operator_cls.description,
                constraints=operator_cls.constraints,
                default_effect=operator_cls.default_effect,
                few_shot_examples=list(operator_cls.few_shot_examples),
            )
        )
    return operators
