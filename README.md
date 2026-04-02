# AMACL MVP

AMACL 的最小可运行版本第一阶段代码，只覆盖出题 agent：

- 宏观层：4 种 seed 采样策略
- 微观层：8 种 mutation operator
- 规划器：基于 `Qwen2.5-7B-Instruct` 选择采样策略并生成变异样本

当前版本不包含 reward、student 训练、GRPO 和 replay buffer。

## 目录

```text
src/amacl/
  config.py
  schemas.py
  registry.py
  seed_samplers.py
  operators.py
  modeling.py
  planner.py
  agent.py
  cli.py
tests/
```

## 默认服务器路径

- 工作目录：`/home/sankuai/buyixin02/AMACL`
- 模型路径：`/home/sankuai/buyixin02/hidden_prob/Model/Qwen2.5-7B-Instruct`

## 安装

```bash
cd /home/sankuai/buyixin02/AMACL
pip install -e .
```

如果服务器环境没有安装 `torch`，请先按服务器环境单独安装。当前仓库只声明了 `transformers` 依赖。

## seed pool 格式

输入使用 JSONL，每行一个 seed。MVP 里 4 种采样策略依赖下面这些字段中的一部分：

```json
{
  "seed_id": "case_001",
  "query": "苹果手机",
  "doc": "Apple iPhone 15 Pro",
  "label": 1,
  "query_frequency": 5600,
  "exposure": 2300,
  "click": 120,
  "ctr": 0.052,
  "student_confidence": 0.53,
  "tags": ["online_hardcase"],
  "meta": {
    "biz_scene": "search"
  }
}
```

字段缺失时，对应采样器会自动跳过这条规则。

## 运行

```bash
cd /home/sankuai/buyixin02/AMACL
PYTHONPATH=src python -m amacl.cli \
  --seed-file /home/sankuai/buyixin02/AMACL/data/seed_pool.jsonl \
  --output-file /home/sankuai/buyixin02/AMACL/outputs/generated_cases.jsonl \
  --num-samples 20
```

## 已实现的 seed sampler

- `high_freq_query`
- `tail_query`
- `online_hardcase`
- `boundary_confidence`

## 已实现的 operators

- `synonym_substitution`
- `entity_replacement`
- `brand_confusion`
- `category_drift`
- `attribute_injection`
- `intent_shift`
- `negation_insertion`
- `colloquial_rewrite`

## 扩展方式

新增 seed sampler：

1. 继承 `SeedSampler`
2. 实现 `is_match`
3. 用 `@register_sampler` 注册

新增 operator：

1. 继承 `MutationOperator`
2. 填写元信息和约束
3. 用 `@register_operator` 注册

Agent 主流程不需要修改。
