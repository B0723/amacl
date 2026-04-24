from __future__ import annotations

from dataclasses import dataclass, field


SERVER_WORKSPACE_ROOT = "/home/sankuai/buyixin02/amacl-main"
DEFAULT_MODEL_PATH = "/home/sankuai/buyixin02/hidden_prob/Model/Qwen2.5-7B-Instruct"

# Seed pool 文件路径（预处理后的 JSONL，由 scripts/preprocess_csv_to_jsonl.py 生成）
HIGH_FREQ_SEED_FILE = f"{SERVER_WORKSPACE_ROOT}/data/high_frequency_seed_pool.jsonl"

# Mutation 输出目录
DATA_MUTATION_DIR = f"{SERVER_WORKSPACE_ROOT}/data_mutation"


@dataclass(slots=True)
class RuntimeConfig:
    workspace_root: str = SERVER_WORKSPACE_ROOT
    model_path: str = DEFAULT_MODEL_PATH

    # LLM 生成参数
    temperature: float = 0.7
    top_p: float = 0.9
    max_new_tokens: int = 512

    # 冷启动采样参数
    cold_start_sample_n: int = 100          # 从高频 pool 中随机采样的条数
    high_freq_seed_file: str = HIGH_FREQ_SEED_FILE

    # 输出路径
    data_mutation_dir: str = DATA_MUTATION_DIR

    # 随机种子
    random_seed: int = 20260401
