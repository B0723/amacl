from __future__ import annotations

from dataclasses import dataclass


SERVER_WORKSPACE_ROOT = "/home/sankuai/buyixin02/amacl-main"
DEFAULT_MODEL_PATH = "/home/sankuai/buyixin02/hidden_prob/Model/Qwen2.5-7B-Instruct"
DEFAULT_SEED_FILE = f"{SERVER_WORKSPACE_ROOT}/data/seed_pool.jsonl"
DEFAULT_OUTPUT_FILE = f"{SERVER_WORKSPACE_ROOT}/outputs/generated_cases.jsonl"


@dataclass(slots=True)
class RuntimeConfig:
    workspace_root: str = SERVER_WORKSPACE_ROOT
    model_path: str = DEFAULT_MODEL_PATH
    temperature: float = 0.2
    top_p: float = 0.9
    max_new_tokens: int = 384
    high_freq_threshold: int = 1000
    tail_freq_threshold: int = 10
    boundary_low: float = 0.4
    boundary_high: float = 0.6
    online_hardcase_min_exposure: int = 100
    online_hardcase_max_ctr: float = 0.05
    random_seed: int = 20260401
