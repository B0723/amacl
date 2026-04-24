from __future__ import annotations

import argparse

from amacl.agent import ColdStartAgent
from amacl.config import DATA_MUTATION_DIR, HIGH_FREQ_SEED_FILE, RuntimeConfig
from amacl.modeling import QwenLocalGenerator
from amacl.planner import MutationPlanner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "AMACL 冷启动出题 Agent：从高频 seed pool 采样 N 条，"
            "用全部 8 种算子各改写一次，共生成 N×8 条变异样本"
        )
    )
    parser.add_argument(
        "--seed-pool-file",
        default=HIGH_FREQ_SEED_FILE,
        help="预处理后的 JSONL 文件路径（由 preprocess_csv_to_jsonl.py 生成，默认：high_frequency_seed_pool.jsonl）",
    )
    parser.add_argument(
        "--seed-pool-name",
        default="high_frequency",
        help="seed pool 名称标识，用于输出文件命名（默认：high_frequency）",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="输出 JSONL 文件路径；默认在 data_mutation_dir 下自动命名",
    )
    parser.add_argument(
        "--sample-n",
        type=int,
        default=100,
        help="从 seed pool 中随机采样的条数（默认：100）",
    )
    parser.add_argument(
        "--model-path",
        default=RuntimeConfig().model_path,
        help="LLM 模型路径",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=RuntimeConfig().temperature,
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=RuntimeConfig().top_p,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=RuntimeConfig().max_new_tokens,
    )
    parser.add_argument(
        "--data-mutation-dir",
        default=DATA_MUTATION_DIR,
        help="变异样本输出目录（默认：data_mutation/）",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=RuntimeConfig().random_seed,
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    config = RuntimeConfig(
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        cold_start_sample_n=args.sample_n,
        high_freq_seed_file=args.seed_pool_file,
        data_mutation_dir=args.data_mutation_dir,
        random_seed=args.random_seed,
    )

    generator = QwenLocalGenerator(config)
    mutation_planner = MutationPlanner(generator)
    agent = ColdStartAgent(mutation_planner, config)

    agent.run(
        seed_pool_file=args.seed_pool_file,
        seed_pool_name=args.seed_pool_name,
        output_file=args.output_file,
    )


if __name__ == "__main__":
    main()
