from __future__ import annotations

import argparse

from amacl.agent import QuestionGenerationAgent
from amacl.config import DEFAULT_OUTPUT_FILE, DEFAULT_SEED_FILE, RuntimeConfig
from amacl.modeling import QwenLocalGenerator
from amacl.planner import MutationPlanner, StrategyPlanner
from amacl.utils import dump_jsonl, load_seed_cases


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AMACL MVP question-generation agent")
    parser.add_argument("--seed-file", default=DEFAULT_SEED_FILE)
    parser.add_argument("--output-file", default=DEFAULT_OUTPUT_FILE)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--model-path", default=RuntimeConfig().model_path)
    parser.add_argument("--temperature", type=float, default=RuntimeConfig().temperature)
    parser.add_argument("--top-p", type=float, default=RuntimeConfig().top_p)
    parser.add_argument("--max-new-tokens", type=int, default=RuntimeConfig().max_new_tokens)
    parser.add_argument("--high-freq-threshold", type=int, default=RuntimeConfig().high_freq_threshold)
    parser.add_argument("--tail-freq-threshold", type=int, default=RuntimeConfig().tail_freq_threshold)
    parser.add_argument("--boundary-low", type=float, default=RuntimeConfig().boundary_low)
    parser.add_argument("--boundary-high", type=float, default=RuntimeConfig().boundary_high)
    parser.add_argument("--random-seed", type=int, default=RuntimeConfig().random_seed)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = RuntimeConfig(
        model_path=args.model_path,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        high_freq_threshold=args.high_freq_threshold,
        tail_freq_threshold=args.tail_freq_threshold,
        boundary_low=args.boundary_low,
        boundary_high=args.boundary_high,
        random_seed=args.random_seed,
    )

    seed_cases = load_seed_cases(args.seed_file)
    generator = QwenLocalGenerator(config)
    strategy_planner = StrategyPlanner(generator, config)
    mutation_planner = MutationPlanner(generator)
    agent = QuestionGenerationAgent(strategy_planner, mutation_planner, config)
    generated = agent.generate_many(seed_cases, args.num_samples)

    dump_jsonl(args.output_file, [item.to_dict() for item in generated])
    print(f"Generated {len(generated)} cases -> {args.output_file}")


if __name__ == "__main__":
    main()
