"""Command-line entry point for benchmark runs."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from wmbo.control import OptimizerConfig, RunConfig
from wmbo.runner import run_benchmark_suite
from wmbo.utils import parse_csv, parse_int_csv


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser.

    Input:
        None.

    Output:
        Configured ``ArgumentParser``.
    """

    parser = argparse.ArgumentParser(description="Run WMBO benchmark experiments.")
    parser.add_argument("--benchmarks", type=str, default="branin", help="Comma-separated benchmark names.")
    parser.add_argument("--methods", type=str, default="random", help="Comma-separated optimiser names.")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated random seeds.")
    parser.add_argument("--budget", type=int, default=20, help="Evaluation budget per run.")
    parser.add_argument("--initial-samples", type=int, default=5, help="Initial design size for model-based methods.")
    parser.add_argument("--candidate-pool-size", type=int, default=256, help="Candidate pool size for acquisition search.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for JSON and CSV outputs.")
    return parser


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Convert command-line arguments into a run configuration.

    Input:
        args: Parsed command-line arguments.

    Output:
        ``RunConfig`` consumed by the benchmark runner.
    """

    benchmarks = parse_csv(args.benchmarks) or ["branin"]
    methods = parse_csv(args.methods) or ["random"]
    seeds = parse_int_csv(args.seeds) or [0]
    budget = int(args.budget)
    if budget <= 0:
        raise ValueError("--budget must be positive.")
    optimizer = OptimizerConfig(
        method=methods[0],
        budget=budget,
        initial_samples=max(1, min(int(args.initial_samples), budget)),
        candidate_pool_size=max(1, int(args.candidate_pool_size)),
        seed=seeds[0],
    )
    return RunConfig(
        benchmarks=benchmarks,
        methods=methods,
        seeds=seeds,
        output_dir=args.output_dir,
        optimizer=optimizer,
    )


def main() -> None:
    """Parse command-line arguments and run benchmark experiments.

    Input:
        Command-line arguments from ``sys.argv``.

    Output:
        None. Result artifacts are written under ``--output-dir``.
    """

    parser = build_parser()
    args = parser.parse_args()
    config = build_run_config(args)
    results = run_benchmark_suite(config)

    print("completed benchmark runs:")
    for result in results:
        summary = asdict(result.summary)
        print(
            "- {benchmark_name} | {method} | seed={seed} | best={final_best:.6g} | n={num_evaluations}".format(
                benchmark_name=summary["benchmark_name"],
                method=summary["method"],
                seed=summary["seed"],
                final_best=summary["final_best"] if summary["final_best"] is not None else float("nan"),
                num_evaluations=summary["num_evaluations"],
            )
        )
    print(f"saved results to: {config.output_dir}")


if __name__ == "__main__":
    main()
