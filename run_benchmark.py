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

from wmbo.config import default_run_config, load_run_config, merge_run_config
from wmbo.control import RunConfig
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
    parser.add_argument("--config", type=str, default=None, help="YAML experiment configuration file.")
    parser.add_argument("--benchmarks", type=str, default=None, help="Comma-separated benchmark names.")
    parser.add_argument("--methods", type=str, default=None, help="Comma-separated optimiser names.")
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated random seeds.")
    parser.add_argument("--budget", type=int, default=None, help="Evaluation budget per run.")
    parser.add_argument("--initial-samples", type=int, default=None, help="Initial design size for model-based methods.")
    parser.add_argument("--candidate-pool-size", type=int, default=None, help="Candidate pool size for acquisition search.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for JSON and CSV outputs.")
    return parser


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Convert command-line arguments into a run configuration.

    Input:
        args: Parsed command-line arguments.

    Output:
        ``RunConfig`` consumed by the benchmark runner.
    """

    config = load_run_config(args.config) if args.config else default_run_config()
    benchmarks = parse_csv(args.benchmarks) if args.benchmarks else None
    methods = parse_csv(args.methods) if args.methods else None
    seeds = parse_int_csv(args.seeds) if args.seeds else None
    return merge_run_config(
        config,
        benchmarks=benchmarks,
        methods=methods,
        seeds=seeds,
        budget=args.budget,
        initial_samples=args.initial_samples,
        candidate_pool_size=args.candidate_pool_size,
        output_dir=args.output_dir,
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
