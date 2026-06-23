"""Command-line entry point for future benchmark runs."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--benchmarks", type=str, default="", help="Comma-separated benchmark names.")
    parser.add_argument("--methods", type=str, default="", help="Comma-separated optimiser names.")
    parser.add_argument("--seeds", type=str, default="0", help="Comma-separated random seeds.")
    parser.add_argument("--budget", type=int, default=20, help="Evaluation budget per run.")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory for future outputs.")
    return parser


def build_run_config(args: argparse.Namespace) -> RunConfig:
    """Convert command-line arguments into a run configuration.

    Input:
        args: Parsed command-line arguments.

    Output:
        ``RunConfig`` consumed by the benchmark runner.
    """

    benchmarks = parse_csv(args.benchmarks) or []
    methods = parse_csv(args.methods) or []
    seeds = parse_int_csv(args.seeds) or [0]
    optimizer = OptimizerConfig(
        method=methods[0] if methods else "",
        budget=args.budget,
        initial_samples=0,
        candidate_pool_size=0,
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
    """Parse command-line arguments and start the benchmark runner.

    Input:
        Command-line arguments from ``sys.argv``.

    Output:
        None. Future implementation will write benchmark artifacts to disk.
    """

    parser = build_parser()
    args = parser.parse_args()
    config = build_run_config(args)
    run_benchmark_suite(config)


if __name__ == "__main__":
    main()
