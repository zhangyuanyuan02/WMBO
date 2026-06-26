# RP-WMBO

Initial repository for a research project on world-model reasoning for black-box optimisation.

This repository is still at an early implementation stage. The current commit adds YAML experiment configs so benchmark runs can be reproduced from checked-in settings.

## Structure

```text
configs/         Reproducible experiment settings
src/wmbo/        Python package source code
run_benchmark.py Command-line entry point
requirements.txt Project dependencies
```

## Current status

Implemented so far:

- Synthetic benchmark functions and search-space utilities
- Gaussian-process surrogate wrapper
- Acquisition scoring utilities for candidate selection
- Baseline optimisers: random search, Sobol search, BO-EI/PI/LCB, and a simple evolutionary strategy
- Rule-based WMBO landscape descriptor and reasoning agent
- WMBO candidate generation through `ask -> evaluate -> tell`
- Minimal benchmark runner with JSON/CSV output
- YAML configuration files for reproducible runs

## Examples

Run the default CLI settings:

```bash
python run_benchmark.py --benchmarks branin --methods random,wmbo --seeds 0 --budget 10
```

Run from a checked-in config:

```bash
python run_benchmark.py --config configs/debug.yaml
```

Command-line values can override config values:

```bash
python run_benchmark.py --config configs/debug.yaml --budget 5 --output-dir results/quick_check
```

Outputs are written under the configured output directory.

## TODO

- Add tests and result analysis
- Add plotting utilities for saved outputs
- Add optional LLM-backed reasoning after the rule-based agent is stable
