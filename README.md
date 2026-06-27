# RP-WMBO

Initial repository for a research project on world-model reasoning for black-box optimisation.

This repository is still at an early implementation stage. The current commit adds plotting utilities for saved benchmark outputs.

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
- Plotting utilities for saved runner outputs

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

Generate figures from saved results:

```bash
python - <<'PY'
from wmbo.plotting import plot_results_directory

plot_results_directory("results/debug")
PY
```

This writes figures under `results/debug/figures/`.

## TODO

- Add tests and result analysis
- Add optional LLM-backed reasoning after the rule-based agent is stable
