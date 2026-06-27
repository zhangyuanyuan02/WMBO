# RP-WMBO

Initial repository for a research project on world-model reasoning for black-box optimisation.

This repository is still at an early implementation stage. The current commit adds a WMBO control state for budget-aware strategy gating, trust updates, cooldowns, and hypothesis tracking.

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
- WMBO control state with:
  - budget phase detection
  - strategy trust scores
  - exploration cooldowns
  - strategy gating and fallback
  - hypothesis tracking
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

## WMBO control options

The runner accepts optional WMBO control settings through the `optimizer.wmbo_control` section of a YAML config:

```yaml
optimizer:
  wmbo_control:
    early_fraction: 0.35
    late_fraction: 0.70
    trust_window: 5
    failure_cooldown_trials: 2
    hypothesis_window: 3
```

These options only affect the rule-based WMBO controller at this stage. No external LLM backend is used yet.

## TODO

- Add tests and result analysis
- Add optional LLM-backed reasoning after the control state is stable
