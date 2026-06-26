# RP-WMBO

Initial repository for a research project on world-model reasoning for black-box optimisation.

This repository is still at an early implementation stage. The current commit adds a minimal experiment runner and command-line path for saving benchmark outputs.

## Structure

```text
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

## Example

```bash
python run_benchmark.py --benchmarks branin --methods random,wmbo --seeds 0 --budget 10
```

Outputs are written under `results/` by default.

## TODO

- Add configs, tests, and result analysis
- Add plotting utilities for saved outputs
- Add optional LLM-backed reasoning after the rule-based agent is stable
