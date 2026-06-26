# RP-WMBO

Initial repository for a research project on world-model reasoning for black-box optimisation.

This repository is still at an early implementation stage. The current commit adds a deterministic rule-based WMBO reasoning agent on top of the existing benchmark, surrogate, acquisition, and baseline optimiser utilities.

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

## TODO

- Add experiment runner logic
- Add configs, tests, and result analysis
- Add optional LLM-backed reasoning after the rule-based agent is stable
