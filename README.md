# RP-WMBO

Initial repository for a research project on world-model reasoning for black-box optimisation.

This repository is still at an early implementation stage. The current commit adds a lightweight surrogate model and acquisition utilities on top of the existing synthetic benchmark functions.

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

## TODO

- Implement baseline optimisers
- Implement the WMBO reasoning agent
- Add experiment runner logic
- Add configs, tests, and result analysis
