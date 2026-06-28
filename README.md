# RP-WMBO

Initial repository for a research project on world-model reasoning for black-box optimisation.

This repository is still at an early implementation stage. The current commit adds YAML configuration files for OpenAI-compatible LLM WMBO experiments while keeping API keys out of version control.

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
- Optional OpenAI-compatible LLM decision backend with:
  - chat-completions HTTP client
  - structured world-model prompt construction
  - JSON response parsing and validation
  - retry/timeout handling
  - fallback to the rule-based agent
- Minimal benchmark runner with JSON/CSV output
- YAML configuration files for reproducible rule-based and LLM-assisted runs
- LLM debug config and local secrets template
- Plotting utilities for saved runner outputs

## Examples

Run the default CLI settings:

```bash
python run_benchmark.py --benchmarks branin --methods random,wmbo --seeds 0 --budget 10
```

Run from a checked-in rule-based config:

```bash
python run_benchmark.py --config configs/debug.yaml
```

Run the LLM-assisted debug config. Without an API key, WMBO falls back to the rule-based agent by default:

```bash
python run_benchmark.py --config configs/llm_debug.yaml
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

The control state is still applied after an LLM decision, so the LLM proposes a strategy but does not directly control the optimiser.

## Optional LLM backend

The default WMBO method remains rule based unless a config or CLI option enables the LLM agent. The checked-in LLM config uses an environment variable for the key:

```bash
export OPENAI_API_KEY=...
python run_benchmark.py --config configs/llm_debug.yaml
```

For compatible providers, either edit a local copy of the config or override the checked-in config from the CLI:

```bash
export SILICONFLOW_API_KEY=...
python run_benchmark.py \
  --config configs/llm_debug.yaml \
  --llm-base-url https://api.siliconflow.cn/v1 \
  --llm-api-key-env SILICONFLOW_API_KEY \
  --llm-model Qwen/Qwen2.5-72B-Instruct
```

For private local settings, copy the example secrets file and keep the copy untracked:

```bash
cp configs/llm_secrets.example.yaml configs/llm_secrets.yaml
python run_benchmark.py --config configs/llm_secrets.yaml
```

The ignored `configs/llm_secrets.yaml` file may contain provider-specific keys or model names for local experiments. Prefer `api_key_env` over writing keys directly in YAML.

## TODO

- Add tests and result analysis
