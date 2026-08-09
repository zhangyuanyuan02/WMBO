# RP-WMBO

Initial repository for a research project on world-model reasoning for black-box optimisation.

This repository is still at an early implementation stage. The current commit adds YAML configuration files for OpenAI-compatible LLM WMBO experiments while keeping API keys out of version control.

## Structure

```text
configs/         Reproducible experiment settings
data/            Versioned benchmark inputs and checksums
julia/           Pinned PowerModels/Ipopt backend
src/wmbo/        Python package source code
tests/           Unit and optional Julia integration tests
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
- PGLib-OPF v23.07 case14 TYP/API benchmarks through a persistent Julia AC power-flow backend

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

## OPF benchmark

The OPF benchmark vendors the PGLib-OPF v23.07 IEEE case14 TYP and API cases. Each benchmark has six controls in a fixed order: active power `Pg2` for the one dispatchable non-slack generator, followed by voltage setpoints `Vg1`, `Vg2`, `Vg3`, `Vg6`, and `Vg8` for all online generator buses, including the slack bus.

Install Julia 1.10 LTS with Juliaup, then resolve the pinned PowerModels 0.19.9 and Ipopt 1.4.1 environment:

```powershell
juliaup add 1.10
julia +1.10 julia/setup.jl
```

List all available benchmarks without starting Julia:

```powershell
python run_benchmark.py --list-benchmarks
```

Run the 12-evaluation smoke configuration or the complete 2-case, 7-method, 10-seed suite:

```powershell
python run_benchmark.py --config configs/opf_debug.yaml
python run_benchmark.py --config configs/opf_benchmark.yaml
```

Every method receives the same pinned, strictly feasible, suboptimal first point. The checked-in starts were generated deterministically with `julia/generate_pgvg_starts.jl` and have reference-cost gaps of about 40.32% for TYP and 10.75% for API. A full AC-OPF is solved once to obtain the reference cost and controls; optimisation candidates set `Pg+Vg` and run AC power flow only. Each observation records the controls, cost, reference cost, feasibility, convergence, violations, and timing. Constrained optimisers model the normalised cost gap and the log constraint ratio separately, then combine objective acquisition with predicted probability of feasibility. The diagnostic scalar target is `normalised_cost_gap + penalty_weight * max(0, max_normalized_violation / feasibility_tolerance - 1)^2`; a non-converged power flow receives `1e6`. OPF comparisons rank methods by `primary_score` (best strictly feasible gap), with the legacy penalised best retained for historical analysis.

Run the optional real Julia integration tests after setup:

```powershell
$env:WMBO_RUN_OPF_INTEGRATION = "1"
python -m pytest tests/test_opf_integration.py
```

## WMBO control options

The runner accepts optional WMBO control settings through the `optimizer.wmbo_control` section of a YAML config:

```yaml
optimizer:
  wmbo_control:
    early_fraction: 0.35
    late_fraction: 0.70
    candidate_options_per_strategy: 3
    trust_window: 5
    failure_cooldown_trials: 2
    hypothesis_window: 3
    information_gain_weight: 0.35
    hypothesis_support_probability: 0.80
    hypothesis_rejection_probability: 0.20
    hypothesis_support_likelihood_ratio: 3.0
    hypothesis_failure_likelihood_ratio: 0.50
```

The control state is still applied after an LLM decision, so the LLM proposes a strategy but does not directly control the optimiser.

The rule proposer supports a backwards-compatible legacy mode and an experimental
constraint-aware continuous scorer:

```yaml
optimizer:
  options:
    rule_policy:
      mode: continuous_v4
      landscape_weight: 0.45
      candidate_weight: 0.35
      history_weight: 0.20
      temperature: 0.20
```

`legacy_v1` remains the default for reproducibility, while `continuous_v2` is
retained for ablation. `continuous_v3` adds final-regret-aware phase priors.
`continuous_v4` strengthens the trust-region prior and turns successful-point
follow-up into a scored competition between `exploit_ei` and `trust_region`
instead of forcing one of them. Continuous policies mask controller-forbidden strategies and log
score components and selected-candidate evidence with every observation.


The P0 reasoning path now keeps both human-readable labels and calibrated
categorical posteriors for smoothness, modality, curvature, and anisotropy.
Each descriptor also records credible intervals, posterior confidence, and
world-model entropy.

After the initial design, each WMBO strategy exposes optimisation,
confirmation, and falsification candidates. Candidate ranking combines
expected improvement with approximate information gain. Hypotheses are updated
with likelihood ratios only when an evaluation lies in the stated region or was
explicitly targeted as a confirmation/falsification test. Hypotheses that reach
their deadline without decisive relevant evidence become `inconclusive` rather
than being rejected automatically.

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

- Add N-1 and load-scenario OPF extensions after the deterministic case14 benchmark is stable
