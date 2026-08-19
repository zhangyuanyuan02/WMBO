# WMBO V5.1 RC3 Paper Ablations

This package adds three controlled ablation experiments on top of the frozen
V5.1 RC3 scientific policy.  The GP surrogate, candidate generators, budgets,
initial Sobol design, macro mechanics, and objective/reward definitions are
unchanged unless explicitly stated below.

## Experiment 1: Landscape-variable groups

Five leave-one-group-out variants are provided:

- `v51_ablate_smoothness`
- `v51_ablate_modality`
- `v51_ablate_curvature`
- `v51_ablate_geometry`
- `v51_ablate_identifiability`

Ablated quantities are represented as unknown/neutral evidence; they are not set
to a numeric zero. Property and regime posteriors are recomputed from the masked
world-model view. Operator-internal search geometry is not removed, so this
experiment isolates information available to the policy rather than crippling
CMA/TuRBO implementations.

## Experiment 2: Complete landscape/world-model ablation

`v51_no_landscape` removes all five landscape-information groups, sets the
landscape and geometry routing weights to zero, and disables structured
hypothesis tracking. The GP surrogate, candidate evidence, phase, objective
progress, global strategy reward/trust, and the same portfolio of optimisers are
retained.

This asks whether explicit landscape reasoning adds value beyond the common GP +
candidate + history machinery.

## Experiment 3: Routing ablations

Two variants are provided:

- `v51_route_context_only`: keeps current landscape/geometry/candidate evidence
  and the RC3 legality/safety gates, but neutralises strategy-history scores and
  RC3 soft routing penalties.
- `v51_route_static_balanced`: bypasses adaptive score-based operator choice and
  chooses the least-used currently eligible local operator. RC3 feasibility,
  sparse-exploration and macro legality gates remain unchanged so the comparison
  isolates adaptive routing rather than safety constraints.

## Formal suite

All configs use the same formal suite as V5.1 RC3:

- 16 BBOB functions
- dimensions 5 and 10
- instance 1
- seeds 0, 1, 2
- budget `20 * d`
- initial Sobol design `2 * d`
- candidate pool 2048

The existing RC3 96-run formal result should be reused as the paired reference.
The eight new variants therefore require 768 new runs.

Run all ablations:

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
$env:OPENBLAS_NUM_THREADS="1"

python run_benchmark.py `
  --config configs/paper_ablations/v51_all_paper_ablations.yaml `
  --workers 8
```

Or run any individual YAML under `configs/paper_ablations/`.

## Logging

Each non-initial WMBO decision records:

```json
"ablation": {
  "landscape_groups": ["geometry"],
  "routing": "adaptive"
}
```

For landscape ablations, `landscape_descriptor` is the masked descriptor actually
seen by the policy and `landscape_descriptor_unablated` stores the diagnostic
unmasked descriptor. The latter is logging-only and is not supplied to the
router.

## Recommended analysis

For every variant, pair against the frozen RC3 result by
`(function, dimension, instance, seed)` and report:

- log10 final simple regret
- relative-regret AUC
- 25%, 50%, 75%, 100% budget checkpoints
- W/T/L
- median paired delta
- bootstrap 95% CI
- Wilcoxon signed-rank test

For the five landscape-group tests, apply Holm correction across the five primary
comparisons. Also report results by BBOB landscape family.
