# RCWT Artifact

Replication artifact for **RCWT: Measuring Task-Budget Displacement from
Coordination Content in LLM Calls**.

The original BRACIS submission framed the result as "reasoning under context
competition". The revised artifact narrows the claim: RCWT measures how accuracy
changes when coordination content consumes a fixed per-call context budget and
therefore reduces the remaining task block. A new intact-task ablation separates
this displacement effect from semantic interference.

## Contents

```text
src/                         experiment runners and analysis scripts
results/                     aggregate CSV/JSON summaries and figures
paper/                       manuscript source
REVISION_PLAN.md             reviewer-to-fix matrix after rejection
requirements.txt             Python dependencies
```

## Key findings in this artifact

1. **Fixed-budget RCWT:** at `W=4096`, the main context-dependent recall task
   stays near baseline through moderate overhead and degrades sharply when the
   residual task block falls to a few hundred tokens.
2. **Residual-budget interpretation:** window-scaling summaries are consistent
   with a task-specific remaining-task-budget estimate, not a fixed percentage
   threshold. This is descriptive, not a universal law.
3. **Intact-task ablation:** when the full task/reference block is kept intact
   and coordination tokens are added by increasing total prompt length, accuracy
   stays at 1.000 for GPT-4.1-mini, Claude Haiku 4.5, and Gemini 2.5 Flash
   across tested coordination ratios up to 95%. This rules out a large
   cliff-sized semantic-interference effect in this setup, not small effects.
4. **Boundary tasks:** self-contained algorithmic tasks remain stable; a
   contradictory-coordination task shows model-specific semantic distraction;
   DROP-style packs require much larger residual task budgets.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provider API keys are read from environment variables:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

No API keys, credentials, local machine paths, or institution-specific files are
included.

## Reproducing the intact-task ablation

```bash
PYTHONPATH=src python src/rcwt_intact_ablation.py \
  --models gpt-4.1-mini,claude-haiku-4-5-20251001,gemini-2.5-flash \
  --ratios 0,0.5,0.75,0.9,0.95 \
  --orders coord_first,reason_first \
  --n-trials 5 \
  --output-dir results/intact_ablation

PYTHONPATH=src python src/rescore_intact_ablation.py \
  --responses results/intact_ablation/rcwt_intact_ablation_responses.jsonl \
  --output-dir results/intact_ablation
```

Outputs:

- `results/intact_ablation/rcwt_intact_ablation.csv`
- `results/intact_ablation/rcwt_intact_ablation_aggregates.json`
- `results/intact_ablation/rcwt_intact_ablation_responses.jsonl`

In the intact ablation, `target_ratio = c/(c+t)`, where `t=698` is the
intact task/reference block. Ratios `0,0.5,0.75,0.9,0.95` correspond to
estimated prompt sizes `702,1401,2797,6985,13965` construction tokens. Each
model-ratio cell pools 10 calls and 80 binary field decisions.

## Existing result summaries used by the paper

- `results/rcwt_controlled.csv`
- `results/rcwt_controlled_aggregates.json`
- `results/rcwt_curve_fits.json`
- `results/cross_benchmark_pack_summary_with_drop.csv`
- `results/w32768_summary.csv`
- `results/output_length_analysis.csv`
- `results/intact_ablation/rcwt_intact_ablation_aggregates.json`

## Model availability note

The submitted fixed-budget Gemini rows used `gemini-2.0-flash`. On
2026-06-11, rerunning that model returned a provider 404 stating that the model
is no longer available. The new intact-task ablation therefore uses
`gemini-2.5-flash`. Historical aggregate files are retained for reproducibility
of the submitted tables; new confirmatory reruns should use current model IDs.

## Scope

RCWT is a local single-call measurement primitive. It does not measure the net
benefit of multi-agent coordination, turn scheduling, tool reliability, memory
retrieval policy, or long-running session dynamics. Those require separate
session-level experiments.
