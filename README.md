# Roundtable Context Window Test (RCWT) Artifact

This repository contains the anonymized replication artifact for the paper **Reasoning Under Context Competition: A Controlled Study of Coordination Overhead in LLM Calls**.

The artifact includes:

- experiment and analysis scripts under `src/`;
- aggregate result tables and figure data under `results/`;
- the anonymous BRACIS/LNCS manuscript source under `paper/`;
- a minimal Python requirements file.

The experiments study single-call context competition: a fixed context budget is divided between coordination content and task content, and model quality is measured as coordination overhead increases.

## Contents

```text
src/                         experiment runners and analysis scripts
results/                     aggregate CSV/JSON summaries and figures
paper/                       anonymous manuscript PDF and LNCS source
requirements.txt             Python dependencies used by the scripts
```

Raw model responses are not included in this anonymous artifact to keep the review package compact and to avoid publishing unnecessary provider-output text. The included aggregate CSV/JSON files are sufficient to reproduce the paper tables and plots.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Provider API keys are read from environment variables when rerunning experiments:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

No API keys, credentials, local machine paths, or institution-specific files are included.

## Reproducing analysis artifacts

Examples:

```bash
python src/analyze_benchmark_results.py --help
python src/analyze_output_length.py --help
python src/plot_combined.py --help
```

The main result summaries used in the paper are:

- `results/rcwt_controlled.csv`
- `results/rcwt_controlled_aggregates.json`
- `results/rcwt_curve_fits.json`
- `results/cross_benchmark_pack_summary_with_drop.csv`
- `results/w32768_summary.csv`
- `results/output_length_analysis.csv`

## Anonymity note

This repository is prepared for double-anonymous review. Names, affiliations, local paths, credential loaders, and unrelated project references have been removed from the artifact copy.
