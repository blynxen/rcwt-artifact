# RCWT Artifact Project State

Date: 2026-06-11

## Current state

The artifact is in a rejection-response revision branch: `revise/rejection-response`.
The revised manuscript and artifact now support a narrower claim: RCWT measures fixed-budget task-budget displacement from coordination content, not a general semantic competition law.

## Important files

- `paper/bracis_24828_submission-v2.7-revision_2.md` — revised manuscript source.
- `paper/bracis_24828_submission-v2.7-revision_2.pdf` — regenerated PDF.
- `paper/bracis_lncs_v2.7/main.tex` — regenerated LNCS LaTeX.
- `src/rcwt_intact_ablation.py` — intact-task ablation runner.
- `src/rcwt_intact_scoring.py` — deterministic JSON scoring.
- `src/rescore_intact_ablation.py` — deterministic rescore path.
- `results/intact_ablation/` — 150-call intact-task ablation outputs.
- `REVISION_PLAN.md` — reviewer-to-fix matrix.
- `Makefile` — verification and build targets.

## Revised claim boundaries

Supported:

- Under fixed context budget, coordination content can displace or truncate task evidence.
- The main cliff is best explained as a residual task-budget effect for the tested recall task.
- When the full task/reference block remains present in the new ablation, the tested models stay at ceiling accuracy; this rules out a large cliff-sized semantic-interference effect in this setup, not small effects.

Not claimed:

- Coordination is net harmful.
- RCWT measures session-level multi-agent value.
- The fitted logistic curve is a universal mechanism.
- The residual reserve parameter is stable across tasks, models, or windows.
- Semantic interference is absent in general or exactly zero in the intact ablation.

## Next action

Publish the branch and use the regenerated PDF as the revised artifact baseline. If preparing a new conference submission, the next scientific step is a pre-registered task-complexity ladder plus human/judge agreement for the original open-ended scoring path.
