# RCWT Artifact Project State

Date: 2026-07-07

## Current state

The artifact is on branch `revise/rejection-response` in repository `brendaclelis/rcwt-artifact`.
The revised manuscript supports the narrower claim that RCWT measures fixed-budget task-budget displacement from coordination content, not a general semantic competition law.

Two publication-format branches of work exist in the working tree:

1. An ENIAC/SBC anonymous draft from the prior formatting attempt.
2. A newer arXiv/preprint draft using a generic NeurIPS-like layout with no visible conference-submission footer text.

## Important files

Core reviewed manuscript:

- `paper/bracis_24828_submission-v2.7-revision_2.md` — reviewed scientific manuscript source.
- `paper/bracis_24828_submission-v2.7-revision_2.pdf` — BRACIS/LNCS-style regenerated PDF.
- `paper/bracis_lncs_v2.7/main.tex` — BRACIS/LNCS LaTeX.

arXiv/preprint artifacts:

- `paper/arxiv/main.md` — generated arXiv Markdown source.
- `paper/arxiv/main.tex` — generated arXiv LaTeX source.
- `paper/arxiv/template.tex` — Pandoc wrapper for the generic preprint layout.
- `paper/arxiv/preprint_2026.sty` — generic NeurIPS-like preprint style; first-page notice is only `Preprint.`.
- `paper/arxiv/figures/rcwt_cross_provider_arxiv.png` — restored, more legible RCWT figure.
- `paper/arxiv/main.pdf` — build-local copy of the arXiv PDF.
- `paper/rcwt_arxiv_preprint.pdf` — final arXiv/preprint PDF in the repo.
- `/Users/brendalelis/Downloads/rcwt_arxiv_preprint.pdf` — final arXiv/preprint PDF copy for manual review/upload.
- `paper/rcwt_arxiv_source.zip` — TeX source package containing `main.tex`, `preprint_2026.sty`, and the figure.
- `src/build_arxiv_preprint_markdown.py` — generator for the arXiv Markdown source.
- `src/plot_rcwt_arxiv_figure.py` — generator for the restored arXiv figure.

ENIAC/SBC artifacts from the prior attempt:

- `paper/eniac_sbc_anonymous/main.md` — generated ENIAC/SBC Markdown source.
- `paper/eniac_sbc_anonymous/main.tex` — generated ENIAC/SBC LaTeX source.
- `paper/eniac_sbc_anonymous/template.tex` — Pandoc wrapper for SBC style.
- `paper/eniac_sbc_anonymous/sbc-template.sty` — SBC style copied from the local ENIAC template ZIP.
- `paper/rcwt_eniac_sbc_anonymous.pdf` — generated ENIAC/SBC anonymous PDF.
- `src/build_eniac_sbc_markdown.py` — generator for the ENIAC/SBC submission source.

Build entrypoint:

- `Makefile` — includes `verify`, `verify-eniac`, and `verify-arxiv`.

## Revised claim boundaries

Supported:

- Under fixed context budget, coordination content can displace or truncate task evidence.
- The main cliff is best explained as a residual task-budget effect for the tested recall task.
- When the full task/reference block remains present in the intact-task ablation, the tested models stay at ceiling accuracy; this rules out a large cliff-sized semantic-interference effect in this extraction-style setup, not small effects or harder-task interference.

Not claimed:

- Coordination is net harmful.
- RCWT measures session-level multi-agent value.
- The fitted logistic curve is a universal mechanism.
- The residual reserve parameter is stable across tasks, models, or windows.
- Semantic interference is absent in general, exactly zero in the intact ablation, or ruled out for the original open-ended task format.

## arXiv/preprint state

- Format: generic NeurIPS-like single-column preprint layout, letter paper.
- Footer/notice: first page says only `Preprint.`.
- Authorship: currently `Anonymous Authors`; must be replaced before a real arXiv posting if/when approved.
- Figure: restored and regenerated as `paper/arxiv/figures/rcwt_cross_provider_arxiv.png`.
- Removed table: the old intact-task ablation table with all-1.000 cells is removed and replaced with prose.
- Limitations: renamed from `Threats to Validity` and rewritten as paragraphs.
- Appendix: `Appendix A. Reproduction commands` is present.
- Verification: `make verify-arxiv` passes; the final PDF and source ZIP were compiled and visually checked.

## Next action

Confirm with Terra whether and when to post to arXiv, then replace `Anonymous Authors` with the approved author list. For a source upload, use `paper/rcwt_arxiv_source.zip`; for manual PDF review, use `/Users/brendalelis/Downloads/rcwt_arxiv_preprint.pdf`.

If pushing this branch to `brendaclelis`, use the personal GitHub identity protocol and restore the default CloudWalk identity after push.
