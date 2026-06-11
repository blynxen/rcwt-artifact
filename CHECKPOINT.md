# Checkpoint: RCWT Rejection Revision

Date: 2026-06-11
Branch: `revise/rejection-response`
Repository: `blynxen/rcwt-artifact`

## Completed

- Reframed the manuscript from broad "context competition" to fixed-budget **task-budget displacement**.
- Added `REVISION_PLAN.md` with a reviewer-to-fix matrix for BRACIS and Roundtable critiques.
- Added an intact-task ablation that keeps the full task/reference block present while adding coordination tokens by increasing total prompt length.
- Added deterministic scoring for the intact-task ablation and a rescore path.
- Regenerated the LNCS LaTeX and PDF manuscript.
- Updated README with setup, reproduction commands, result files, and model availability notes.
- Added `Makefile` targets for paper build, Python compile checks, rescore, and full verification.

## New evidence

- Full intact-task ablation: 150 calls across GPT-4.1-mini, Claude Haiku 4.5, and Gemini 2.5 Flash.
- Tested ratios: `0, 0.5, 0.75, 0.9, 0.95`.
- Tested orders: `coord_first`, `reason_first`.
- Result: all deterministic scores are `1.0` after rescore.
- Observed total cost in saved CSV: `0.4929728` USD.

## Verification

Executed successfully:

```bash
make PYTHON=<python-with-project-dependencies> verify
```

This ran:

- Python compile checks for the new scripts.
- Deterministic rescore of `results/intact_ablation/rcwt_intact_ablation_responses.jsonl`.
- Pandoc + `latexmk` PDF build.

Additional checks:

- `git diff --check` passed.
- No `Task 4` references remain in the revised manuscript Markdown or generated PDF text.
- Tracked SVG/PDF/HTML remote-asset scan found only the standard SVG namespace `http://www.w3.org/2000/svg`; no external `href`, `src`, `url(http...)`, `https://`, or `cdnjs` asset references.


## Follow-up cleanup after Weak Accept re-review

- Removed paper-body language that referred to the review process.
- Defined intact-ablation ratio as `c/(c+t)` and reported token counts for each ratio.
- Added call/field-level denominators and Wilson intervals for the ceiling result.
- Qualified the intact ablation as ruling out a large cliff-sized effect in an extraction-style intact-evidence setting, not small semantic effects or harder-task interference.
- Neutralized model-availability wording in the paper.
- Expanded judge-calibration limitations to include scoring-method asymmetry.
- Applied final neutral wording polish: `The contributions are`, `artifact aggregate files`, and complementary deterministic/human validation for the main judge path.

## Known limits

- The revision does not claim to solve net multi-agent benefit. RCWT remains a single-call cost-side measurement primitive.
- Main-task scoring still uses an LLM judge; this is retained as a limitation. The new intact-task ablation uses deterministic scoring.
- Gemini 2.0 Flash appears only as historical fixed-budget evidence. New reruns use Gemini 2.5 Flash because Gemini 2.0 Flash returned provider 404 on 2026-06-11.
- External OpenAI adversarial review helper failed because 1Password authorization timed out; local adversarial role-switch review was used instead.
