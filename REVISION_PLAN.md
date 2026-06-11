# RCWT Revision Plan After BRACIS Rejection

Date: 2026-06-11

## Diagnosis

The BRACIS rejection and Roundtable review point to the same root problem:
the submitted paper over-claimed the construct. The data support a controlled
measurement of **task-budget displacement** under fixed context allocation.
They do not, by themselves, prove a broad mechanism of reasoning degradation
or semantic context competition.

## Revised thesis

RCWT is a protocol for measuring how model accuracy changes when coordination
content consumes a fixed per-call context budget and therefore reduces the
remaining task block. In the main task, the observed cliff is best interpreted
as a task-specific residual-budget effect. A new intact-task ablation keeps the
full task block present while increasing coordination tokens; it shows no
accuracy drop across the tested models and ratios. This supports the
displacement interpretation and narrows the claim.

## Reviewer-to-fix matrix

| Critique | Fix in revised artifact |
|---|---|
| "Context competition" may just be truncation | Add `src/rcwt_intact_ablation.py` and `results/intact_ablation/`; revise claim to task-budget displacement. |
| Task 4 is undefined/unsupported | Remove Task 4 from the manuscript claims; keep any script as exploratory, not as evidence. |
| Protocol underspecified | Add protocol schematic, token accounting table, exact scored items, truncation rule, and reproduction commands. |
| Judge reliability weak | Add deterministic scoring for the intact ablation and state that main-task open-ended scoring remains a limitation despite cross-vendor rescoring. |
| Logistic/AIC overclaimed | Reframe logistic fit as descriptive interpolation; report that cliff-point sampling biases AIC toward threshold-like families. |
| θ overfit with few windows | Reframe θ as an empirical residual-budget estimate, not validated law. |
| Coordination homogeneous/synthetic | Add explicit limitation and distinguish structured state, transcripts, tool outputs, and contradictory coordination as future/secondary axes. |
| Cost but not benefit | Add scope boundary: RCWT measures per-call cost, not net multi-agent value; Task 4-style coordination-as-evidence belongs to a separate benefit study. |
| Real-world production transfer unclear | State synthetic structured coordination is controlled and may understate degradation from noisy transcripts. |
| Gemini 2.0 Flash availability | Record that Gemini 2.0 Flash is a historical run; new ablation uses Gemini 2.5 Flash because 2.0 returns 404 on 2026-06-11. |

## Acceptance criteria for the revision

- No orphan "Task 4" claim in manuscript.
- No title/abstract claim that broad reasoning competition is proven.
- New intact-task ablation is documented, reproducible, and cited.
- Artifact has a clear command path for tables and ablation.
- Paper distinguishes: `implemented`, `measured`, `interpreted`, and `future work`.

## Adversarial validation notes

External OpenAI review was attempted through the standard helper, but 1Password
authorization timed out. A local hard role-switch review was used instead. It
identified three concrete risks that were fixed before commit:

- The intact-task ablation wording called the result "decisive"; this was
  narrowed to "directly constrains" to avoid overstating a single ablation.
- The fixed-budget protocol still lacked exact construction details; the paper
  now states the `cl100k_base` token accounting and repeat/prefix-truncation
  rule for both coordination and task/reference blocks.
- The manual bibliography entry for `Towards a Science of Scaling Agent
  Systems` used the wrong arXiv id; it now matches arXiv:2512.08296.

Final pre-submission cleanup after the Weak Accept re-review:

- Removed response-letter language from the manuscript body.
- Defined the intact-ablation ratio as `c/(c+t)` and distinguished it from `c/W`.
- Reported task tokens, coordination tokens, estimated prompt sizes, calls, field-level denominators, and Wilson intervals for the intact ablation.
- Qualified the ceiling result as no large detected semantic-interference effect in an extraction-style intact-evidence setting, not proof of zero effect or a full test of the original task demand.
- Moved the Gemini availability note from operational error language to neutral paper language.
- Expanded the judge-calibration threat to state the scoring-method asymmetry between the open-ended main task and deterministic intact ablation.
