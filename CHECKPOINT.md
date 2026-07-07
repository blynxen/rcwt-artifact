# Checkpoint: RCWT arXiv preprint draft

Date: 2026-07-07
Branch: `revise/rejection-response`
Repository: `brendaclelis/rcwt-artifact`

## Completed

- Built an arXiv/preprint manuscript variant from `paper/bracis_24828_submission-v2.7-revision_2.md`.
- Kept the revised scientific framing: RCWT measures fixed-budget task-budget displacement, not a general semantic-interference law.
- Restored the cross-provider RCWT figure and regenerated it as a more legible preprint figure:
  - `paper/arxiv/figures/rcwt_cross_provider_arxiv.png`
  - generator: `src/plot_rcwt_arxiv_figure.py`
- Removed the low-information intact-task ablation table with all-1.000 cells from the arXiv draft; the result is now stated in prose with denominators and Wilson intervals.
- Renamed `Threats to Validity` to `Limitations` and rewrote that section as paragraphs rather than bold bullet-style entries.
- Restored the reproduction-command appendix in the arXiv/preprint version.
- Added a generic NeurIPS-like preprint layout without visible conference-submission footer text:
  - `paper/arxiv/preprint_2026.sty`
  - `paper/arxiv/template.tex`
  - first-page notice is only `Preprint.`
- Added `src/build_arxiv_preprint_markdown.py` to generate the arXiv Markdown source from the reviewed paper.
- Added Makefile targets:
  - `build-arxiv-figure`
  - `build-arxiv-paper`
  - `verify-arxiv`
- Generated final artifacts:
  - PDF: `paper/rcwt_arxiv_preprint.pdf`
  - Upload copy: `/Users/brendalelis/Downloads/rcwt_arxiv_preprint.pdf`
  - TeX source ZIP: `paper/rcwt_arxiv_source.zip`

## Verification

Executed successfully:

```bash
make verify-arxiv
```

This ran:

- Python compile checks, including both paper generators and the arXiv figure generator.
- Figure generation from the manuscript aggregate values.
- arXiv Markdown generation.
- Pandoc conversion to LaTeX.
- Forced `latexmk` PDF build.
- PDF and source ZIP copy/package steps.

Additional checks performed:

- Rendered the final PDF with `pdftoppm` and visually inspected pages 1, 4, 5, 6, 7, and 8.
- Page 1 shows only the footer notice `Preprint.`; no `Submitted to ...` or `Work in progress` footer appears.
- The restored figure is legible in the PDF.
- The old intact-task ablation table caption/text is absent from the PDF.
- The `Limitations` section appears as prose paragraphs.
- `Appendix A. Reproduction commands` appears in the PDF.
- `paper/rcwt_arxiv_source.zip` was unpacked and compiled successfully with `latexmk` in a clean temporary directory.
- `git diff --check` passed.

## Known limits

- The arXiv draft still uses `Anonymous Authors` because author disclosure/publication approval with Terra is pending.
- Bibliographic references still cite NeurIPS as venue where relevant; the removed NeurIPS reference is the template/submission footer, not legitimate citation metadata.
- LaTeX reports only minor layout warnings: one underfull vbox and one 1.4pt overfull hbox in an appendix command line. Visual inspection showed no clipping or unreadable text.
- Existing ENIAC/SBC artifacts remain in the working tree from the prior formatting attempt.

## Next action

Before actual arXiv upload, replace `Anonymous Authors` with the approved author list after the Terra/publication decision. Use `paper/rcwt_arxiv_source.zip` for a TeX-source upload or `/Users/brendalelis/Downloads/rcwt_arxiv_preprint.pdf` for manual review.
