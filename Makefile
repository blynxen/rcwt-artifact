.PHONY: build-paper check-python rescore-intact verify

PYTHON ?= python3

build-paper:
	pandoc paper/bracis_24828_submission-v2.7-revision_2.md \
		--from markdown+tex_math_dollars \
		--to latex \
		--standalone \
		--metadata documentclass=llncs \
		-o paper/bracis_lncs_v2.7/main.tex
	cd paper/bracis_lncs_v2.7 && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
	cp paper/bracis_lncs_v2.7/main.pdf paper/bracis_24828_submission-v2.7-revision_2.pdf

check-python:
	$(PYTHON) -m py_compile src/rcwt_intact_ablation.py src/rcwt_intact_scoring.py src/rescore_intact_ablation.py

rescore-intact:
	PYTHONPATH=src $(PYTHON) src/rescore_intact_ablation.py \
		--responses results/intact_ablation/rcwt_intact_ablation_responses.jsonl \
		--output-dir results/intact_ablation

verify: check-python rescore-intact build-paper
