---
title: "RCWT: Measuring Task-Budget Displacement from Coordination Content in LLM Calls"
author: "Anonymous Authors"
documentclass: llncs
classoption: runningheads
geometry: margin=1in
header-includes:
  - \usepackage{booktabs}
  - \usepackage{amsmath,amssymb}
  - \usepackage{graphicx}
  - \usepackage{microtype}
  - \usepackage{array}
  - \setlength{\tabcolsep}{4pt}
  - \renewcommand{\arraystretch}{0.95}
  - \pagestyle{plain}
---

\begin{abstract}
Multi-agent and memory-augmented LLM systems often place coordination content---shared state, prior discussion, tool outputs, summaries, and role instructions---inside the same finite prompt used for the current task. This creates a practical allocation problem: every token spent on coordination is unavailable to the task block when a call is assembled under a fixed context budget. We introduce the Roundtable Context Window Test (RCWT), a controlled protocol for measuring this task-budget displacement effect. RCWT varies coordination content while controlling total budget, position order, task family, and scoring. In the main context-dependent recall task at $W=4096$, three commercial models remain near baseline through moderate overhead and then degrade sharply once the residual task block falls to a few hundred tokens. Window-scaling summaries are consistent with a task-specific residual-budget interpretation rather than a fixed percentage threshold, but we treat this as descriptive evidence rather than a universal law. To test whether the fixed-budget cliff persists when task evidence remains intact, we add an intact-task ablation: the full task/reference block is kept present while coordination tokens increase by expanding total prompt length. In that setting, accuracy remains at 1.000 across GPT-4.1-mini, Claude Haiku 4.5, and Gemini 2.5 Flash up to a 95\% coordination ratio. This ablation narrows the claim: the main RCWT cliff is best read as task-budget displacement, not as proof that coordination volume alone causes semantic interference in the original open-ended task. RCWT is therefore a measurement primitive for context-allocation budgeting, not a complete theory of multi-agent benefit or session-level coordination.
\keywords{Large language models \and Multi-agent systems \and Context windows \and Benchmarking \and Prompt allocation}
\end{abstract}

## 1. Introduction

Modern LLM applications assemble many kinds of information into a single call: task instructions, retrieved documents, memory summaries, prior agent messages, tool observations, role prompts, and shared state \cite{ram-etal-2023-context,toolformer,autogen,guo2024multiagents}. The application-level question is not only whether a model advertises a large context window, but how much of the current call is allocated to the task block that must be read and acted on. Prior long-context work shows that models do not use all positions uniformly and that longer inputs can degrade performance even when relevant information is available \cite{lostmiddle,longbench,ruler,du2025contextlength}.

This paper studies a narrow, operational version of that problem. For a single model call with context budget $W$, let $c$ be coordination content and let $W-c$ be the remaining task block. We estimate

$$
R_M(c,W,T) \in [0,1],
$$

where $M$ is the model and $T$ is the task family. We use the term **task-budget displacement** for the main effect: under fixed $W$, increasing coordination content can physically reduce or truncate the task evidence needed by the current call.

We introduce the Roundtable Context Window Test (RCWT). RCWT is a single-call benchmark protocol that varies coordination allocation while controlling prompt order and scoring. It intentionally excludes full multi-agent session dynamics such as turn scheduling, retrieval policy, memory writes, tool failures, and agent topology. Those factors matter, but mixing them into the same experiment would obscure the local allocation effect.

The contributions are:

1. **A controlled protocol.** RCWT varies coordination allocation under fixed budget with position control, explicit token accounting, and task-level scoring.
2. **A fixed-budget displacement result.** On a technical-specification recall task, accuracy remains high at moderate overhead and drops sharply only when the residual task block becomes very small.
3. **A truncation-disambiguating ablation.** When the full task block remains intact and total prompt length grows to accommodate coordination, we detect no cliff-sized degradation across tested models and ratios.
4. **Task dependence and boundary evidence.** Self-contained tasks remain stable, contradictory coordination can produce model-specific distraction, and passage-heavy DROP packs require much larger residual task budgets.
5. **A scoped engineering implication.** Coordination context should be budgeted against task-specific residual needs. RCWT does not measure the net benefit of coordination.

## 2. Related Work

**Long-context behavior.** The Lost in the Middle result showed that LLMs can fail to use relevant information depending on position \cite{lostmiddle}. LongBench and RULER broaden long-context evaluation across retrieval, QA, summarization, synthetic tasks, and configurable sequence lengths \cite{longbench,ruler}. RCWT differs by holding a call budget fixed and varying allocation between coordination-like content and task evidence.

**Attention and systems bottlenecks.** Long-context use is constrained by attention and KV-cache costs. LongLoRA, FlashAttention, Longformer, and BigBird address feasibility through training or attention mechanisms \cite{longlora,flashattention,longformer,bigbird}. RCWT asks a complementary application-level question: given a feasible context, how much of it is left for the task block?

**Context compression.** Prompt compression systems such as LLMLingua and LLMLingua-2 reduce token count while trying to preserve task-relevant information \cite{jiang-etal-2023-llmlingua,llmlingua2}. RCWT can evaluate whether compression moves a call away from a high-displacement regime.

**Multi-agent coordination.** Multi-agent LLM systems introduce communication, role, orchestration, verification, and memory overhead \cite{mast,scalingagents}. They may also produce benefits through decomposition, critique, and specialized roles. RCWT measures only the per-call token cost side of that ledger. It does not estimate net system value.

## 3. Method

### 3.1 Fixed-budget RCWT

The main RCWT task is context-dependent recall over a technical software specification. The task asks for a structured technical analysis whose scored facts must be recovered from a reference block. The coordination block is synthetic but structured like multi-agent shared context: role/protocol text, agent messages, shared propositions, and tool schemas.

Token accounting is deterministic in the artifact. Prompt construction uses the `cl100k_base` tokenizer to size blocks. For each target allocation, the coordination template is repeated or prefix-truncated to the requested coordination-token count. The reference template is likewise repeated or prefix-truncated to the remaining task-token count. Provider-reported token counts are saved with every trial, but the allocation itself is built from this common tokenizer so the sweep is reproducible across providers.

The main experiment uses $W=4096$ and coordination proportions

$$p \in \{0,25,50,75,90,92,94,96,98\}\%.$$

Each condition is run in two orders: `coord_first` and `reason_first`. The primary suite uses $N=20$ trials per cell. The original open-ended responses are scored with a binary fact-checking judge over 10 items, with two known floor-effect items excluded from the effective score. The 8 effective items are:

1. three encryption options;
2. recommendation of per-document symmetric key;
3. PostgreSQL LISTEN/NOTIFY 8KB payload limit;
4. Redis Pub/Sub approximately 100K messages/sec;
5. Redis Streams approximately 5ms latency versus Pub/Sub approximately 1ms;
6. ProseMirror as the rich-text recommendation;
7. backend team needs 1 week CRDT ramp-up;
8. WebSocket/infrastructure migration takes about 2--3 weeks.

The two excluded items are `crdt_throughput` and `mls_rfc`, which behaved as floor-effect parsing items in the original baseline. The artifact reports both raw and effective scores.

### 3.2 Candidate curves and residual-budget summary

For the main task, we fit logistic, power-law, exponential, quadratic, and piecewise-linear descriptive curves. The logistic form is

$$
R_M(c,W,T)=\frac{R_0}{1+\exp(k(c/W-p_0))}.
$$

This is not a mechanistic proof. The cliff-region sampling makes a threshold-like family likely to fit well. We use the curve as compact interpolation, not as evidence of a universal law.

We also summarize the transition with a task-specific residual budget parameter $\theta$:

$$
p_0(W)=1-\frac{\theta}{W}.
$$

This relation is an empirical description over the tested window sizes. With only a small number of window sizes, $\theta$ should be read as a calibration estimate, not as a validated invariant.

### 3.3 Intact-task ablation

We add an ablation that directly tests whether the main cliff is caused by task truncation or by semantic interference from extra coordination text. The full task/reference block is kept intact in every condition. Coordination tokens are varied around it, and the total prompt length grows accordingly. Thus, the task is never physically shortened.

The ablation uses deterministic JSON scoring rather than an LLM judge. It asks for eight exact fields matching the same reference facts used in the main task. In this ablation, the coordination ratio is $c/(c+t)$, where $t$ is the intact task/reference block; this differs from the fixed-budget experiment, where the ratio is $c/W$. The measured task/reference block has $t=698$ construction tokens. Tested ratios are $0, 0.50, 0.75, 0.90, 0.95$, corresponding to 0, 698, 2,094, 6,282, and 13,262 coordination tokens and estimated prompt sizes of 702, 1,401, 2,797, 6,985, and 13,965 tokens. Both prompt orders are tested with $N=5$ per order, so each model-ratio cell pools $10$ calls and $80$ binary field decisions. Models are GPT-4.1-mini, Claude Haiku 4.5, and Gemini 2.5 Flash. The Gemini model differs from the historical main table because Gemini 2.0 Flash was unavailable during later reruns.

### 3.4 Boundary tasks and packs

We additionally retain three boundary probes:

- **Self-contained algorithmic task:** the task block contains all information needed to answer execution-trace questions; coordination is irrelevant.
- **Context-distraction task:** the coordination block contains salient project-specific decisions that conflict with broad-prior questions.
- **External benchmark packs:** GSM8K, MMLU-Pro, and DROP are grouped into 10-item packs so that task-block size can become binding under fixed $W$.

## 4. Results

### 4.1 Fixed-budget main task

Table 1 shows the main fixed-budget result. Accuracy is stable through moderate overhead, then degrades sharply when the task block becomes very small.

\begin{table}[t]
\centering
\small
\caption{Main context-dependent RCWT task at $W=4096$, pooled over position order. Scores are effective binary accuracy from the artifact aggregate files.}
\begin{tabular}{lrrrr}
\toprule
Coord. overhead & Task tokens & Gemini 2.0 Flash & Haiku 4.5 & GPT-4.1-mini \\
\midrule
0\%  & 4096 & 1.000 & 0.972 & 1.000 \\
25\% & 3072 & 0.960 & 0.906 & 0.972 \\
50\% & 2048 & 0.944 & 0.894 & 0.988 \\
75\% & 1024 & 0.894 & 0.875 & 0.981 \\
90\% & 410  & 0.659 & 0.666 & 0.853 \\
92\% & 328  & 0.404 & 0.456 & 0.600 \\
94\% & 246  & 0.012 & 0.119 & 0.406 \\
96\% & 164  & 0.012 & 0.088 & 0.319 \\
98\% & 82   & 0.000 & 0.050 & 0.337 \\
\bottomrule
\end{tabular}
\end{table}

At 90\% overhead, the drop from baseline is 34.1 percentage points for Gemini, 30.6 for Haiku, and 14.7 for GPT. This supports a fixed-budget warning: average prompt length and nominal context window are not sufficient safety signals; the residual task block matters.

### 4.2 Window scaling

A fixed-proportion account would predict similar degradation at 90\% overhead across window sizes. The observed summaries do not match that account. At $W=4096$, 90\% overhead leaves about 410 task tokens and lies near the degradation region. At $W=8192$, the same proportion leaves about 819 task tokens and produces smaller degradation.

\begin{table}[t]
\centering
\small
\caption{Window-scaling summary for the main task. $\theta$ is estimated from the $W=4096$ fit. This is descriptive calibration, not a validated invariant.}
\begin{tabular}{lrrrr}
\toprule
Model & $\theta$ & Pred. $p_0$(8K) & Obs. $p_0$(8K) & Obs. $p_0$(16K) \\
\midrule
Gemini 2.0 Flash & 356 & 95.7\% & 96.0\% & 97.9\% \\
Claude Haiku 4.5 & 340 & 95.8\% & 96.2\% & 98.1\% \\
GPT-4.1-mini & 250 & 96.9\% & 97.1\% & 98.4\% \\
\bottomrule
\end{tabular}
\end{table}

The relation is consistent with residual-budget displacement over these runs. It is not strong evidence that $\theta$ is stable outside this task, these models, or these window sizes.

### 4.3 Intact-task ablation

Table 3 reports the ablation. Here, the task/reference block is never truncated. Coordination tokens are added around the intact task block, increasing total prompt length. Accuracy remains at ceiling across all tested conditions.

\begin{table}[t]
\centering
\small
\caption{Intact-task ablation. The full task/reference block is present in every condition. Values are pooled over both prompt orders, $N=10$ calls and $80$ binary field decisions per ratio per model.}
\begin{tabular}{lrrrrr}
\toprule
Model & 0\% & 50\% & 75\% & 90\% & 95\% \\
\midrule
GPT-4.1-mini & 1.000 & 1.000 & 1.000 & 1.000 & 1.000 \\
Claude Haiku 4.5 & 1.000 & 1.000 & 1.000 & 1.000 & 1.000 \\
Gemini 2.5 Flash & 1.000 & 1.000 & 1.000 & 1.000 & 1.000 \\
\bottomrule
\end{tabular}
\end{table}

Each model-ratio cell is $80/80$ correct, giving a Wilson 95\% interval of approximately $[0.954,1.000]$ per cell; pooled across all model-ratio cells, the result is $1200/1200$ field decisions with interval $[0.9968,1.000]$. This result directly constrains the revised interpretation. The main fixed-budget cliff does not, by itself, show that coordination text harms reasoning while task evidence remains intact. In this task and scoring setup, we detect no large semantic-interference effect even with large coordination blocks, as long as the reference remains present. The ablation does not rule out small effects, for example a 3--5 percentage-point degradation that this sample size would have low power to detect. Because it also changes the task from open-ended structured analysis to exact-field extraction, it is easier than the main task; it should therefore be interpreted as ruling out a large interference effect in this extraction-style intact-evidence setting, not as a full semantic-interference test for the original task format. The main effect is therefore best interpreted as displacement/truncation of task evidence under fixed budget.

### 4.4 Boundary tasks and external packs

The self-contained algorithmic task remains stable across overhead levels for all three providers. This shows that extra text alone is not sufficient to cause degradation in the tested range.

The context-distraction task shows a different and weaker phenomenon: Haiku degrades when coordination content contains salient claims conflicting with the target prior, while GPT and Gemini remain near baseline. Because this effect is model-specific and order-sensitive, we treat it as exploratory evidence of semantic distraction, not as a general law.

The external pack probes are consistent with the residual-budget account: performance remains near baseline while the full task pack fits, then falls when task content is partially or fully truncated.

\begin{table}[t]
\centering
\small
\caption{Task-dependent residual budget estimates at $W=4096$. Pack probes are RCWT stress probes, not leaderboard estimates.}
\begin{tabular}{lrrl}
\toprule
Probe & Fitted $p_0$ range & Approx. reserve & Truncation onset \\
\midrule
Task 1 spec recall & 0.913--0.939 & 250--356 tok & main task \\
GSM8K-pack & 0.853--0.893 & 438--602 tok & 90\% \\
MMLU-Pro-pack & 0.733--0.755 & 1003--1094 tok & 50\% partial \\
DROP-pack & 0.481--0.624 & 1539--2126 tok & 50\% \\
\bottomrule
\end{tabular}
\end{table}

## 5. Discussion

### 5.1 Scope of the contribution

The controlled allocation protocol is useful. RCWT isolates one factor that appears in real multi-agent systems: coordination content can consume the same prompt budget needed for task evidence. The main data support a practical engineering rule: measure the residual task budget needed by a task family and budget coordination around it.

### 5.2 Limitations of the stronger interpretation

The stronger title-level reading---that the paper proves broad reasoning degradation under context competition---is not supported. The main task is recall-heavy; the cliff aligns with small residual task blocks; the intact-task ablation stays at ceiling. The supported claim is therefore narrower and stronger: RCWT measures fixed-budget task displacement, and semantic interference requires separate evidence. The ablation limits any semantic effect in this extraction-style setup to below a large cliff-sized degradation; it should not be read as proof that semantic interference is zero in the original open-ended task format.

### 5.3 Cost versus benefit

RCWT measures cost, not net value. Real coordination can improve quality through decomposition, verification, tool use, or memory. A coordination block can be overhead for one agent and the task input for another. The correct practical question is not whether coordination hurts, but whether this coordination content provides enough benefit to justify the task-budget it consumes. RCWT supplies only the cost-side measurement.

### 5.4 Coordination heterogeneity

The central coordination block is synthetic and structured. Real coordination varies: dense tool outputs, verbose transcripts, retrieved documents, distilled state, uncertainty annotations, or contradictory agent claims may behave differently at the same token count. The current results should not be generalized across all coordination types without a design that varies content type independently of length.

## 6. Threats to Validity

**Exploratory design.** The fixed-budget sweep was developed iteratively. Cliff cells were added after early flat-region observations. Curve fits are descriptive.

**Task coverage.** The main result is strongest for one technical-specification recall task. Benchmark packs broaden coverage but do not constitute a pre-registered task-complexity ladder.

**Judge calibration.** The main task uses an LLM judge for open-ended parsing. Cross-vendor rescoring did not remove the qualitative cliff, but judge-specific strictness remains a limitation. The intact-task ablation uses deterministic JSON scoring to reduce this risk, but it also changes response format, task demand, and scoring method. Therefore, the ablation is best interpreted as a test for a large semantic-interference effect in an extraction-style intact-evidence setting, not as a token-identical rescore of the open-ended main task. A deterministic rescore or human audit of the main open-ended responses remains complementary validation.

**Tokenization.** Provider tokenizers differ. The scripts use `cl100k_base` for construction plus provider-reported token counts where available. Native token accounting should be preferred in future runs.

**Model availability.** Gemini 2.0 Flash was available during the original fixed-budget runs but was unavailable during later reruns. Historical aggregate files are retained; new reruns and the intact-task ablation use current model IDs.

**Single-call scope.** RCWT does not model session-level multi-agent dynamics, retrieval policy, tool failure, memory summarization, or coordination benefit.

## 7. Conclusion

RCWT is a controlled protocol for measuring task-budget displacement from coordination content in LLM calls. The main fixed-budget experiment shows a sharp high-overhead cliff on a context-dependent recall task, and window summaries are consistent with a task-specific residual-budget interpretation. The intact-task ablation shows that when the full task block remains present, we detect no large degradation from extra coordination content in an extraction-style setup, while small semantic effects and harder-task interference remain possible. This narrows the contribution: coordination context is not cost-free under fixed budgets, but the main evidence supports displacement of task evidence rather than a general semantic competition law. Practical systems should track residual task budget, not only nominal context size or total prompt length.

## Acknowledgements

LLM tools were used for grammar correction and drafting assistance. The authors take responsibility for the experiments, analysis, and claims.

## Appendix A. Reproduction commands

Run the intact-task ablation:

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

Main submitted result files:

- `results/rcwt_controlled_aggregates.json`
- `results/rcwt_curve_fits.json`
- `results/cross_benchmark_pack_summary_with_drop.json`
- `results/intact_ablation/rcwt_intact_ablation_aggregates.json`

## References

\begin{thebibliography}{99}

\bibitem{ram-etal-2023-context}
Ram, O., Levine, Y., Dalmedigos, I., Muhlgay, D., Shashua, A., Leyton-Brown, K., Shoham, Y.: In-Context Retrieval-Augmented Language Models. Transactions of the Association for Computational Linguistics 11, 1316--1331 (2023).

\bibitem{toolformer}
Schick, T., Dwivedi-Yu, J., Dessi, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., Scialom, T.: Toolformer: Language Models Can Teach Themselves to Use Tools. In: NeurIPS (2023).

\bibitem{autogen}
Wu, Q., Bansal, G., Zhang, J., Wu, Y., Li, B., Zhu, E., Jiang, L., Zhang, X., Zhang, S., Liu, J., Awadallah, A.H., White, R.W., Burger, D., Wang, C.: AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversations. In: First Conference on Language Modeling (2024).

\bibitem{guo2024multiagents}
Guo, T., Chen, X., Wang, Y., Chang, R., Pei, S., Chawla, N.V., Wiest, O., Zhang, X.: Large Language Model based Multi-Agents: A Survey of Progress and Challenges. arXiv:2402.01680 (2024).

\bibitem{lostmiddle}
Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F., Liang, P.: Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics 12, 157--173 (2024).

\bibitem{longbench}
Bai, Y., Lv, X., Zhang, J., Lyu, H., Tang, J., Huang, Z., Du, Z., Liu, X., Zeng, A., Hou, L., Dong, Y., Tang, J., Li, J.: LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. In: ACL (2024).

\bibitem{ruler}
Hsieh, C.P., Sun, S., Kriman, S., Acharya, S., Rekesh, D., Jia, F., Zhang, Y., Ginsburg, B.: RULER: What's the Real Context Size of Your Long-Context Language Models? In: First Conference on Language Modeling (2024).

\bibitem{du2025contextlength}
Du, Y., Tian, M., Ronanki, S., Rongali, S., Bodapati, S.B., Galstyan, A., Wells, A., Schwartz, R., Huerta, E.A., Peng, H.: Context Length Alone Hurts LLM Performance Despite Perfect Retrieval. In: Findings of ACL: EMNLP (2025).

\bibitem{longlora}
Chen, Y., Qian, S., Tang, H., Lai, X., Liu, Z., Han, S., Jia, J.: LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models. In: ICLR (2024).

\bibitem{flashattention}
Dao, T., Fu, D.Y., Ermon, S., Rudra, A., Ré, C.: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. In: NeurIPS (2022).

\bibitem{longformer}
Beltagy, I., Peters, M.E., Cohan, A.: Longformer: The Long-Document Transformer. arXiv:2004.05150 (2020).

\bibitem{bigbird}
Zaheer, M., Guruganesh, G., Dubey, A., Ainslie, J., Alberti, C., Ontanon, S., Pham, P., Ravula, A., Wang, Q., Yang, L., Ahmed, A.: Big Bird: Transformers for Longer Sequences. In: NeurIPS (2020).

\bibitem{jiang-etal-2023-llmlingua}
Jiang, H., Wu, Q., Lin, C.Y., Yang, Y., Qiu, L.: LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models. In: EMNLP, pp. 13358--13376 (2023).

\bibitem{llmlingua2}
Pan, Z., Wu, Q., Jiang, H., Xia, M., Luo, X., Zhang, J., Lin, Q., Rühle, V., Yang, Y., Lin, C.Y., Zhao, H.V., Qiu, L., Zhang, D.: LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression. In: Findings of ACL (2024).

\bibitem{mast}
Cemri, M., Pan, M.Z., Yang, S., Agrawal, L.A., Chopra, B., Tiwari, R., Keutzer, K., Parameswaran, A., Klein, D., Ramchandran, K., Zaharia, M., Gonzalez, J.E., Stoica, I.: Why Do Multi-Agent LLM Systems Fail? arXiv:2503.13657 (2025).

\bibitem{scalingagents}
Kim, Y., Gu, K., Park, C., Park, C., Schmidgall, S., Heydari, A.A., Yan, Y., Zhang, Z., Zhuang, Y., Liu, Y., Malhotra, M., Liang, P.P., Park, H.W., Yang, Y., Xu, X., Du, Y., Patel, S., Althoff, T., McDuff, D., Liu, X.: Towards a Science of Scaling Agent Systems. arXiv:2512.08296 (2025).

\bibitem{gsm8k}
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., Schulman, J.: Training Verifiers to Solve Math Word Problems. arXiv:2110.14168 (2021).

\bibitem{mmlupro}
Wang, Y., Ma, X., Zhang, G., Ni, Y., Chandra, A., Guo, S., Ren, W., Arulraj, A., He, X., Jiang, Z., Li, T., Ku, M., Wang, K., Zhuang, A., Fan, R., Yue, X., Chen, W.: MMLU-Pro: A More Robust and Challenging Multi-Task Language Understanding Benchmark. In: NeurIPS Datasets and Benchmarks Track (2024).

\bibitem{drop}
Dua, D., Wang, Y., Dasigi, P., Stanovsky, G., Singh, S., Gardner, M.: DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs. In: NAACL-HLT, pp. 2368--2378 (2019).

\bibitem{wilson1927probable}
Wilson, E.B.: Probable Inference, the Law of Succession, and Statistical Inference. Journal of the American Statistical Association 22(158), 209--212 (1927).

\end{thebibliography}
