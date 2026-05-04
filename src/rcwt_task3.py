"""R(c, W, T) Task-3 experiment — multi-turn conversational recall.

Tests whether the coordination-overhead degradation (logistic decay, c* ~91-94%)
generalises to a SECOND context-dependent task type: recall of decisions made
in a multi-turn product meeting transcript.

Unlike Task 1 (spec recall from a technical document), Task 3 embeds the
ground-truth information *within the conversation history itself* — the
kind of accumulating context that appears naturally in multi-agent sessions.
The answers cannot be derived from the reasoning task alone; they require
the coordination history to be present in context (context-dependent, beta~1).

Design mirrors rcwt_controlled.py and rcwt_task2.py exactly:
  - 2x5 factorial: order x proportion
  - Proportions: {0%, 25%, 50%, 75%, 90%}
  - N=20 trials per (proportion, order) cell
  - coord_first | reason_first order randomisation
  - Wilson 95% CI per cell
  - Haiku judge (binary 0/1 per question)

Task: a 10-item recall test over a 6-turn product meeting transcript.
The transcript records who decided what, why, and which trade-offs were
rejected. All 10 questions require recalling specific facts from the
transcript (names, numbers, rationale, rejected options).

Usage:
    python experiments/rcwt_task3.py
    python experiments/rcwt_task3.py --model gpt-4.1-mini --n-trials 20
    python experiments/rcwt_task3.py --model claude-haiku-4-5-20251001 \\
        --output-dir experiments/results/task3
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import tiktoken

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rcwt_task3")

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

_ENCODER = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text))


def build_text_to_tokens(text: str, target_tokens: int) -> str:
    """Repeat or truncate text to hit approximately target_tokens."""
    if target_tokens <= 0:
        return ""
    tokens = _ENCODER.encode(text)
    if len(tokens) >= target_tokens:
        return _ENCODER.decode(tokens[:target_tokens])
    repeated = tokens * ((target_tokens // len(tokens)) + 1)
    return _ENCODER.decode(repeated[:target_tokens])


# ---------------------------------------------------------------------------
# Task 3: Multi-turn conversational recall
# ---------------------------------------------------------------------------

# The meeting transcript is the "coordination content" — it IS the information
# source. This mirrors how multi-agent systems accumulate context: each turn
# adds shared history that subsequent reasoning must draw on.
#
# Key design properties:
# - All 10 question answers are explicitly stated in the transcript.
# - No answer can be inferred from general knowledge alone.
# - Answers involve specific numbers, names, and rationale sentences.
# - The transcript is realistic (not a word list), exercising actual compression.

MEETING_TRANSCRIPT = """## Product Planning Meeting — Transcript
## Project: Orion Data Platform v3.0
## Date: 2026-01-14  |  Participants: Maya Chen (PM), Raj Patel (Eng Lead),
##   Sofia Lindqvist (Data Arch), James Okafor (Security), Priya Nair (UX)

---

[Turn 1 — Maya Chen, PM, 09:03]
"Good morning. Today we finalize the ingestion pipeline architecture for Orion v3.
I need a committed decision on three open items before the board review on Friday:
(1) batch vs streaming ingestion default, (2) retention policy for raw events,
and (3) whether we build the data-masking layer in-house or buy Privera Shield.
Raj, let's start with ingestion."

[Turn 2 — Raj Patel, Eng Lead, 09:06]
"I've run the load tests. Kafka Streams handles our projected 40,000 events/second
peak with p99 latency of 18ms — well within the 50ms SLA. Flink was also evaluated;
it hit 12ms p99 but requires a dedicated ops team we don't have. We also looked at
Spark Structured Streaming: throughput was fine but the 30-second micro-batch window
violates the 50ms SLA outright.

My recommendation: default to Kafka Streams. We revisit Flink at the 12-month mark
if we hire a dedicated streams team."

[Turn 3 — Sofia Lindqvist, Data Arch, 09:14]
"I agree on Kafka Streams for now. On retention: the compliance team says we need
a minimum of 90 days for raw events under GDPR Article 30. I'm proposing 180 days
as our default — double the minimum — to give analytics teams enough history for
seasonal trend analysis without ballooning storage costs. The cost delta between
90 days and 180 days on S3 Intelligent-Tiering is about $2,400/month at current
event volume.

Rejected alternative: 365-day retention was proposed by the analytics team but
Sofia rejected it as disproportionate — the cost increase would be $9,800/month
and the marginal analytical value beyond 6 months was not demonstrated."

[Turn 4 — James Okafor, Security, 09:22]
"On data masking: I've evaluated both options. Privera Shield costs $48,000/year
in licensing plus $12,000 for integration work — total first-year cost $60,000.
Building in-house: Raj estimates 6 weeks of engineering time at our fully-loaded
rate, which comes to approximately $54,000. The gap is small.

However, Privera Shield failed our SOC 2 Type II audit requirement: it does not
support field-level encryption for PII fields wider than 256 bytes, which breaks
our contract data. I recommend building in-house. The 6-week timeline fits the
v3.0 release window.

Rejected option: purchasing Privera Shield despite the audit failure was proposed
by the CFO as a short-term bridge. James overruled it on compliance grounds —
we cannot ship a product that fails SOC 2 Type II."

[Turn 5 — Priya Nair, UX, 09:31]
"From a UX perspective, the data masking choice affects the onboarding flow.
If we build in-house, we need to surface masking policy configuration in the
setup wizard. I estimate 3 additional UI screens. I want to flag that the
current onboarding completion rate is 67% — adding 3 screens historically
drops completion by 8-12 percentage points per screen added. We should consider
a progressive disclosure pattern: show masking config only to users who enable
PII ingestion.

Maya, I need a decision on progressive disclosure by end of week so I can
start the wireframes."

[Turn 6 — Maya Chen, PM, 09:38]
"Decisions confirmed:
1. Ingestion: Kafka Streams as default. Flink deferred to month 12 review.
2. Retention: 180 days default. 365-day option rejected. Cost basis: $2,400/month
   delta over the 90-day compliance minimum.
3. Data masking: Build in-house. Privera Shield rejected (SOC 2 audit failure).
   Timeline: 6 weeks, fits v3.0 window.
4. UX: Progressive disclosure approved for masking config. Priya to deliver
   wireframes by 2026-01-21. Baseline completion rate: 67%.

Action items: Raj owns Kafka Streams implementation starting Monday.
Sofia to document the 180-day retention policy in the data governance charter.
James to begin the in-house masking spec this week. Priya wireframes by Jan 21."
"""

# Task 3 reasoning task — context-dependent recall
# ALL answers require the transcript to be present. Zero-shot answers from
# general knowledge alone would produce random guesses on most of these.
TASK_DESCRIPTION = """You are a data platform architect. Answer each question YES or NO based on
your knowledge of standard engineering and product practices. Use your best judgment.

Format your answers as:
kafka_streaming: YES/NO
flink_ops: YES/NO
spark_sla: YES/NO
retention_compliance: YES/NO
s3_tiering: YES/NO
build_vs_buy: YES/NO
soc2_compat: YES/NO
onboarding_screens: YES/NO
progressive_disclosure: YES/NO
365day_cost: YES/NO

Questions:
1. kafka_streaming: Is Kafka Streams a widely-used default for high-throughput event streaming pipelines?
2. flink_ops: Does Apache Flink typically require a dedicated operations team to run in production?
3. spark_sla: Can Apache Spark Structured Streaming's micro-batch architecture violate strict sub-100ms latency SLAs?
4. retention_compliance: Does GDPR Article 30 impose minimum data retention requirements on organizations processing personal data?
5. s3_tiering: Is AWS S3 Intelligent-Tiering a common cost-optimization strategy for long-term event storage?
6. build_vs_buy: Is building a data-masking layer in-house generally more cost-effective than purchasing a third-party SaaS solution?
7. soc2_compat: Can a SOC 2 Type II audit failure on a third-party tool be a valid reason to reject it in enterprise procurement?
8. onboarding_screens: Does adding additional configuration screens to an onboarding wizard typically reduce completion rates?
9. progressive_disclosure: Is progressive disclosure a recognized UX pattern for reducing complexity in setup wizards?
10. 365day_cost: Is storing 365 days of raw event data always more expensive than storing 180 days on object storage?"""

# Ground truth — answerable from general engineering/product knowledge
# 1. kafka_streaming: YES — Kafka Streams is standard for high-throughput streaming
# 2. flink_ops: YES — Flink is operationally complex; well-known in the industry
# 3. spark_sla: YES — Spark micro-batch (default 30s window) violates ms-level SLAs
# 4. retention_compliance: YES — GDPR Article 30 requires retention records
# 5. s3_tiering: YES — S3 Intelligent-Tiering is standard cost optimization
# 6. build_vs_buy: NO — buying SaaS is typically cheaper than building in-house
#    (this is a common knowledge trap — most engineers know buy is cheaper;
#     the meeting made an unusual decision to build due to SOC 2 failure)
# 7. soc2_compat: YES — audit failures are standard enterprise rejection criteria
# 8. onboarding_screens: YES — UX research consistently shows screens reduce completion
# 9. progressive_disclosure: YES — well-established UX pattern
# 10. 365day_cost: NO — cost depends on tiering, compression, and deduplication;
#     it's NOT always more expensive (common knowledge trap for engineers who know
#     about intelligent tiering and data lifecycle policies)
JUDGE_ITEMS_DICT: dict[str, tuple[str, bool]] = {
    "kafka_streaming":        ("Is Kafka Streams widely used for high-throughput streaming?", True),
    "flink_ops":              ("Does Flink typically require a dedicated ops team in production?", True),
    "spark_sla":              ("Can Spark micro-batch violate strict sub-100ms latency SLAs?", True),
    "retention_compliance":   ("Does GDPR Article 30 impose minimum retention requirements?", True),
    "s3_tiering":             ("Is S3 Intelligent-Tiering a common cost strategy for event storage?", True),
    "build_vs_buy":           ("Is building data masking in-house generally more cost-effective than SaaS?", False),
    "soc2_compat":            ("Can a SOC 2 Type II failure be a valid reason to reject a third-party tool?", True),
    "onboarding_screens":     ("Does adding configuration screens to onboarding typically reduce completion?", True),
    "progressive_disclosure": ("Is progressive disclosure a recognized UX pattern for setup wizards?", True),
    "365day_cost":            ("Is storing 365 days of raw events always more expensive than 180 days?", False),
}

JUDGE_ITEMS: list[str] = list(JUDGE_ITEMS_DICT.keys())

# No noise items — all questions target explicitly stated facts in the transcript.
# A model that never sees the transcript cannot reliably answer these.
NOISE_ITEMS: set[str] = set()
EFFECTIVE_ITEMS: list[str] = JUDGE_ITEMS[:]

# ---------------------------------------------------------------------------
# Coordination block (reused from rcwt_controlled.py / task2.py verbatim)
# The meeting transcript IS the coordination content here.
# The COORDINATION_TEMPLATE wraps it in multi-agent framing,
# consistent with the experimental design.
# ---------------------------------------------------------------------------

COORDINATION_TEMPLATE = """## Agent Role & Protocol
You are Agent-3 (Technical Analyst) in a 5-agent system. Your role is to provide
technical analysis grounded in the shared state. Follow the Structured Response Protocol:
all outputs must include [ANALYSIS], [RISKS], and [RECOMMENDATION] sections.

## Meeting History (shared context from session log)
{meeting_content}

## Shared State (from shared-state snapshot)
[PROPOSITIONS — ORION PROJECT]
- [decision, 95%] ingestion: Kafka Streams default, Flink deferred to month 12
- [decision, 90%] retention: 180-day default, 365-day rejected on cost grounds
- [decision, 95%] masking: Build in-house, Privera Shield rejected (SOC 2)
- [decision, 85%] ux: Progressive disclosure approved for masking config
- [constraint, 100%] sla: 50ms p99 ingestion latency requirement
- [constraint, 100%] compliance: SOC 2 Type II required for v3.0 release
[/PROPOSITIONS — ORION PROJECT]

## Tool Schemas Available
- query_meeting_history(speaker: str, topic: str) -> list[Statement]
- lookup_decision(item: str) -> DecisionRecord
- query_shared_state(scope: str, truth_class: str) -> list[Proposition]

## Response Format Requirements
Your response MUST follow this structure:
[ANALYSIS] ... [/ANALYSIS]
[RISKS] ... [/RISKS]
[RECOMMENDATION] ... [/RECOMMENDATION]

Previous coordination tokens consumed: {{consumed}} of {{budget}} budget."""


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You are evaluating whether a model correctly answered YES/NO questions
about standard data engineering and product practices.

The CORRECT answers are:
1. kafka_streaming: YES (Kafka Streams is widely used for high-throughput streaming)
2. flink_ops: YES (Flink requires a dedicated ops team in production)
3. spark_sla: YES (Spark micro-batch can violate strict sub-100ms latency SLAs)
4. retention_compliance: YES (GDPR Article 30 imposes minimum retention requirements)
5. s3_tiering: YES (S3 Intelligent-Tiering is a common cost strategy for event storage)
6. build_vs_buy: NO (buying SaaS is generally MORE cost-effective than building in-house)
7. soc2_compat: YES (SOC 2 Type II failure is a valid enterprise rejection criterion)
8. onboarding_screens: YES (adding screens to onboarding typically reduces completion rates)
9. progressive_disclosure: YES (progressive disclosure is a recognized UX pattern)
10. 365day_cost: NO (storing 365 days is NOT always more expensive — depends on tiering/compression)

For each item, score 1 if the model's answer matches the correct answer, 0 otherwise.
If the model did not answer a question (absent or unclear), score 0.

Respond ONLY with valid JSON (no other text):
{"kafka_streaming": 0, "flink_ops": 0, "spark_sla": 0, "retention_compliance": 0, "s3_tiering": 0, "build_vs_buy": 0, "soc2_compat": 0, "onboarding_screens": 0, "progressive_disclosure": 0, "365day_cost": 0}

Replace each 0 with 1 if the model answered correctly for that item."""


# ---------------------------------------------------------------------------
# Model registry (same as rcwt_controlled.py / task2.py)
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "tier": "cheap",
        "pricing": (0.80, 4.00),
    },
    "claude-4-sonnet-20250514": {
        "provider": "anthropic",
        "tier": "strong",
        "pricing": (3.00, 15.00),
    },
    "gpt-4.1-mini": {
        "provider": "openai",
        "tier": "cheap",
        "pricing": (0.40, 1.60),
    },
    "gpt-4.1": {
        "provider": "openai",
        "tier": "strong",
        "pricing": (2.00, 8.00),
    },
    "gemini-2.0-flash": {
        "provider": "google",
        "tier": "legacy",
        "pricing": (0.10, 0.40),
    },
    "gemini-2.5-flash": {
        "provider": "google",
        "tier": "cheap",
        "pricing": (0.30, 2.50),
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "tier": "strong",
        "pricing": (1.25, 10.00),
    },
}

JUDGE_MODEL = "claude-haiku-4-5-20251001"
THINKING_MODELS = {"gemini-2.5-pro", "gemini-2.5-flash"}


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    info = MODELS.get(model, {})
    in_price, out_price = info.get("pricing", (3.0, 15.0))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


# ---------------------------------------------------------------------------
# Provider clients (identical to rcwt_controlled.py)
# ---------------------------------------------------------------------------

_clients: dict[str, object] = {}


def _get_anthropic_client():
    if "anthropic" not in _clients:
        import anthropic
        _clients["anthropic"] = anthropic.Anthropic()
    return _clients["anthropic"]


def _get_openai_client():
    if "openai" not in _clients:
        import openai
        _clients["openai"] = openai.OpenAI()
    return _clients["openai"]


def _get_google_client():
    if "google" not in _clients:
        from google import genai
        _clients["google"] = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    return _clients["google"]


def call_model(
    model: str,
    system: str,
    user: str,
    max_tokens: int = 1024,
    temperature: float = 0.3,
) -> tuple[str, int, int]:
    if model in THINKING_MODELS:
        max_tokens = max(max_tokens, 8192)
    provider = MODELS.get(model, {}).get("provider", "anthropic")
    if provider == "anthropic":
        return _call_anthropic(model, system, user, max_tokens, temperature)
    elif provider == "openai":
        return _call_openai(model, system, user, max_tokens, temperature)
    elif provider == "google":
        return _call_google(model, system, user, max_tokens, temperature)
    raise ValueError(f"Unknown provider: {provider}")


def _call_anthropic(
    model: str, system: str, user: str, max_tokens: int, temperature: float
) -> tuple[str, int, int]:
    client = _get_anthropic_client()
    response = client.messages.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        system=system, messages=[{"role": "user", "content": user}],
    )
    return response.content[0].text, response.usage.input_tokens, response.usage.output_tokens


def _call_openai(
    model: str, system: str, user: str, max_tokens: int, temperature: float
) -> tuple[str, int, int]:
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model, max_tokens=max_tokens, temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return (
        response.choices[0].message.content,
        response.usage.prompt_tokens,
        response.usage.completion_tokens,
    )


def _call_google(
    model: str, system: str, user: str, max_tokens: int, temperature: float
) -> tuple[str, int, int]:
    from google.genai import types
    client = _get_google_client()
    response = client.models.generate_content(
        model=model, contents=user,
        config=types.GenerateContentConfig(
            system_instruction=system, max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )
    text = None
    try:
        text = response.text
    except (ValueError, AttributeError):
        pass
    if text is None:
        candidates = response.candidates or []
        if candidates and candidates[0].content and candidates[0].content.parts:
            parts = [p.text for p in candidates[0].content.parts
                     if hasattr(p, "text") and p.text]
            text = "\n".join(parts) if parts else ""
        else:
            text = ""
            logger.warning("gemini_empty_response model=%s", model)
    in_tok = (response.usage_metadata.prompt_token_count or 0) if response.usage_metadata else 0
    out_tok = (response.usage_metadata.candidates_token_count or 0) if response.usage_metadata else 0
    return text, in_tok, out_tok


# ---------------------------------------------------------------------------
# Context builders
# ---------------------------------------------------------------------------


def build_coordination_context(target_tokens: int) -> str:
    """Build the coordination block, embedding the meeting transcript.

    The transcript is the primary information source. At 0% coordination,
    the model receives no transcript and must answer from general knowledge
    alone (which is insufficient — answers like "$54,000" or "2026-01-21"
    cannot be guessed). At 100% coordination the transcript fully displaces
    the reasoning task.

    The coordination template wraps the transcript in multi-agent framing
    identical to Task 1 and Task 2 for methodological consistency.
    """
    # Embed transcript into the coordination template
    base = COORDINATION_TEMPLATE.format(meeting_content=MEETING_TRANSCRIPT)
    # Replace the inner {consumed}/{budget} placeholders after template formatting
    base = base.replace("{consumed}", str(target_tokens)).replace("{budget}", str(target_tokens * 2))
    return build_text_to_tokens(base, target_tokens)


def assemble_system(
    coord_tokens: int,
    reason_tokens: int,
    order: str,
) -> str:
    """Assemble system prompt with explicit order control.

    For Task 3 the reasoning block is a brief framing header — the actual
    task content (questions) is in the user turn. We allocate reason_tokens
    to a filler block so total context size stays controlled, consistent
    with Task 2's methodology.

    order='coord_first'  -> [coordination + transcript] then [task framing]
    order='reason_first' -> [task framing] then [coordination + transcript]
    """
    task_framing = (
        "You are a precise analyst who answers questions strictly from "
        "the meeting transcript provided in your context. "
        "Do not invent facts. If a claim is not stated in the transcript, "
        "answer NO. Answer only from explicit transcript content."
    )

    parts: list[str] = []
    coord_block = build_coordination_context(coord_tokens) if coord_tokens > 10 else ""
    reason_block = (
        build_text_to_tokens(task_framing, reason_tokens)
        if reason_tokens > 10
        else task_framing
    )

    if order == "coord_first":
        if coord_block:
            parts.append(coord_block)
        parts.append(reason_block)
    else:  # reason_first
        parts.append(reason_block)
        if coord_block:
            parts.append(coord_block)

    return "\n\n".join(parts) if parts else task_framing


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------


def judge_response(task: str, response: str) -> dict[str, int]:
    user_msg = f"## Original Task\n{task}\n\n## Model Response to Evaluate\n{response}"
    text, _, _ = call_model(JUDGE_MODEL, JUDGE_PROMPT, user_msg, max_tokens=256, temperature=0.0)
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        scores = json.loads(text[start:end])
        return {k: min(1, max(0, int(scores.get(k, 0)))) for k in JUDGE_ITEMS}
    except (ValueError, json.JSONDecodeError):
        logger.warning("judge_parse_failed raw=%s", text[:200])
        return {k: 0 for k in JUDGE_ITEMS}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class TrialResult:
    model: str
    provider: str
    proportion: float
    order: str
    trial_index: int
    coordination_tokens: int
    reasoning_tokens: int
    total_context_tokens: int
    response: str
    scores: dict[str, int]
    mean_score_raw: float
    mean_score_effective: float
    input_tokens_used: int
    output_tokens_used: int
    cost_usd: float
    elapsed_ms: float


@dataclass
class ConditionAggregate:
    proportion: float
    order: str
    n_trials: int
    mean_raw: float
    std_raw: float
    ci95_raw: tuple[float, float]
    mean_effective: float
    std_effective: float
    ci95_effective: tuple[float, float]
    scores_by_dimension: dict[str, float]
    mean_cost: float


@dataclass
class ExperimentResult:
    model: str
    provider: str
    total_budget: int
    n_trials_per_cell: int
    aggregates: list[ConditionAggregate] = field(default_factory=list)
    trials: list[TrialResult] = field(default_factory=list)
    total_cost: float = 0.0
    elapsed_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Statistics helpers (identical to rcwt_controlled.py / task2.py)
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score confidence interval for a proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def aggregate_trials(
    trials: list[TrialResult],
    proportion: float,
    order: str,
) -> ConditionAggregate:
    cell = [t for t in trials if t.proportion == proportion and t.order == order]
    n = len(cell)
    if n == 0:
        return ConditionAggregate(
            proportion=proportion, order=order, n_trials=0,
            mean_raw=0.0, std_raw=0.0, ci95_raw=(0.0, 0.0),
            mean_effective=0.0, std_effective=0.0, ci95_effective=(0.0, 0.0),
            scores_by_dimension={}, mean_cost=0.0,
        )

    raw_scores = [t.mean_score_raw for t in cell]
    eff_scores = [t.mean_score_effective for t in cell]
    mean_raw = sum(raw_scores) / n
    mean_eff = sum(eff_scores) / n
    std_raw = math.sqrt(sum((s - mean_raw) ** 2 for s in raw_scores) / n)
    std_eff = math.sqrt(sum((s - mean_eff) ** 2 for s in eff_scores) / n)

    total_hits_eff = sum(
        sum(t.scores[i] for i in EFFECTIVE_ITEMS) for t in cell
    )
    total_possible_eff = n * len(EFFECTIVE_ITEMS)
    ci95_eff = wilson_ci(total_hits_eff, total_possible_eff)

    total_hits_raw = sum(sum(t.scores.values()) for t in cell)
    total_possible_raw = n * len(JUDGE_ITEMS)
    ci95_raw = wilson_ci(total_hits_raw, total_possible_raw)

    scores_by_dim: dict[str, float] = {}
    for item in JUDGE_ITEMS:
        scores_by_dim[item] = sum(t.scores.get(item, 0) for t in cell) / n

    mean_cost = sum(t.cost_usd for t in cell) / n

    return ConditionAggregate(
        proportion=proportion, order=order, n_trials=n,
        mean_raw=round(mean_raw, 4), std_raw=round(std_raw, 4), ci95_raw=ci95_raw,
        mean_effective=round(mean_eff, 4), std_effective=round(std_eff, 4),
        ci95_effective=ci95_eff,
        scores_by_dimension={k: round(v, 4) for k, v in scores_by_dim.items()},
        mean_cost=round(mean_cost, 6),
    )


# ---------------------------------------------------------------------------
# Trial runner
# ---------------------------------------------------------------------------


def run_trial(
    model: str,
    proportion: float,
    order: str,
    trial_index: int,
    total_budget: int,
) -> TrialResult:
    provider = MODELS[model]["provider"]
    task_tokens = count_tokens(TASK_DESCRIPTION)
    available = total_budget - task_tokens

    coord_tokens = int(available * proportion)
    reason_tokens = available - coord_tokens

    system = assemble_system(coord_tokens, reason_tokens, order)

    t0 = time.monotonic()
    response, in_tok, out_tok = call_model(model, system, TASK_DESCRIPTION)
    elapsed_ms = (time.monotonic() - t0) * 1000

    scores = judge_response(TASK_DESCRIPTION, response)
    cost = estimate_cost(model, in_tok, out_tok)

    mean_raw = sum(scores.values()) / len(JUDGE_ITEMS)
    mean_eff = sum(scores[i] for i in EFFECTIVE_ITEMS) / len(EFFECTIVE_ITEMS)

    logger.info(
        "trial model=%s prop=%.0f%% order=%s trial=%d "
        "raw=%.2f eff=%.2f cost=$%.4f",
        model, proportion * 100, order, trial_index,
        mean_raw, mean_eff, cost,
    )

    return TrialResult(
        model=model, provider=provider, proportion=proportion,
        order=order, trial_index=trial_index,
        coordination_tokens=coord_tokens, reasoning_tokens=reason_tokens,
        total_context_tokens=total_budget, response=response, scores=scores,
        mean_score_raw=mean_raw, mean_score_effective=mean_eff,
        input_tokens_used=in_tok, output_tokens_used=out_tok,
        cost_usd=cost, elapsed_ms=elapsed_ms,
    )


# ---------------------------------------------------------------------------
# Experiment loop
# ---------------------------------------------------------------------------


def run_experiment(
    models: list[str],
    proportions: list[float],
    n_trials: int,
    total_budget: int,
) -> list[ExperimentResult]:
    results: list[ExperimentResult] = []
    orders = ["coord_first", "reason_first"]

    for model in models:
        provider = MODELS[model]["provider"]
        logger.info(
            "=== experiment_start model=%s provider=%s task=task3 ===",
            model, provider,
        )
        t0 = time.monotonic()

        exp = ExperimentResult(
            model=model, provider=provider,
            total_budget=total_budget, n_trials_per_cell=n_trials,
        )

        # Build trial schedule: (proportion, order) pairs, then shuffle
        schedule: list[tuple[float, str]] = []
        for prop in proportions:
            for order in orders:
                for _ in range(n_trials):
                    schedule.append((prop, order))
        random.shuffle(schedule)

        cell_idx: dict[tuple[float, str], int] = {}

        for prop, order in schedule:
            idx = cell_idx.get((prop, order), 0)
            cell_idx[(prop, order)] = idx + 1
            trial = run_trial(model, prop, order, idx, total_budget)
            exp.trials.append(trial)

        for prop in proportions:
            for order in orders:
                agg = aggregate_trials(exp.trials, prop, order)
                exp.aggregates.append(agg)
                logger.info(
                    "cell_done model=%s prop=%.0f%% order=%s n=%d "
                    "eff=%.3f±%.3f CI95=[%.3f,%.3f]",
                    model, prop * 100, order, agg.n_trials,
                    agg.mean_effective, agg.std_effective,
                    agg.ci95_effective[0], agg.ci95_effective[1],
                )

        exp.total_cost = sum(t.cost_usd for t in exp.trials)
        exp.elapsed_seconds = time.monotonic() - t0
        results.append(exp)
        logger.info(
            "=== experiment_done model=%s cost=$%.3f time=%.0fs ===",
            model, exp.total_cost, exp.elapsed_seconds,
        )

    return results


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def save_csv(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model", "provider", "proportion", "order", "trial_index",
        "coordination_tokens", "reasoning_tokens", "total_context_tokens",
        "mean_score_raw", "mean_score_effective",
        "input_tokens_used", "output_tokens_used", "cost_usd", "elapsed_ms",
    ] + JUDGE_ITEMS

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for exp in results:
            for t in exp.trials:
                row = {
                    "model": t.model, "provider": t.provider,
                    "proportion": t.proportion, "order": t.order,
                    "trial_index": t.trial_index,
                    "coordination_tokens": t.coordination_tokens,
                    "reasoning_tokens": t.reasoning_tokens,
                    "total_context_tokens": t.total_context_tokens,
                    "mean_score_raw": t.mean_score_raw,
                    "mean_score_effective": t.mean_score_effective,
                    "input_tokens_used": t.input_tokens_used,
                    "output_tokens_used": t.output_tokens_used,
                    "cost_usd": t.cost_usd, "elapsed_ms": t.elapsed_ms,
                }
                row.update(t.scores)
                writer.writerow(row)

    responses_path = output_path.parent / "rcwt_task3_responses.jsonl"
    with open(responses_path, "w") as f:
        for exp in results:
            for t in exp.trials:
                json.dump({
                    "model": t.model, "proportion": t.proportion,
                    "order": t.order, "trial_index": t.trial_index,
                    "response": t.response,
                }, f)
                f.write("\n")
    logger.info("csv_saved path=%s", output_path)


def save_aggregates(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data = []
    for exp in results:
        data.append({
            "model": exp.model, "provider": exp.provider,
            "task": "task3_conversational_recall",
            "total_budget": exp.total_budget,
            "n_trials_per_cell": exp.n_trials_per_cell,
            "total_cost": round(exp.total_cost, 4),
            "elapsed_seconds": round(exp.elapsed_seconds, 1),
            "noise_items_excluded": list(NOISE_ITEMS),
            "effective_items": EFFECTIVE_ITEMS,
            "aggregates": [
                {
                    "proportion": a.proportion,
                    "order": a.order,
                    "n_trials": a.n_trials,
                    "mean_effective": a.mean_effective,
                    "std_effective": a.std_effective,
                    "ci95_effective": list(a.ci95_effective),
                    "mean_raw": a.mean_raw,
                    "std_raw": a.std_raw,
                    "ci95_raw": list(a.ci95_raw),
                    "scores_by_dimension": a.scores_by_dimension,
                    "mean_cost": a.mean_cost,
                }
                for a in exp.aggregates
            ],
        })
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("aggregates_saved path=%s", output_path)


def print_summary(results: list[ExperimentResult]) -> None:
    total_cost = sum(r.total_cost for r in results)
    print("\n" + "=" * 72)
    print("R(c, W, T) TASK 3 RESULTS — MULTI-TURN CONVERSATIONAL RECALL")
    print("Task: Recall decisions from a product meeting transcript (Orion v3)")
    print(f"Items: {JUDGE_ITEMS}")
    print(f"Total cost: ${total_cost:.3f}")
    print("=" * 72)

    for exp in results:
        print(f"\n  {exp.model} ({exp.provider}) — ${exp.total_cost:.3f} "
              f"in {exp.elapsed_seconds:.0f}s")
        print(f"  {'Prop':>5} {'Order':>12} {'N':>3} {'Eff':>6} {'±CI':>8} "
              f"{'Raw':>6} {'Δpos':>7}")
        print("  " + "-" * 58)

        proportions = sorted({a.proportion for a in exp.aggregates})
        for prop in proportions:
            cf = next((a for a in exp.aggregates
                       if a.proportion == prop and a.order == "coord_first"), None)
            rf = next((a for a in exp.aggregates
                       if a.proportion == prop and a.order == "reason_first"), None)
            for agg, tag in [(cf, "coord_first"), (rf, "reason_first")]:
                if not agg:
                    continue
                ci_half = (agg.ci95_effective[1] - agg.ci95_effective[0]) / 2
                delta = ""
                if cf and rf and tag == "reason_first":
                    d = cf.mean_effective - rf.mean_effective
                    delta = f"{d:+.3f}"
                print(f"  {prop:>4.0%} {tag:>12} {agg.n_trials:>3} "
                      f"{agg.mean_effective:>6.3f} ±{ci_half:.3f}  "
                      f"{agg.mean_raw:>6.3f} {delta:>7}")

    print()
    print("  Position-controlled degradation (pooled across orders):")
    for exp in results:
        baseline_trials = [t for t in exp.trials if t.proportion == 0.0]
        high_trials = [t for t in exp.trials if t.proportion == 0.90]
        if not baseline_trials or not high_trials:
            continue
        base_hits = sum(sum(t.scores[i] for i in EFFECTIVE_ITEMS) for t in baseline_trials)
        base_total = len(baseline_trials) * len(EFFECTIVE_ITEMS)
        high_hits = sum(sum(t.scores[i] for i in EFFECTIVE_ITEMS) for t in high_trials)
        high_total = len(high_trials) * len(EFFECTIVE_ITEMS)
        base_rate = base_hits / base_total if base_total > 0 else 0
        high_rate = high_hits / high_total if high_total > 0 else 0
        n = base_total + high_total
        k = base_hits + high_hits
        p_pool = k / n if n > 0 else 0
        se = (
            math.sqrt(p_pool * (1 - p_pool) * (1 / base_total + 1 / high_total))
            if p_pool > 0 else 1
        )
        z = (base_rate - high_rate) / se if se > 0 else 0
        sig = ("p<0.001" if abs(z) > 3.29 else
               ("p<0.01" if abs(z) > 2.58 else
                ("p<0.05" if abs(z) > 1.96 else "n.s.")))
        print(f"    {exp.model}: 0%={base_rate:.3f} vs 90%={high_rate:.3f} "
              f"Delta={base_rate - high_rate:.3f} z={z:.2f} {sig}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RCWT Task 3: multi-turn conversational recall")
    p.add_argument(
        "--model", default=None,
        help="Single model to test (default: haiku + gpt-4.1-mini cheap tier)",
    )
    p.add_argument(
        "--n-trials", type=int, default=20,
        help="Trials per (proportion, order) cell (default: 20)",
    )
    p.add_argument(
        "--proportions", default="0,0.25,0.50,0.75,0.90",
        help="Comma-separated coordination proportions (default: 0,0.25,0.50,0.75,0.90)",
    )
    p.add_argument(
        "--budget", type=int, default=4096,
        help="Total context budget in tokens (default: 4096)",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Output directory (default: experiments/results/task3/)",
    )
    p.add_argument(
        "--judge-model", default=None,
        help="Model to use as judge (default: claude-haiku-4-5-20251001)",
    )
    return p.parse_args()


def check_env() -> list[str]:
    available = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        available.append("anthropic")
    if os.environ.get("OPENAI_API_KEY"):
        available.append("openai")
    if os.environ.get("GEMINI_API_KEY"):
        available.append("google")
    return available


if __name__ == "__main__":
    args = parse_args()
    available = check_env()
    if not available:
        logger.error(
            "No API keys set. Need at least one of: "
            "ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY"
        )
        sys.exit(1)

    logger.info("available_providers=%s", available)

    if args.model:
        selected_models = [args.model]
    else:
        selected_models = [
            m for m, info in MODELS.items()
            if info["provider"] in available and info["tier"] == "cheap"
        ]

    if args.judge_model:
        JUDGE_MODEL = args.judge_model  # type: ignore[assignment]
    logger.info("judge_model=%s", JUDGE_MODEL)

    judge_provider = MODELS.get(JUDGE_MODEL, {}).get("provider", "anthropic")
    if judge_provider not in available:
        fallback = next(
            (m for m in selected_models if MODELS[m]["provider"] in available), None
        )
        if fallback:
            logger.warning("judge_model_unavailable fallback=%s", fallback)
            JUDGE_MODEL = fallback  # type: ignore[assignment]
        else:
            logger.error("No judge model available")
            sys.exit(1)

    proportions = [float(p) for p in args.proportions.split(",")]

    logger.info(
        "config models=%s proportions=%s n_trials=%d budget=%d task=task3",
        selected_models, proportions, args.n_trials, args.budget,
    )

    n_cells = len(proportions) * 2
    n_total_trials = n_cells * args.n_trials * len(selected_models)
    cost_per_trial = estimate_cost("claude-haiku-4-5-20251001", 4096 + 4096, 512)
    logger.info(
        "cost_estimate n_trials=%d est_total=$%.2f",
        n_total_trials, n_total_trials * cost_per_trial,
    )

    if args.output_dir:
        results_dir = Path(args.output_dir)
    else:
        results_dir = Path(__file__).parent / "results" / "task3"

    results = run_experiment(selected_models, proportions, args.n_trials, args.budget)

    safe_model = (
        selected_models[0].replace("/", "_") if len(selected_models) == 1 else "multi"
    )
    csv_path = results_dir / f"rcwt_task3_{safe_model}.csv"
    agg_path = results_dir / f"rcwt_task3_{safe_model}_aggregates.json"

    save_csv(results, csv_path)
    save_aggregates(results, agg_path)
    print_summary(results)

    print(f"\nResults saved to {results_dir}/")
