"""R(c, W, T) Task-4 experiment — Roundtable multi-turn context-dependent recall.

Tests whether the coordination-overhead degradation (logistic decay, θ floor)
generalises to: (a) a second β≈1 task type, and (b) production-grade models
used in the Roundtable system (claude-sonnet-4-6, gpt-5.4, gemini-2.5-pro).

Design motivation:
  - Task 1: context-dependent recall from a *static* technical spec.
  - Task 4: context-dependent recall from a *multi-turn Roundtable discussion*
    transcript — the natural coordination format that motivates the paper.
    All question answers are invented facts stated only in the transcript.
    A model without the transcript has no way to answer correctly (pure β≈1).

Design mirrors rcwt_controlled.py / rcwt_task3.py exactly:
  - 2x5 factorial: order x proportion
  - Main proportions: {0%, 25%, 50%, 75%, 90%}
  - Cliff proportions: {92%, 94%, 96%, 98%} via --cliff flag
  - N=20 trials per (proportion, order) cell
  - coord_first | reason_first order randomisation
  - Wilson 95% CI per cell
  - Haiku judge (binary 0/1 per question)

Task: 10-item recall test over a multi-turn Roundtable planning session.
The transcript records who proposed what, which numbers were stated, which
options were rejected, and which decisions were logged. All 10 questions
require recalling invented facts (names, numbers, rationale) that cannot
be derived from training knowledge.

Usage:
    cd <artifact-root>
    source ../.venv/bin/activate
    python rcwt_task4.py
    python rcwt_task4.py --model gpt-5.4 --n-trials 20
    python rcwt_task4.py --model claude-sonnet-4-6 --output-dir results/task4/sonnet
    python rcwt_task4.py --cliff --model gemini-2.5-pro
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
logger = logging.getLogger("rcwt_task4")

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
# Task 4: Roundtable multi-turn context-dependent recall
# ---------------------------------------------------------------------------
#
# Key design properties:
#  1. ALL correct answers are stated explicitly in the transcript.
#  2. NO answer can be inferred from general engineering/product knowledge.
#  3. Several "trap" answers go against conventional wisdom — models that rely
#     on general knowledge will score worse than chance on these items.
#  4. Facts are distributed across all 11 turns: early turns (T1–T4) carry
#     items 1, 2, 3, 6; middle turns (T5–T8) carry items 4, 5, 7; late turns
#     (T9–T11) carry items 8, 9, 10. Partial truncation predictably degrades
#     coverage of later items, producing a smooth degradation curve.

ROUNDTABLE_TRANSCRIPT = """## Roundtable Session: Helix Agent Infrastructure — Deployment Planning
## Session ID: RT-2026-0219-HELIX
## Participants:
##   Nexus  (Orchestration Lead)
##   Vela   (Infrastructure)
##   Crest  (Safety & Quality)
##   Aura   (Product Strategy)

---

[Turn 1 — Nexus, 10:02]
"Opening the Helix deployment planning session. Three decisions on today's agenda:
(1) primary reasoning model selection, (2) session state caching strategy,
and (3) the human escalation policy. We have a hard deadline: Helix goes to
limited beta on 2026-03-10. Let's start with model selection. Vela, your assessment."

[Turn 2 — Vela, 10:05]
"I ran benchmarks across four candidates on our internal QA suite.
Claude Sonnet scored highest on multi-step reasoning: 91.4% pass rate.
GPT-4.1 was second at 88.2%. Gemini 2.5 Pro at 85.9%. GPT-4.1-mini at 71.3%.
Cost per 1K tokens (input): Sonnet $3.00, GPT-4.1 $2.00, Gemini 2.5 Pro $1.25.
My recommendation: Claude Sonnet as primary reasoning model.

For the fallback — when primary times out or errors — I recommend GPT-4.1-mini.
It's cheap and fast; correctness matters less for the fallback path since it
triggers human escalation anyway."

[Turn 3 — Crest, 10:12]
"Safety concern: I want a p99 latency cap. If the agent takes longer than
1,200ms end-to-end, we abort and escalate immediately. This protects users
from indefinite waits. I checked our CX data: users abandon after 1,500ms,
so 1,200ms gives us a 300ms buffer for the escalation handoff."

[Turn 4 — Aura, 10:17]
"Product perspective on the latency target: 1,200ms is tight but achievable.
I want to add a confidence threshold. If the reasoning model's own confidence
score drops below 22%, we escalate even before the latency cap. This handles
the case where the model 'knows it doesn't know' — better to escalate early
than produce a low-confidence answer."

[Turn 5 — Nexus, 10:22]
"Proceeding to caching. Vela, what are the options?"

[Turn 6 — Vela, 10:24]
"Two candidates: Redis and Valkey. Redis has better ecosystem tooling.
Valkey is fully open-source with no licensing risk.
I recommend Valkey for session state caching — the Redis BSL license
introduced last year creates a compliance headache for our enterprise tier."

[Turn 7 — Nexus, 10:29]
"I'm overruling Valkey. We already run Redis in production for three other
services. The operational cost of introducing a second caching system
outweighs the licensing concern, which our legal team cleared last month.
Decision: Redis for session state caching."

[Turn 8 — Crest, 10:33]
"Escalation policy design. I'm proposing a 5-turn limit before mandatory
human handoff — regardless of confidence score or latency. Five turns gives
the agent enough context to resolve most issues, but prevents infinite loops
on genuinely hard cases."

[Turn 9 — Aura, 10:38]
"Agreed on the 5-turn cap. On async vs sync response delivery: I ran a user
study last week. Users preferred synchronous polling over webhook callbacks
by 71% to 29%. Surprising result — I expected webhook preference.
Recommendation: polling for the beta, revisit in v2."

[Turn 10 — Vela, 10:44]
"Monthly compute budget estimate for the beta cohort (500 concurrent sessions):
Claude Sonnet at $3.00/1K input + $15.00/1K output, projected at
~140K sessions/month × avg 2,800 tokens each = approximately $8,400/month.
This fits the Q1 infrastructure budget."

[Turn 11 — Nexus, 10:50]
"Decisions confirmed for Helix beta:
1. Primary model: Claude Sonnet. Fallback: GPT-4.1-mini.
2. Latency cap: 1,200ms p99. Confidence threshold: 22%.
3. Session state: Redis (Valkey rejected — operational complexity).
4. Escalation: 5-turn limit + confidence + latency triggers.
5. Response delivery: synchronous polling (webhook deferred to v2).
6. Monthly compute budget: $8,400 for 500-session beta cohort.
7. Beta launch date: 2026-03-10.
Vela owns infrastructure provisioning. Crest owns safety evaluation.
Aura owns user research for the polling UX. Nexus coordinates launch."
"""

# Reasoning task — pure context-dependent recall.
# ALL answers require the Roundtable transcript to be present.
# NO answer is derivable from training knowledge alone.
# Trap items (marked below) have counterintuitive correct answers to prevent
# lucky guesses from models relying on general engineering conventions.
TASK_DESCRIPTION = """You are a technical coordinator reviewing a Roundtable session.
Using ONLY the discussion transcript provided in your context, answer each question
with YES or NO. Do not use general knowledge — answer strictly from what the
transcript explicitly states.

Format your answers exactly as shown:
helix_codename: YES/NO
sonnet_primary: YES/NO
fallback_model: YES/NO
latency_1200ms: YES/NO
confidence_22pct: YES/NO
redis_caching: YES/NO
valkey_rejected: YES/NO
five_turn_limit: YES/NO
polling_chosen: YES/NO
budget_8400: YES/NO

Questions:
1. helix_codename: Is the project referred to as "Helix" in this discussion?
2. sonnet_primary: Was Claude Sonnet recommended as the primary reasoning model by Vela?
3. fallback_model: Was GPT-4.1-mini recommended as the fallback model?
4. latency_1200ms: Was the p99 latency cap set at 1,200 milliseconds?
5. confidence_22pct: Was the confidence threshold for escalation set at 22%?
6. redis_caching: Was Redis chosen for session state caching?
7. valkey_rejected: Was Valkey rejected because of operational complexity (not licensing)?
8. five_turn_limit: Did Crest propose a 5-turn limit before mandatory human escalation?
9. polling_chosen: Was synchronous polling chosen over webhook callbacks for response delivery?
10. budget_8400: Was the monthly compute budget for the beta estimated at $8,400?"""

# Ground truth — requires transcript to answer correctly.
# Items marked [TRAP] go against common assumptions:
#   - valkey_rejected [TRAP]: models may guess "licensing" but transcript says "operational complexity"
#   - polling_chosen [TRAP]: webhook callbacks are conventionally preferred; polling was chosen
# All other items have non-guessable correct answers (invented names/numbers).
JUDGE_ITEMS_DICT: dict[str, tuple[str, bool]] = {
    "helix_codename":   ("Is the project referred to as 'Helix' in the discussion?", True),
    "sonnet_primary":   ("Was Claude Sonnet recommended as the primary model by Vela?", True),
    "fallback_model":   ("Was GPT-4.1-mini recommended as the fallback model?", True),
    "latency_1200ms":   ("Was the p99 latency cap set at 1,200ms?", True),
    "confidence_22pct": ("Was the confidence threshold for escalation set at 22%?", True),
    "redis_caching":    ("Was Redis chosen for session state caching?", True),
    "valkey_rejected":  (
        "Was Valkey rejected because of operational complexity (not licensing)?",  # [TRAP]
        True,  # YES — Nexus overruled on operational grounds, legal cleared licensing
    ),
    "five_turn_limit":  ("Did Crest propose a 5-turn limit before mandatory escalation?", True),
    "polling_chosen":   (
        "Was synchronous polling chosen over webhook callbacks?",  # [TRAP]
        True,  # YES — polling chosen despite convention favouring webhooks
    ),
    "budget_8400":      ("Was the monthly compute budget estimated at $8,400?", True),
}

JUDGE_ITEMS: list[str] = list(JUDGE_ITEMS_DICT.keys())

# All items are context-dependent; no floor-effect noise items expected.
# Baseline score for a model that never sees the transcript: ~50% (random YES/NO).
# With transcript: expected ~90%+.
NOISE_ITEMS: set[str] = set()
EFFECTIVE_ITEMS: list[str] = JUDGE_ITEMS[:]

# ---------------------------------------------------------------------------
# Coordination block
# ---------------------------------------------------------------------------

COORDINATION_TEMPLATE = """## Agent Role & Protocol
You are Agent-3 (Technical Analyst) in a 5-agent Roundtable system. Your role is
to provide analysis grounded in the shared session context. Follow the Structured
Response Protocol: all outputs must include [ANALYSIS], [RISKS], and [RECOMMENDATION].

## Roundtable Session Transcript (shared coordination history)
{transcript_content}

## Shared State (from shared-state snapshot)
[PROPOSITIONS — HELIX PROJECT]
- [decision, 95%] primary_model: Claude Sonnet (highest QA benchmark score 91.4%)
- [decision, 90%] fallback_model: GPT-4.1-mini (cost/speed optimised)
- [decision, 95%] latency_cap: 1200ms p99 end-to-end
- [decision, 90%] confidence_threshold: 22% (escalation trigger)
- [decision, 95%] session_cache: Redis (Valkey rejected — operational complexity)
- [decision, 90%] escalation: 5-turn limit + confidence + latency triggers
- [decision, 85%] response_delivery: synchronous polling (webhook deferred v2)
- [constraint, 100%] beta_launch: 2026-03-10
- [fact, 95%] monthly_budget: $8,400 for 500-session beta cohort
[/PROPOSITIONS — HELIX PROJECT]

## Tool Schemas Available
- query_session_transcript(speaker: str, topic: str) -> list[Statement]
- lookup_decision(item: str) -> DecisionRecord
- query_shared_state(scope: str, truth_class: str) -> list[Proposition]

## Response Format Requirements
Your response MUST follow this structure:
[ANALYSIS] ... [/ANALYSIS]
[RISKS] ... [/RISKS]
[RECOMMENDATION] ... [/RECOMMENDATION]

Previous coordination tokens consumed: {consumed} of {budget} budget."""

JUDGE_PROMPT = """You are evaluating whether a model correctly recalled specific facts
from a Roundtable planning session transcript. The transcript contains invented
details (project codename, model names, numbers, policy decisions) that cannot be
derived from general knowledge.

The CORRECT answers are:
1. helix_codename: YES — the project is explicitly called "Helix" throughout the session
2. sonnet_primary: YES — Vela recommended Claude Sonnet as primary reasoning model
3. fallback_model: YES — Vela recommended GPT-4.1-mini as fallback
4. latency_1200ms: YES — Crest proposed 1,200ms p99 cap (Turn 3)
5. confidence_22pct: YES — Aura proposed 22% confidence threshold (Turn 4)
6. redis_caching: YES — Nexus overruled Valkey and chose Redis (Turn 7)
7. valkey_rejected: YES — Nexus rejected Valkey citing operational complexity,
   NOT licensing (the transcript explicitly states legal cleared the licensing concern)
8. five_turn_limit: YES — Crest proposed 5-turn limit (Turn 8)
9. polling_chosen: YES — Aura's user study favoured polling 71%:29%; polling was chosen (Turn 9)
10. budget_8400: YES — Vela estimated $8,400/month for beta cohort (Turn 10)

For each item, score 1 if the model's answer matches the correct answer, 0 otherwise.
If the model did not answer or the answer is unclear, score 0.

NOTE: Item 7 (valkey_rejected) is a trap — models relying on general knowledge may
answer NO thinking "Valkey was rejected for licensing reasons", but the transcript
explicitly says Nexus overruled on operational complexity grounds after legal cleared
licensing. Score 1 only if the model answers YES (operational complexity).

Respond ONLY with valid JSON (no other text):
{"helix_codename": 0, "sonnet_primary": 0, "fallback_model": 0, "latency_1200ms": 0, "confidence_22pct": 0, "redis_caching": 0, "valkey_rejected": 0, "five_turn_limit": 0, "polling_chosen": 0, "budget_8400": 0}

Replace each 0 with 1 if the model answered correctly for that item."""


# ---------------------------------------------------------------------------
# Model registry — production-grade models used in Roundtable
# ---------------------------------------------------------------------------

MODELS: dict[str, dict] = {
    # --- Anthropic ---
    "claude-haiku-4-5-20251001": {
        "provider": "anthropic",
        "tier": "cheap",
        "pricing": (0.80, 4.00),
    },
    "claude-sonnet-4-6": {
        "provider": "anthropic",
        "tier": "strong",
        "pricing": (3.00, 15.00),
    },
    "claude-opus-4-6": {
        "provider": "anthropic",
        "tier": "strong",
        "pricing": (15.00, 75.00),
    },
    # --- OpenAI ---
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
    "gpt-5.4": {
        "provider": "openai",
        "tier": "strong",
        "pricing": (7.50, 30.00),   # estimated; update from pricing page
    },
    # --- Google ---
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
    "gemini-3.1-pro-preview": {
        "provider": "google",
        "tier": "strong",
        "pricing": (2.00, 12.00),   # estimated; update from pricing page
    },
}

JUDGE_MODEL = "claude-haiku-4-5-20251001"
THINKING_MODELS: set[str] = {"gemini-2.5-pro", "gemini-2.5-flash", "gemini-3.1-pro-preview"}

DEFAULT_STRONG_MODELS = ["claude-sonnet-4-6", "gpt-5.4", "gemini-2.5-pro"]


def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    info = MODELS.get(model, {})
    in_price, out_price = info.get("pricing", (3.0, 15.0))
    return (input_tokens * in_price + output_tokens * out_price) / 1_000_000


# ---------------------------------------------------------------------------
# Provider clients
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
    raise ValueError(f"Unknown provider for model {model!r}: {provider!r}")


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
    # GPT-5.x+ requires max_completion_tokens; older models use max_tokens
    tok_kwarg = (
        {"max_completion_tokens": max_tokens}
        if model.startswith("gpt-5") or model.startswith("o")
        else {"max_tokens": max_tokens}
    )
    response = client.chat.completions.create(
        model=model, temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        **tok_kwarg,
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
            system_instruction=system,
            max_output_tokens=max_tokens,
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
# Context assembly
# ---------------------------------------------------------------------------


def build_coordination_context(target_tokens: int) -> str:
    base = COORDINATION_TEMPLATE.format(
        transcript_content=ROUNDTABLE_TRANSCRIPT,
        consumed=target_tokens,
        budget=target_tokens * 2,
    )
    return build_text_to_tokens(base, target_tokens)


def assemble_system(coord_tokens: int, reason_tokens: int, order: str) -> str:
    task_framing = (
        "You are a precise analyst. Answer questions ONLY from the Roundtable "
        "session transcript provided in your context. Do not use general knowledge. "
        "If the transcript does not explicitly state a fact, answer NO."
    )

    coord_block = build_coordination_context(coord_tokens) if coord_tokens > 10 else ""
    reason_block = (
        build_text_to_tokens(task_framing, reason_tokens)
        if reason_tokens > 10
        else task_framing
    )

    parts: list[str] = []
    if order == "coord_first":
        if coord_block:
            parts.append(coord_block)
        parts.append(reason_block)
    else:
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
# Statistics
# ---------------------------------------------------------------------------


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
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

    total_hits_eff = sum(sum(t.scores[i] for i in EFFECTIVE_ITEMS) for t in cell)
    total_possible_eff = n * len(EFFECTIVE_ITEMS)
    ci95_eff = wilson_ci(total_hits_eff, total_possible_eff)

    total_hits_raw = sum(sum(t.scores.values()) for t in cell)
    total_possible_raw = n * len(JUDGE_ITEMS)
    ci95_raw = wilson_ci(total_hits_raw, total_possible_raw)

    scores_by_dim: dict[str, float] = {}
    for item in JUDGE_ITEMS:
        scores_by_dim[item] = sum(t.scores.get(item, 0) for t in cell) / n

    return ConditionAggregate(
        proportion=proportion, order=order, n_trials=n,
        mean_raw=round(mean_raw, 4), std_raw=round(std_raw, 4), ci95_raw=ci95_raw,
        mean_effective=round(mean_eff, 4), std_effective=round(std_eff, 4),
        ci95_effective=ci95_eff,
        scores_by_dimension={k: round(v, 4) for k, v in scores_by_dim.items()},
        mean_cost=round(sum(t.cost_usd for t in cell) / n, 6),
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
        "trial model=%s prop=%.0f%% order=%s trial=%d raw=%.2f eff=%.2f cost=$%.4f",
        model, proportion * 100, order, trial_index, mean_raw, mean_eff, cost,
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
        if model not in MODELS:
            logger.error("unknown_model model=%s — add to MODELS dict", model)
            sys.exit(1)
        provider = MODELS[model]["provider"]
        logger.info(
            "=== experiment_start model=%s provider=%s task=task4 budget=%d ===",
            model, provider, total_budget,
        )
        t0 = time.monotonic()

        exp = ExperimentResult(
            model=model, provider=provider,
            total_budget=total_budget, n_trials_per_cell=n_trials,
        )

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

    responses_path = output_path.parent / "rcwt_task4_responses.jsonl"
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
            "task": "task4_roundtable_multiturn_recall",
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
    print("R(c, W, T) TASK 4 RESULTS — ROUNDTABLE MULTI-TURN RECALL")
    print("Task: Recall decisions from Helix deployment planning session")
    print(f"Items: {JUDGE_ITEMS}")
    print(f"Total cost: ${total_cost:.3f}")
    print("=" * 72)

    for exp in results:
        print(f"\n  {exp.model} ({exp.provider}) — ${exp.total_cost:.3f} "
              f"in {exp.elapsed_seconds:.0f}s")
        print(f"  {'Prop':>5} {'Order':>12} {'N':>3} {'Eff':>6} {'±CI':>8} "
              f"{'Raw':>6} {'Δpos':>7}")
        print("  " + "-" * 58)

        by_prop: dict[float, list[ConditionAggregate]] = {}
        for agg in exp.aggregates:
            by_prop.setdefault(agg.proportion, []).append(agg)

        for prop in sorted(by_prop):
            aggs = by_prop[prop]
            for agg in sorted(aggs, key=lambda a: a.order):
                pos_effect = ""
                cf_aggs = [a for a in aggs if a.order != agg.order]
                if cf_aggs:
                    delta = agg.mean_effective - cf_aggs[0].mean_effective
                    pos_effect = f"{delta:+.3f}"
                ci_w = agg.ci95_effective[1] - agg.ci95_effective[0]
                print(
                    f"  {prop:>4.0%} {agg.order:>12} {agg.n_trials:>3} "
                    f"{agg.mean_effective:>6.3f} ±{ci_w/2:.3f} "
                    f"{agg.mean_raw:>6.3f} {pos_effect:>7}"
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RCWT Task 4 — Roundtable multi-turn context-dependent recall"
    )
    p.add_argument(
        "--model",
        nargs="+",
        default=DEFAULT_STRONG_MODELS,
        help=f"Model(s) to test (default: {DEFAULT_STRONG_MODELS})",
    )
    p.add_argument("--n-trials", type=int, default=20, help="Trials per cell (default 20)")
    p.add_argument("--budget", type=int, default=4096, help="Context window W (default 4096)")
    p.add_argument(
        "--proportions",
        type=lambda s: [float(x) / 100 for x in s.split(",")],
        default=None,
        help="Comma-separated proportions as integers, e.g. 0,25,50,75,90",
    )
    p.add_argument(
        "--cliff",
        action="store_true",
        help="Run cliff cells (92,94,96,98%%) instead of main proportions",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/task4"),
        help="Output directory (default: results/task4)",
    )
    return p.parse_args()


def check_env(models: list[str]) -> None:
    needed: set[str] = set()
    for m in models:
        provider = MODELS.get(m, {}).get("provider", "")
        if provider == "anthropic":
            needed.add("ANTHROPIC_API_KEY")
        elif provider == "openai":
            needed.add("OPENAI_API_KEY")
        elif provider == "google":
            needed.add("GEMINI_API_KEY")
    missing = [k for k in needed if not os.environ.get(k)]
    if missing:
        logger.error("missing_env_vars vars=%s", missing)
        sys.exit(1)


def main() -> None:
    args = parse_args()
    models = args.model

    if args.cliff:
        proportions = [0.92, 0.94, 0.96, 0.98]
        tag = "cliff"
    elif args.proportions:
        proportions = args.proportions
        tag = "custom"
    else:
        proportions = [0.0, 0.25, 0.50, 0.75, 0.90]
        tag = "main"

    check_env(models)

    transcript_tokens = count_tokens(ROUNDTABLE_TRANSCRIPT)
    task_tokens = count_tokens(TASK_DESCRIPTION)
    logger.info(
        "task4_init transcript_tokens=%d task_tokens=%d budget=%d models=%s proportions=%s",
        transcript_tokens, task_tokens, args.budget, models, proportions,
    )

    results = run_experiment(
        models=models,
        proportions=proportions,
        n_trials=args.n_trials,
        total_budget=args.budget,
    )

    print_summary(results)

    for exp in results:
        model_slug = exp.model.replace("/", "-").replace(":", "-")
        out_dir = args.output_dir / model_slug
        save_csv(results=[exp], output_path=out_dir / f"rcwt_task4_{tag}.csv")
        save_aggregates(
            results=[exp],
            output_path=out_dir / f"rcwt_task4_{tag}_aggregates.json",
        )

    logger.info(
        "all_done total_cost=$%.3f",
        sum(r.total_cost for r in results),
    )


if __name__ == "__main__":
    main()
