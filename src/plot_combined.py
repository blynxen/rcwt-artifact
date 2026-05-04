"""
R(c,W,T) Combined Plot -- All 6 Models (Cheap + Strong Tiers)

Generates a 3-panel figure:
  1. Absolute fact recall vs coordination proportion
  2. Normalized degradation (% of each model's 0% baseline)
  3. Cheap vs Strong tier average comparison
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

RESULTS_DIR = Path(__file__).parent / "results"
CHEAP_FILE = RESULTS_DIR / "rcwt_v2_cheap_aggregates.json"
OUTPUT_FILE = RESULTS_DIR / "rcwt_combined.png"

PROPORTIONS = [0.0, 0.10, 0.25, 0.50, 0.75, 0.90]

# -- Strong tier data (hardcoded from multi-run results) ----------------------
STRONG_DATA: dict[str, dict[str, Any]] = {
    "claude-4-sonnet": {
        "provider": "anthropic",
        "scores": [0.80, 0.80, 0.76, 0.80, 0.80, 0.56],
    },
    "gpt-4.1": {
        "provider": "openai",
        "scores": [0.86, 0.82, 0.78, 0.78, 0.76, 0.60],
    },
    "gemini-2.5-pro": {
        "provider": "google",
        "scores": [0.80, 0.84, 0.78, 0.80, 0.80, 0.64],
    },
}

# -- Color palette by provider x tier ----------------------------------------
COLORS = {
    "anthropic": {"cheap": "#E07B53", "strong": "#B5432A"},
    "openai":    {"cheap": "#6BA368", "strong": "#2D6A2E"},
    "google":    {"cheap": "#5B8BD4", "strong": "#2856A3"},
}

# -- Display names ------------------------------------------------------------
DISPLAY_NAMES: dict[str, str] = {
    "claude-haiku-4-5-20251001": "Haiku 4.5",
    "gpt-4.1-mini": "GPT-4.1-mini",
    "gemini-2.0-flash": "Gemini Flash",
    "claude-4-sonnet": "Sonnet 4",
    "gpt-4.1": "GPT-4.1",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
}


def load_cheap_data() -> dict[str, dict[str, Any]]:
    """Load cheap tier aggregates from JSON."""
    with open(CHEAP_FILE) as fh:
        raw: list[dict[str, Any]] = json.load(fh)

    result: dict[str, dict[str, Any]] = {}
    for entry in raw:
        model = entry["model"]
        scores = [agg["mean_score"] for agg in entry["aggregates"]]
        result[model] = {
            "provider": entry["provider"],
            "scores": scores,
        }
    return result


def normalize(scores: list[float]) -> list[float]:
    """Normalize scores as percentage of the 0%-proportion baseline."""
    baseline = scores[0]
    if baseline == 0:
        return [0.0] * len(scores)
    return [s / baseline * 100.0 for s in scores]


def setup_style() -> None:
    """Apply publication-quality defaults."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def plot_panel_absolute(
    ax: plt.Axes,
    cheap: dict[str, dict[str, Any]],
    strong: dict[str, dict[str, Any]],
) -> None:
    """Panel 1: absolute fact recall scores for all 6 models."""
    x = np.array(PROPORTIONS)

    for model, data in cheap.items():
        color = COLORS[data["provider"]]["cheap"]
        label = DISPLAY_NAMES.get(model, model)
        ax.plot(x, data["scores"], "o-", color=color, label=label,
                markersize=7, linewidth=1.8, markeredgecolor="white",
                markeredgewidth=0.8)

    for model, data in strong.items():
        color = COLORS[data["provider"]]["strong"]
        label = DISPLAY_NAMES.get(model, model)
        ax.plot(x, data["scores"], "s-", color=color, label=label,
                markersize=7, linewidth=1.8, markeredgecolor="white",
                markeredgewidth=0.8)

    ax.set_xlabel("Coordination Proportion", fontsize=11, fontweight="medium")
    ax.set_ylabel("Fact Recall Score", fontsize=11, fontweight="medium")
    ax.set_title("(a) Absolute Fact Recall", fontsize=12, fontweight="bold",
                 pad=10)
    ax.set_ylim(0.40, 1.0)
    ax.set_xlim(-0.02, 0.92)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower left", ncol=2, fontsize=8.5)


def plot_panel_normalized(
    ax: plt.Axes,
    cheap: dict[str, dict[str, Any]],
    strong: dict[str, dict[str, Any]],
) -> None:
    """Panel 2: normalized degradation (% of own baseline)."""
    x = np.array(PROPORTIONS)

    for model, data in cheap.items():
        color = COLORS[data["provider"]]["cheap"]
        label = DISPLAY_NAMES.get(model, model)
        normed = normalize(data["scores"])
        ax.plot(x, normed, "o-", color=color, label=label,
                markersize=7, linewidth=1.8, markeredgecolor="white",
                markeredgewidth=0.8)

    for model, data in strong.items():
        color = COLORS[data["provider"]]["strong"]
        label = DISPLAY_NAMES.get(model, model)
        normed = normalize(data["scores"])
        ax.plot(x, normed, "s-", color=color, label=label,
                markersize=7, linewidth=1.8, markeredgecolor="white",
                markeredgewidth=0.8)

    ax.axhline(y=100, color="#999999", linewidth=0.8, linestyle="--",
               alpha=0.6)
    ax.set_xlabel("Coordination Proportion", fontsize=11, fontweight="medium")
    ax.set_ylabel("% of Own Baseline", fontsize=11, fontweight="medium")
    ax.set_title("(b) Normalized Degradation", fontsize=12, fontweight="bold",
                 pad=10)
    ax.set_ylim(55, 115)
    ax.set_xlim(-0.02, 0.92)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%d%%"))
    ax.legend(loc="lower left", ncol=2, fontsize=8.5)


def plot_panel_tiers(
    ax: plt.Axes,
    cheap: dict[str, dict[str, Any]],
    strong: dict[str, dict[str, Any]],
) -> None:
    """Panel 3: tier-averaged comparison (cheap vs strong)."""
    x = np.array(PROPORTIONS)

    cheap_scores = np.array([d["scores"] for d in cheap.values()])
    strong_scores = np.array([d["scores"] for d in strong.values()])

    cheap_mean = cheap_scores.mean(axis=0)
    cheap_std = cheap_scores.std(axis=0)
    strong_mean = strong_scores.mean(axis=0)
    strong_std = strong_scores.std(axis=0)

    # Cheap tier
    ax.plot(x, cheap_mean, "o-", color="#E07B53", label="Cheap Tier (avg)",
            markersize=8, linewidth=2.2, markeredgecolor="white",
            markeredgewidth=1.0)
    ax.fill_between(x, cheap_mean - cheap_std, cheap_mean + cheap_std,
                    color="#E07B53", alpha=0.12)

    # Strong tier
    ax.plot(x, strong_mean, "s-", color="#2856A3", label="Strong Tier (avg)",
            markersize=8, linewidth=2.2, markeredgecolor="white",
            markeredgewidth=1.0)
    ax.fill_between(x, strong_mean - strong_std, strong_mean + strong_std,
                    color="#2856A3", alpha=0.12)

    # Annotate endpoint deltas
    cheap_drop = (1 - cheap_mean[-1] / cheap_mean[0]) * 100
    strong_drop = (1 - strong_mean[-1] / strong_mean[0]) * 100
    ax.annotate(
        f"-{cheap_drop:.0f}%",
        xy=(0.90, cheap_mean[-1]),
        xytext=(0.82, cheap_mean[-1] - 0.06),
        fontsize=9, fontweight="bold", color="#B5432A",
        arrowprops={"arrowstyle": "->", "color": "#B5432A", "lw": 1.0},
    )
    ax.annotate(
        f"-{strong_drop:.0f}%",
        xy=(0.90, strong_mean[-1]),
        xytext=(0.82, strong_mean[-1] + 0.06),
        fontsize=9, fontweight="bold", color="#2856A3",
        arrowprops={"arrowstyle": "->", "color": "#2856A3", "lw": 1.0},
    )

    ax.set_xlabel("Coordination Proportion", fontsize=11, fontweight="medium")
    ax.set_ylabel("Fact Recall Score", fontsize=11, fontweight="medium")
    ax.set_title("(c) Cheap vs Strong Tier", fontsize=12, fontweight="bold",
                 pad=10)
    ax.set_ylim(0.40, 1.0)
    ax.set_xlim(-0.02, 0.92)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.legend(loc="lower left", fontsize=9.5)


def main() -> None:
    setup_style()

    cheap = load_cheap_data()
    strong = STRONG_DATA

    logger.info("Cheap tier models: %s", list(cheap.keys()))
    logger.info("Strong tier models: %s", list(strong.keys()))

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), dpi=200)
    fig.suptitle(
        "R(c,W,T): Fact Recall Degradation Under Coordination Load",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plot_panel_absolute(axes[0], cheap, strong)
    plot_panel_normalized(axes[1], cheap, strong)
    plot_panel_tiers(axes[2], cheap, strong)

    fig.tight_layout(rect=[0, 0, 1, 0.94])

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    logger.info("Saved combined plot to %s", OUTPUT_FILE)

    # Print summary stats
    for tier_name, tier_data in [("CHEAP", cheap), ("STRONG", strong)]:
        scores_at_0 = [d["scores"][0] for d in tier_data.values()]
        scores_at_90 = [d["scores"][-1] for d in tier_data.values()]
        mean_0 = np.mean(scores_at_0)
        mean_90 = np.mean(scores_at_90)
        drop_pct = (1 - mean_90 / mean_0) * 100
        logger.info(
            "%s tier: baseline=%.3f, at 90%%=%.3f, drop=%.1f%%",
            tier_name, mean_0, mean_90, drop_pct,
        )


if __name__ == "__main__":
    main()
