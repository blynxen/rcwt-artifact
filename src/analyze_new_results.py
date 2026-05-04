"""Post-experiment analysis: cliff N=20 and W=8192 results.

Run after cliff_n20 and w8192 experiments complete:
    python experiments/analyze_new_results.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit

RESULTS_DIR = Path(__file__).parent / "results"
CLIFF_N20_DIR = RESULTS_DIR / "cliff_n20"
W8192_DIR = RESULTS_DIR / "w8192"

MODEL_MAP = {
    "claude-haiku-4-5-20251001": "haiku",
    "claude-haiku-4-5": "haiku",
    "gpt-4.1-mini": "gpt",
    "gemini-2.0-flash": "gemini",
}

CLIFF_FILE_MAP = {
    "gemini-2.0-flash": CLIFF_N20_DIR / "gemini-2.0-flash_cliff_aggregates.json",
    "claude-haiku-4-5-20251001": CLIFF_N20_DIR / "claude-haiku-4-5-20251001_cliff_aggregates.json",
    "gpt-4.1-mini": CLIFF_N20_DIR / "gpt-4.1-mini_cliff_aggregates.json",
}

MAIN_SOURCES = {
    "gemini-2.0-flash": RESULTS_DIR / "rcwt_controlled_aggregates.json",
    "claude-haiku-4-5-20251001": RESULTS_DIR / "haiku" / "rcwt_controlled_aggregates.json",
    "gpt-4.1-mini": RESULTS_DIR / "gpt" / "rcwt_controlled_aggregates.json",
}

W8192_FILE_MAP = {
    "gemini-2.0-flash": W8192_DIR / "gemini-2.0-flash_w8192_aggregates.json",
    "claude-haiku-4-5-20251001": W8192_DIR / "claude-haiku-4-5-20251001_w8192_aggregates.json",
    "gpt-4.1-mini": W8192_DIR / "gpt-4.1-mini_w8192_aggregates.json",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, centre - half), min(1.0, centre + half))


def load_aggregates(path: Path) -> list[dict]:
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw[0]["aggregates"]
    return raw["aggregates"]


def pooled_by_proportion(aggs: list[dict]) -> dict[float, dict]:
    """Pool coord_first + reason_first per proportion."""
    by_prop: dict[float, list[dict]] = {}
    for a in aggs:
        p = a["proportion"]
        by_prop.setdefault(p, []).append(a)
    result = {}
    for p, cells in by_prop.items():
        mean = float(np.mean([c["mean_effective"] for c in cells]))
        n_total = sum(c["n_trials"] for c in cells)
        # Compute pooled hits for Wilson CI
        hits = sum(
            round(c["mean_effective"] * c["n_trials"] * 8)  # 8 effective items
            for c in cells
        )
        possible = n_total * 8
        ci = wilson_ci(hits, possible)
        result[p] = {"mean": mean, "n": n_total, "ci": ci}
    return result


def logistic(p, R0, k, p0):
    return R0 / (1 + np.exp(k * (p - p0)))


def fit_logistic(proportions: list[float], scores: list[float]) -> dict:
    try:
        popt, pcov = curve_fit(
            logistic, proportions, scores,
            p0=[1.0, 10.0, 0.90],
            bounds=([0.5, 0, 0.5], [1.0, 200.0, 1.0]),
            maxfev=10000,
        )
        R0, k, p0 = popt
        predicted = [logistic(p, R0, k, p0) for p in proportions]
        ss_res = sum((s - pr) ** 2 for s, pr in zip(scores, predicted))
        ss_tot = sum((s - np.mean(scores)) ** 2 for s in scores)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        n = len(proportions)
        aic = n * math.log(ss_res / n) + 2 * 3  # 3 params
        return {"R0": R0, "k": k, "p0": p0, "r2": r2, "aic": aic}
    except Exception as e:
        return {"error": str(e)}


def analyze_cliff_n20():
    print("=" * 60)
    print("CLIFF N=20 RESULTS")
    print("=" * 60)

    for model_id, path in CLIFF_FILE_MAP.items():
        if not path.exists():
            print(f"\n{model_id}: NOT FOUND ({path})")
            continue

        main_aggs = load_aggregates(MAIN_SOURCES[model_id])
        cliff_aggs = load_aggregates(path)
        all_aggs = main_aggs + cliff_aggs

        combined = pooled_by_proportion(all_aggs)
        proportions = sorted(combined.keys())

        print(f"\n{model_id}:")
        print(f"  {'Prop':>6} {'Mean':>6} {'CI95':>18} {'N':>4}")
        print(f"  {'-'*38}")
        for p in proportions:
            d = combined[p]
            print(f"  {p:>6.1%} {d['mean']:>6.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}] {d['n']:>4}")

        # Fit logistic on combined data
        ps = [p for p in proportions]
        ss = [combined[p]["mean"] for p in proportions]
        fit = fit_logistic(ps, ss)
        if "error" not in fit:
            print(f"\n  Logistic fit: R0={fit['R0']:.3f} k={fit['k']:.1f} p0={fit['p0']:.3f} R²={fit['r2']:.4f}")
            print(f"  c* (cliff point) = {fit['p0']:.1%}")
        else:
            print(f"\n  Fit error: {fit['error']}")


def analyze_w8192():
    print("\n" + "=" * 60)
    print("W=8192 SCALING RESULTS")
    print("=" * 60)

    # Also load W=4096 results for comparison
    w4096: dict[str, dict] = {}
    for model_id, path in MAIN_SOURCES.items():
        if path.exists():
            aggs = load_aggregates(path)
            w4096[model_id] = pooled_by_proportion(aggs)

    for model_id, path in W8192_FILE_MAP.items():
        if not path.exists():
            print(f"\n{model_id}: NOT FOUND")
            continue

        aggs = load_aggregates(path)
        combined = pooled_by_proportion(aggs)
        proportions = sorted(combined.keys())

        print(f"\n{model_id}:")
        print(f"  {'Prop':>6} {'W=4K':>6} {'W=8K':>6} {'Δ':>7}")
        print(f"  {'-'*30}")
        for p in proportions:
            d8 = combined[p]
            d4 = w4096.get(model_id, {}).get(p)
            d4_str = f"{d4['mean']:.3f}" if d4 else "  N/A"
            delta_str = f"{d8['mean'] - d4['mean']:+.3f}" if d4 else "   N/A"
            print(f"  {p:>6.1%} {d4_str:>6} {d8['mean']:>6.3f} {delta_str:>7}")

        # Fit logistic on W=8192 data
        ps = [p for p in proportions]
        ss = [combined[p]["mean"] for p in proportions]
        fit = fit_logistic(ps, ss)
        if "error" not in fit:
            c4 = None
            if model_id in w4096:
                ps4 = sorted(w4096[model_id].keys())
                ss4 = [w4096[model_id][p]["mean"] for p in ps4]
                fit4 = fit_logistic(ps4, ss4)
                c4 = fit4.get("p0")
            print(f"\n  W=8192 fit: R0={fit['R0']:.3f} k={fit['k']:.1f} p0={fit['p0']:.3f} R²={fit['r2']:.4f}")
            if c4:
                print(f"  c* comparison: W=4096 → {c4:.3f} | W=8192 → {fit['p0']:.3f} "
                      f"(Δ={fit['p0']-c4:+.3f})")
            w_inv = ("INVARIANT" if c4 and abs(fit['p0'] - c4) < 0.02 else
                     "SHIFTS" if c4 else "N/A")
            print(f"  W-invariance: {w_inv}")


if __name__ == "__main__":
    analyze_cliff_n20()
    analyze_w8192()
