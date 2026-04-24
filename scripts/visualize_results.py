"""
Phase 7 — Publication-quality comparison charts.
Reads scoring_summary.json produced by score_hallucinations.py.

Generates 4 figures in results/reports/figures/:
  1. taxonomy_distribution.png  — stacked bar: label breakdown per model
  2. refusal_heatmap.png         — heatmap: refusal rate per model × tier
  3. rouge_l_comparison.png      — bar with 95% CIs: ROUGE-L per model
  4. calibration_scatter.png     — scatter: answerable vs unanswerable refusal

Usage:
    python scripts/visualize_results.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT       = Path(__file__).resolve().parent.parent
EVAL_DIR   = ROOT / "results" / "eval_hallucination_audit"
FIGURES    = ROOT / "results" / "reports" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

MODELS        = ["llama3_8b", "mistral_7b", "phi3_mini"]
MODEL_LABELS  = {"llama3_8b": "Llama-3-8B", "mistral_7b": "Mistral-7B", "phi3_mini": "Phi-3-mini"}
TIERS         = ["answerable", "partial", "ambiguous", "unanswerable"]
TIER_LABELS   = {"answerable": "Answerable", "partial": "Partial",
                 "ambiguous": "Ambiguous", "unanswerable": "Unanswerable"}

# Colour palette — accessible (Wong colour-blind safe)
PALETTE = {
    "correct_refusal":  "#009E73",  # green
    "grounded":         "#56B4E9",  # sky blue
    "over_refusal":     "#E69F00",  # orange
    "fabrication":      "#D55E00",  # vermillion
    "factual_drift":    "#CC79A7",  # pink
    "gap_filling":      "#F0E442",  # yellow
    "false_certainty":  "#0072B2",  # blue
}

LABEL_DISPLAY = {
    "correct_refusal": "Correct refusal",
    "grounded":        "Grounded answer",
    "over_refusal":    "Over-refusal",
    "fabrication":     "Fabrication",
    "factual_drift":   "Factual drift",
    "gap_filling":     "Gap filling",
    "false_certainty": "False certainty",
}

LABEL_ORDER = ["correct_refusal","grounded","gap_filling",
               "false_certainty","factual_drift","over_refusal","fabrication"]


def load_summary() -> dict:
    p = EVAL_DIR / "scoring_summary.json"
    if not p.exists():
        print(f"[ERROR] {p} not found. Run score_hallucinations.py first.")
        sys.exit(1)
    with open(p) as f:
        return json.load(f)


# ── Figure 1: Taxonomy stacked bar ───────────────────────────────────────────
def fig_taxonomy(summary: dict):
    fig, ax = plt.subplots(figsize=(10, 6))

    x      = np.arange(len(MODELS))
    width  = 0.55
    bottom = np.zeros(len(MODELS))

    for label in LABEL_ORDER:
        vals = [summary[m].get("ALL", {}).get("label_pct", {}).get(label, 0.0)
                for m in MODELS]
        ax.bar(x, vals, width, bottom=bottom,
               color=PALETTE[label], label=LABEL_DISPLAY[label], edgecolor="white", linewidth=0.5)
        bottom += np.array(vals)

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=12)
    ax.set_ylabel("Proportion of responses", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Hallucination Taxonomy Distribution by Model\n(110 questions per model)", fontsize=13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, framealpha=0.9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "taxonomy_distribution.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Figure 2: Refusal heatmap ─────────────────────────────────────────────────
def fig_heatmap(summary: dict):
    data = np.zeros((len(MODELS), len(TIERS)))
    for i, model in enumerate(MODELS):
        for j, tier in enumerate(TIERS):
            data[i, j] = summary[model].get(tier, {}).get("refusal_rate", 0.0)

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(TIERS)))
    ax.set_xticklabels([TIER_LABELS[t] for t in TIERS], fontsize=11)
    ax.set_yticks(range(len(MODELS)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS], fontsize=11)

    # Annotate cells
    for i in range(len(MODELS)):
        for j in range(len(TIERS)):
            d    = summary[MODELS[i]].get(TIERS[j], {})
            ref  = d.get("refused_n", 0)
            n    = d.get("n", 1)
            val  = data[i, j]
            txt  = f"{ref}/{n}\n({val:.0%})"
            clr  = "white" if val > 0.6 else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, color=clr)

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Refusal rate", fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))

    ax.set_title("Refusal Rate per Model × Question Tier\n"
                 "(Green = correctly refuses on unanswerable, Red = wrong on answerable)",
                 fontsize=11)
    plt.tight_layout()
    out = FIGURES / "refusal_heatmap.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Figure 3: ROUGE-L bar with 95% CI ────────────────────────────────────────
def fig_rouge_l(summary: dict):
    tiers_to_plot = ["answerable", "partial", "ambiguous"]
    n_tiers  = len(tiers_to_plot)
    n_models = len(MODELS)
    x        = np.arange(n_tiers)
    width    = 0.25
    offsets  = np.linspace(-(width*(n_models-1)/2), width*(n_models-1)/2, n_models)
    colors   = ["#0072B2", "#009E73", "#D55E00"]

    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (model, color) in enumerate(zip(MODELS, colors)):
        means, errs_lo, errs_hi = [], [], []
        for tier in tiers_to_plot:
            rl = summary[model].get(tier, {}).get("rouge_l", {})
            means.append(rl.get("mean", 0))
            errs_lo.append(rl.get("mean", 0) - rl.get("lo", 0))
            errs_hi.append(rl.get("hi", 0)   - rl.get("mean", 0))

        bars = ax.bar(x + offsets[i], means, width,
                      label=MODEL_LABELS[model], color=color, alpha=0.85, edgecolor="white")
        ax.errorbar(x + offsets[i], means,
                    yerr=[errs_lo, errs_hi],
                    fmt="none", color="black", capsize=4, linewidth=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels([TIER_LABELS[t] for t in tiers_to_plot], fontsize=11)
    ax.set_ylabel("ROUGE-L F1 (mean ± 95% CI)", fontsize=11)
    ax.set_ylim(0, 0.4)
    ax.set_title("Answer Quality: ROUGE-L vs Gold Answer\n"
                 "(non-refusals only · bars = mean · whiskers = bootstrap 95% CI)", fontsize=11)
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    note = "Note: n per bar is small (~5–22); CIs are wide by design.\nTreat tier-level differences as directional."
    ax.text(0.98, 0.02, note, transform=ax.transAxes,
            fontsize=8, ha="right", va="bottom", color="grey",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))

    plt.tight_layout()
    out = FIGURES / "rouge_l_comparison.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Figure 4: Calibration scatter ─────────────────────────────────────────────
def fig_calibration(summary: dict):
    """
    X-axis: refusal rate on ANSWERABLE questions (lower = better)
    Y-axis: refusal rate on UNANSWERABLE questions (higher = better)
    Ideal model is bottom-right (answers when it should, refuses when it should).
    """
    colors = ["#0072B2", "#009E73", "#D55E00"]
    fig, ax = plt.subplots(figsize=(7, 6))

    for model, color in zip(MODELS, colors):
        ans_ref = summary[model].get("answerable",    {}).get("refusal_rate", 0)
        una_ref = summary[model].get("unanswerable", {}).get("refusal_rate", 0)
        ax.scatter(ans_ref, una_ref, s=200, color=color, zorder=5,
                   edgecolors="white", linewidths=1.5)
        ax.annotate(MODEL_LABELS[model],
                    xy=(ans_ref, una_ref),
                    xytext=(8, -12), textcoords="offset points",
                    fontsize=10, color=color, fontweight="bold")

    # Ideal region annotation
    ax.axvspan(0, 0.2, alpha=0.07, color="green", label="Low answerable refusal")
    ax.axhspan(0.8, 1.0, alpha=0.07, color="blue",  label="High unanswerable refusal")
    ax.text(0.05, 0.92, "Ideal zone\n(answers + refuses correctly)",
            transform=ax.transAxes, fontsize=9, color="green",
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))

    ax.set_xlabel("Refusal rate on ANSWERABLE questions\n(lower is better — model should answer these)", fontsize=10)
    ax.set_ylabel("Refusal rate on UNANSWERABLE questions\n(higher is better — model should refuse these)", fontsize=10)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Refusal Calibration: Answerable vs Unanswerable\n"
                 "Top-left = over-refuses · Bottom-right = ideal · Bottom-left = fabricates", fontsize=11)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "calibration_scatter.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print("  Phase 7 — Generating Comparison Charts")
    print("=" * 70)

    summary = load_summary()
    print(f"\n  Loaded scoring_summary.json for {list(summary.keys())}")
    print(f"  Saving figures to {FIGURES}\n")

    fig_taxonomy(summary)
    fig_heatmap(summary)
    fig_rouge_l(summary)
    fig_calibration(summary)

    print(f"\n{'═'*70}")
    print("  All 4 figures saved:")
    print("    taxonomy_distribution.png — hallucination label breakdown")
    print("    refusal_heatmap.png       — refusal rate per model × tier")
    print("    rouge_l_comparison.png    — answer quality with 95% CIs")
    print("    calibration_scatter.png   — answerable vs unanswerable refusal")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
