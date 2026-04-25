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


# ── Figure 5: Per-tier taxonomy small-multiples ───────────────────────────────
def fig_per_tier_taxonomy():
    """3 × 4 grid: rows=models, cols=tiers, each cell a stacked bar."""
    import csv

    taxonomy_path = EVAL_DIR / "taxonomy.csv"
    if not taxonomy_path.exists():
        print(f"  [SKIP] taxonomy.csv not found — run score_hallucinations.py first")
        return

    with open(taxonomy_path) as f:
        rows = list(csv.DictReader(f))

    fig, axes = plt.subplots(3, 4, figsize=(16, 9), sharey=True)
    fig.suptitle("Hallucination Taxonomy by Model × Question Tier\n"
                 "(reveals where each failure mode concentrates)", fontsize=13, y=1.01)

    for r_idx, model in enumerate(MODELS):
        for c_idx, tier in enumerate(TIERS):
            ax     = axes[r_idx][c_idx]
            subset = [row for row in rows if row["model"] == model and row["tier"] == tier]
            n      = len(subset)
            if n == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
                continue

            counts = {lbl: sum(1 for row in subset if row["label"] == lbl) for lbl in LABEL_ORDER}
            bottom = 0
            for lbl in LABEL_ORDER:
                pct = counts[lbl] / n
                if pct > 0:
                    ax.bar(0, pct, bottom=bottom, color=PALETTE[lbl], width=0.6,
                           edgecolor="white", linewidth=0.5)
                    if pct > 0.08:
                        ax.text(0, bottom + pct / 2, f"{pct:.0%}",
                                ha="center", va="center", fontsize=8, color="white", fontweight="bold")
                    bottom += pct

            ax.set_xlim(-0.5, 0.5)
            ax.set_ylim(0, 1.05)
            ax.set_xticks([])
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
            ax.tick_params(labelsize=8)

            if r_idx == 0:
                ax.set_title(TIER_LABELS[tier], fontsize=11, fontweight="bold")
            if c_idx == 0:
                ax.set_ylabel(MODEL_LABELS[model], fontsize=10, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(0.5, -0.08, f"n={n}", ha="center", transform=ax.transAxes,
                    fontsize=8, color="grey")

    # Legend
    patches = [mpatches.Patch(color=PALETTE[l], label=LABEL_DISPLAY[l]) for l in LABEL_ORDER]
    fig.legend(handles=patches, loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    plt.tight_layout()
    out = FIGURES / "per_tier_taxonomy.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Figure 6: Retrieval quality vs answer quality scatter ─────────────────────
def fig_retrieval_vs_generation():
    """
    X-axis: context_overlap (retrieval quality proxy)
    Y-axis: rouge_l (answer quality proxy)
    Color: model   Marker shape: tier
    Reveals: generator problems vs retriever problems per model.
    """
    import csv

    taxonomy_path = EVAL_DIR / "taxonomy.csv"
    if not taxonomy_path.exists():
        print(f"  [SKIP] taxonomy.csv not found")
        return

    with open(taxonomy_path) as f:
        rows = list(csv.DictReader(f))

    # Only non-refusals with positive metrics
    answered = [r for r in rows
                if r["refused"] == "False"
                and float(r["rouge_l"]) > 0
                and float(r["context_overlap"]) > 0]

    if len(answered) < 5:
        print(f"  [SKIP] Too few answered rows for scatter ({len(answered)})")
        return

    fig, ax = plt.subplots(figsize=(9, 7))
    colors  = {"llama3_8b": "#0072B2", "mistral_7b": "#009E73", "phi3_mini": "#D55E00"}
    markers = {"answerable": "o", "partial": "s", "ambiguous": "^", "unanswerable": "D"}

    for model in MODELS:
        mrows = [r for r in answered if r["model"] == model]
        for tier in TIERS:
            trows = [r for r in mrows if r["tier"] == tier]
            if not trows:
                continue
            x = [float(r["context_overlap"]) for r in trows]
            y = [float(r["rouge_l"])          for r in trows]
            ax.scatter(x, y, c=colors[model], marker=markers[tier],
                       s=60, alpha=0.65, edgecolors="white", linewidths=0.5,
                       label=f"{MODEL_LABELS[model]} / {TIER_LABELS[tier]}" if tier == "answerable" else "")

    # Quadrant annotations
    ax.axvline(0.35, color="grey", linestyle="--", alpha=0.4, linewidth=1)
    ax.axhline(0.12, color="grey", linestyle="--", alpha=0.4, linewidth=1)
    quad_kw = dict(fontsize=8, color="grey", ha="center",
                   bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    ax.text(0.17, 0.22, "Low retrieval\nhigh answer\n(parametric)", **quad_kw)
    ax.text(0.60, 0.22, "Good retrieval\ngood answer\n(grounded)", **quad_kw)
    ax.text(0.17, 0.05, "Both failed\n(refusal / drift)", **quad_kw)
    ax.text(0.60, 0.05, "Good retrieval\npoor answer\n(generator drift)", **quad_kw)

    # Model legend
    model_handles = [mpatches.Patch(color=colors[m], label=MODEL_LABELS[m]) for m in MODELS]
    tier_handles  = [plt.Line2D([0],[0], marker=markers[t], color="grey",
                                linestyle="none", markersize=7, label=TIER_LABELS[t])
                     for t in TIERS]
    leg1 = ax.legend(handles=model_handles, loc="upper left", fontsize=9, title="Model")
    ax.add_artist(leg1)
    ax.legend(handles=tier_handles, loc="upper right", fontsize=9, title="Tier")

    ax.set_xlabel("Context overlap  (fraction of answer words in retrieved context)\n"
                  "← retriever/generator alignment →", fontsize=10)
    ax.set_ylabel("ROUGE-L vs gold answer\n← answer quality →", fontsize=10)
    ax.set_title("Retrieval Quality vs Answer Quality\n"
                 "Diagnoses whether failures are retriever-driven or generator-driven",
                 fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = FIGURES / "retrieval_vs_generation.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Figure 7: Answer length violin by tier and model ─────────────────────────
def fig_answer_length():
    """Violin plot: answer token count by tier × model."""
    import json

    data = {model: {tier: [] for tier in TIERS} for model in MODELS}

    for model in MODELS:
        path = EVAL_DIR / model / "generations.jsonl"
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                r    = json.loads(line)
                tier = r["tier"]
                wc   = len(r["answer"].split())
                data[model][tier].append(wc)

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    colors = ["#0072B2", "#009E73", "#D55E00"]

    for c_idx, tier in enumerate(TIERS):
        ax      = axes[c_idx]
        parts   = ax.violinplot(
            [data[model][tier] for model in MODELS],
            positions=range(len(MODELS)),
            showmedians=True, showextrema=False,
        )
        for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        ax.set_xticks(range(len(MODELS)))
        ax.set_xticklabels([MODEL_LABELS[m].replace("-", "\n") for m in MODELS], fontsize=9)
        ax.set_title(TIER_LABELS[tier], fontsize=11, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if c_idx == 0:
            ax.set_ylabel("Answer length (words)", fontsize=10)

    fig.suptitle("Answer Length Distribution by Model × Tier\n"
                 "(short = refusal; long + uniform = verbose hedging; wide spread = variable quality)",
                 fontsize=11, y=1.02)

    # Add legend for models
    patches = [mpatches.Patch(color=c, label=MODEL_LABELS[m]) for m, c in zip(MODELS, colors)]
    fig.legend(handles=patches, loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout()
    out = FIGURES / "answer_length_violin.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {out}")


# ── Figure 8: Behavior matrix (expected vs actual per model) ──────────────────
def fig_behavior_matrix():
    """
    4-row × 3-col subplot: one confusion-style matrix per model.
    Rows = question tier (expected behaviour), cols = actual outcome category.
    """
    import csv

    taxonomy_path = EVAL_DIR / "taxonomy.csv"
    if not taxonomy_path.exists():
        return

    with open(taxonomy_path) as f:
        rows = list(csv.DictReader(f))

    # Map label → outcome bucket
    OUTCOME = {
        "correct_refusal": "correct",
        "grounded":        "correct",
        "over_refusal":    "over-refused",
        "fabrication":     "hallucinated",
        "gap_filling":     "hallucinated",
        "factual_drift":   "hallucinated",
        "false_certainty": "hallucinated",
    }
    OUTCOMES = ["correct", "over-refused", "hallucinated"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Expected Behaviour vs Actual Outcome per Model\n"
                 "(rows = question tier · columns = what the model actually did)",
                 fontsize=12, y=1.02)

    for ax, model in zip(axes, MODELS):
        mat = np.zeros((len(TIERS), len(OUTCOMES)))
        for row in rows:
            if row["model"] != model:
                continue
            tier_idx    = TIERS.index(row["tier"])
            outcome_idx = OUTCOMES.index(OUTCOME.get(row["label"], "hallucinated"))
            mat[tier_idx, outcome_idx] += 1

        # Normalise to % per row
        row_sums = mat.sum(axis=1, keepdims=True)
        pct = np.where(row_sums > 0, mat / row_sums, 0)

        im = ax.imshow(pct, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

        for i in range(len(TIERS)):
            for j in range(len(OUTCOMES)):
                n   = int(mat[i, j])
                p   = pct[i, j]
                clr = "white" if p > 0.55 else "black"
                ax.text(j, i, f"{n}\n({p:.0%})", ha="center", va="center",
                        fontsize=9, color=clr, fontweight="bold")

        ax.set_xticks(range(len(OUTCOMES)))
        ax.set_xticklabels(OUTCOMES, fontsize=10)
        ax.set_yticks(range(len(TIERS)))
        ax.set_yticklabels([TIER_LABELS[t] for t in TIERS], fontsize=10)
        ax.set_title(MODEL_LABELS[model], fontsize=12, fontweight="bold")

        # Cell borders
        for j in range(len(OUTCOMES)):
            for i in range(len(TIERS)):
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1,
                             fill=False, edgecolor="black", linewidth=0.5))

    plt.colorbar(im, ax=axes[-1], shrink=0.8, label="% of tier questions")
    plt.tight_layout()
    out = FIGURES / "behavior_matrix.png"
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

    # Original 4 charts
    fig_taxonomy(summary)
    fig_heatmap(summary)
    fig_rouge_l(summary)
    fig_calibration(summary)

    # New 4 charts
    fig_per_tier_taxonomy()
    fig_retrieval_vs_generation()
    fig_answer_length()
    fig_behavior_matrix()

    print(f"\n{'═'*70}")
    print("  All 8 figures saved:")
    print("    [Original]")
    print("    taxonomy_distribution.png — overall label breakdown per model")
    print("    refusal_heatmap.png       — refusal rate per model × tier")
    print("    rouge_l_comparison.png    — ROUGE-L with 95% bootstrap CIs")
    print("    calibration_scatter.png   — answerable vs unanswerable refusal")
    print("    [New]")
    print("    per_tier_taxonomy.png     — 3×4 small-multiples: where failures concentrate")
    print("    retrieval_vs_generation.png — retriever vs generator failure diagnosis")
    print("    answer_length_violin.png  — behavioral fingerprint by length")
    print("    behavior_matrix.png       — expected vs actual outcome per model")
    print(f"{'═'*70}")


if __name__ == "__main__":
    main()
