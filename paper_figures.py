"""Paper-Figuren für Privacy-Preserving Embeddings via Phase Coherence.

4 Figures:
1. Security-Utility Tradeoff (DCPE vs Phase Coherence)
2. Ablation Heatmap (welche Nichtlinearität erzeugt den Floor)
3. Phase Coherence Mechanism (Ratio vs d + Ablation-Barplot)
4. SHA-256 Avalanche (Key-Distanz vs Score-Kollaps)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
except ImportError:
    raise SystemExit("matplotlib required: pip install matplotlib")


RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")


def load_json(name: str) -> dict:
    path = os.path.join(RESULTS_DIR, name)
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Security-Utility Tradeoff ──────────────────────────────

def fig_security_utility(ag_news_path: str, ng20_path: str = ""):
    fig, ax = plt.subplots(1, 1, figsize=(7, 5))

    for path, marker, dataset_label in [
        (ag_news_path, "o", "AG News"),
        (ng20_path, "s", "20 Newsgroups"),
    ]:
        if not path or not os.path.exists(os.path.join(RESULTS_DIR, path)):
            continue
        data = load_json(path)
        for r in data["results"]:
            m = r["metrics"]["mean"]
            method = r["method"]
            recall = m["recall_at_k"]
            pair_corr = m["pair_corr"]
            label = method
            if "dcpe" in method:
                noise = r["params"].get("noise_std", "?")
                label = f"DCPE σ={noise}"
                color = "#e74c3c"
                alpha = 0.5 + 0.5 * (1 - float(noise))
            elif "wave_superpose" in method:
                label = "Phase Coherence"
                color = "#2ecc71"
                alpha = 1.0
            elif "baseline" in method:
                label = "Baseline (no encryption)"
                color = "#95a5a6"
                alpha = 0.7
            else:
                continue

            ax.scatter(recall, pair_corr, s=120, c=color, alpha=alpha,
                      marker=marker, edgecolors="black", linewidth=0.5, zorder=5)
            ax.annotate(f"{label}\n({dataset_label})", (recall, pair_corr),
                       textcoords="offset points", xytext=(8, 5),
                       fontsize=7, color=color)

    ax.set_xlabel("Recall@10 (Utility →)", fontsize=11)
    ax.set_ylabel("Pairwise Correlation (Leakage →)", fontsize=11)
    ax.set_title("Security–Utility Tradeoff: Phase Coherence vs DCPE/SAP", fontsize=12)
    ax.set_xlim(-0.05, 1.1)
    ax.set_ylim(-0.05, 1.1)
    ax.axhline(0, color="gray", lw=0.5, ls="--")
    ax.axvline(0, color="gray", lw=0.5, ls="--")
    ax.annotate("IDEAL\n(high recall,\nlow leakage)", xy=(0.95, 0.05),
               fontsize=9, color="green", ha="center", style="italic")
    fig.tight_layout()
    return fig


# ── Figure 2: Ablation Heatmap ───────────────────────────────────────

def fig_ablation_heatmap(ablation_path: str):
    data = load_json(ablation_path)
    results = data["results"]

    ablations = []
    dims = sorted(set(r["dim"] for r in results))
    for r in results:
        if r["ablation"] not in ablations:
            ablations.append(r["ablation"])

    matrix = np.zeros((len(ablations), len(dims)))
    for r in results:
        i = ablations.index(r["ablation"])
        j = dims.index(r["dim"])
        matrix[i, j] = max(1.0, r["ratio"])

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   norm=LogNorm(vmin=1, vmax=max(20000, matrix.max())))
    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims], fontsize=9)
    ax.set_yticks(range(len(ablations)))
    ax.set_yticklabels(ablations, fontsize=9)
    ax.set_xlabel("Dimension d", fontsize=11)
    ax.set_ylabel("Disabled Nonlinearity", fontsize=11)
    ax.set_title("Security Ratio by Nonlinearity Ablation (log scale)", fontsize=12)

    for i in range(len(ablations)):
        for j in range(len(dims)):
            val = matrix[i, j]
            color = "white" if val > 100 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                   fontsize=7, color=color, fontweight="bold")

    fig.colorbar(im, ax=ax, label="Magnitude Ratio (correct/wrong key)")
    fig.tight_layout()
    return fig


# ── Figure 3: Phase Coherence Mechanism ──────────────────────────────

def fig_phase_coherence(ablation_path: str):
    data = load_json(ablation_path)
    results = data["results"]
    dims = sorted(set(r["dim"] for r in results))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Ratio vs d for baseline, only_decoy, all_disabled
    for label, color, ls in [
        ("baseline", "#e74c3c", "-"),
        ("only_decoy", "#2ecc71", "--"),
        ("all_disabled", "#3498db", ":"),
    ]:
        ratios = [next((r["ratio"] for r in results if r["dim"] == d and r["ablation"] == label), 0) for d in dims]
        ax1.plot(dims, ratios, f"{ls}o", color=color, label=label, markersize=6, linewidth=2)

    ax1.set_xlabel("Dimension d", fontsize=11)
    ax1.set_ylabel("Security Ratio (log)", fontsize=11)
    ax1.set_yscale("log")
    ax1.set_title("Security Ratio vs Dimension", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Right: Barplot at d=384 — effect of each ablation
    d384 = [r for r in results if r["dim"] == 384]
    baseline_ratio = next((r["ratio"] for r in d384 if r["ablation"] == "baseline"), 50)
    bars = [(r["ablation"], r["ratio"] - baseline_ratio) for r in d384 if r["ablation"] != "baseline"]
    bars.sort(key=lambda x: x[1], reverse=True)

    names = [b[0].replace("only_", "").replace("all_disabled", "ALL OFF") for b in bars]
    deltas = [b[1] for b in bars]
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in deltas]

    ax2.barh(range(len(bars)), deltas, color=colors, edgecolor="black", linewidth=0.5)
    ax2.set_yticks(range(len(bars)))
    ax2.set_yticklabels(names, fontsize=9)
    ax2.set_xlabel("Δ Ratio vs Baseline", fontsize=11)
    ax2.set_title("Ablation Impact at d=384", fontsize=12)
    ax2.axvline(0, color="black", lw=1)
    ax2.grid(True, axis="x", alpha=0.3)

    fig.tight_layout()
    return fig


# ── Figure 4: SHA-256 Avalanche ──────────────────────────────────────

def fig_avalanche():
    from embedding_lab import METHOD_REGISTRY, _set_seed, _safe_normalize
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from datasets import load_dataset

    ds = load_dataset("ag_news", split="train")
    texts = [ds[i]["text"] for i in range(250)]
    vec = TfidfVectorizer(max_features=384, stop_words="english")
    X = _safe_normalize(vec.fit_transform(texts).toarray().astype(np.float32))
    docs, queries = X[:200], X[200:220]

    base_params = {
        "dim": 384, "hidden_dim": 384, "modes": 8, "doc_top_k": 3,
        "query_top_k": 1, "route_temperature": 0.22, "route_scale": 1.25,
        "collapse_gain": 2.2, "phase_scale": 0.78, "envelope_gain": 0.45,
        "decoy_floor": 0.24, "coherence_weight": 0.46, "public_ratio": 0.18,
        "public_mask": 0.84, "public_chunk": 6,
    }
    build = METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"]
    correct_key = "mein-geheimes-passwort-2026"

    rng = _set_seed(42)
    method_ok = build(rng, {**base_params, "secret_key": correct_key})
    doc_state = method_ok.encode_docs(docs)
    q_ok = method_ok.encode_queries(queries)
    scores_ok = method_ok.score(doc_state, q_ok)
    mag_ok = float(np.mean(np.sort(scores_ok, axis=1)[:, -10:]))

    test_keys = [
        (0, correct_key),
        (1, "mein-geheimes-passwort-2027"),
        (1, "Mein-geheimes-passwort-2026"),
        (1, "mein-geheimes-pAsswort-2026"),
        (2, "Mein-geheimes-pAsswort-2026"),
        (3, "Mein-geheimes-pAsswort-2027"),
        (4, "mein-geheimes-passwort-"),
        (5, "mein-geheimes-pass-2026"),
        (10, "anderes-passwort-hier"),
        (25, "xK9#mQ!2pL@7nZ"),
        (27, ""),
    ]

    distances, ratios = [], []
    for lev, key in test_keys:
        rng2 = _set_seed(42)
        method_w = build(rng2, {**base_params, "secret_key": key})
        q_w = method_w.encode_queries(queries)
        scores_w = method_ok.score(doc_state, q_w)
        mag_w = float(np.mean(np.sort(np.abs(scores_w), axis=1)[:, -10:]))
        distances.append(lev)
        ratios.append(mag_ok / max(1e-12, mag_w))

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.scatter(distances, ratios, s=100, c="#e74c3c", edgecolors="black",
              linewidth=0.5, zorder=5)
    ax.axhline(1.0, color="gray", ls="--", lw=1, label="No security (ratio=1)")
    ax.set_xlabel("Levenshtein Distance to Correct Key", fontsize=11)
    ax.set_ylabel("Security Ratio (correct/wrong magnitude)", fontsize=11)
    ax.set_title("SHA-256 Avalanche Effect: 1 Character → Full Collapse", fontsize=12)
    ax.set_yscale("log")
    ax.annotate("Correct key\n(ratio=1)", xy=(0, 1), fontsize=9, color="green",
               xytext=(1, 3), arrowprops=dict(arrowstyle="->", color="green"))
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    ag_news = "embedding_lab_dcpe-sap-agnews-v1.json"
    ng20 = "embedding_lab_dcpe-sap-20ng-v1.json"
    ablation = "nonlinearity_ablation.json"

    print("Figure 1: Security-Utility Tradeoff...")
    fig1 = fig_security_utility(ag_news, ng20)
    fig1.savefig(os.path.join(FIGURES_DIR, "fig1_security_utility.pdf"), dpi=300)
    fig1.savefig(os.path.join(FIGURES_DIR, "fig1_security_utility.png"), dpi=200)
    print(f"  → {FIGURES_DIR}/fig1_security_utility.pdf")

    print("Figure 2: Ablation Heatmap...")
    fig2 = fig_ablation_heatmap(ablation)
    fig2.savefig(os.path.join(FIGURES_DIR, "fig2_ablation_heatmap.pdf"), dpi=300)
    fig2.savefig(os.path.join(FIGURES_DIR, "fig2_ablation_heatmap.png"), dpi=200)
    print(f"  → {FIGURES_DIR}/fig2_ablation_heatmap.pdf")

    print("Figure 3: Phase Coherence Mechanism...")
    fig3 = fig_phase_coherence(ablation)
    fig3.savefig(os.path.join(FIGURES_DIR, "fig3_phase_coherence.pdf"), dpi=300)
    fig3.savefig(os.path.join(FIGURES_DIR, "fig3_phase_coherence.png"), dpi=200)
    print(f"  → {FIGURES_DIR}/fig3_phase_coherence.pdf")

    print("Figure 4: SHA-256 Avalanche...")
    fig4 = fig_avalanche()
    fig4.savefig(os.path.join(FIGURES_DIR, "fig4_avalanche.pdf"), dpi=300)
    fig4.savefig(os.path.join(FIGURES_DIR, "fig4_avalanche.png"), dpi=200)
    print(f"  → {FIGURES_DIR}/fig4_avalanche.pdf")

    plt.close("all")
    print("\nAlle 4 Figures generiert.")


if __name__ == "__main__":
    main()
