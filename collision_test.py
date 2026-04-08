"""Kollisionstest: Selbe Dokumente mit N verschiedenen Keys.

Fragt: Kann ein falscher Key jemals einen brauchbaren Score liefern?
Testet systematisch K Keys × Q Queries und misst:
- Korrekt-Key Scores vs. Fehl-Key Scores
- Ob je ein Fehl-Key über einen Schwellwert kommt
- Cross-Key Score-Verteilung
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_lab import (
    METHOD_REGISTRY,
    _set_seed,
    _safe_normalize,
)

CARRIER_PARAMS = {
    "modes": 8, "doc_top_k": 3, "query_top_k": 1,
    "route_temperature": 0.22, "route_scale": 1.25,
    "collapse_gain": 2.2, "phase_scale": 0.78,
    "envelope_gain": 0.45, "decoy_floor": 0.24,
    "coherence_weight": 0.46, "public_ratio": 0.18,
    "public_mask": 0.84, "public_chunk": 6,
}


def build_kpt(key: str, dim: int = 384):
    params = {**CARRIER_PARAMS, "dim": dim, "hidden_dim": dim, "secret_key": key}
    rng = _set_seed(42)
    return METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"](rng, params)


def run_collision_test(
    n_keys: int = 100,
    n_docs: int = 500,
    n_queries: int = 50,
    dim: int = 384,
):
    print(f"\n{'='*70}")
    print(f"  KOLLISIONSTEST: {n_keys} Keys × {n_docs} Docs × {n_queries} Queries")
    print(f"{'='*70}\n")

    # Generiere zufällige Embeddings (simuliert sbert-Output)
    rng = np.random.default_rng(42)
    raw_docs = _safe_normalize(rng.normal(size=(n_docs, dim)).astype(np.float32))
    raw_queries = _safe_normalize(rng.normal(size=(n_queries, dim)).astype(np.float32))

    # Generiere Keys
    keys = [f"user-key-{i:04d}-{os.urandom(4).hex()}" for i in range(n_keys)]

    print(f"  Verschlüssle {n_docs} Docs mit {n_keys} verschiedenen Keys...")
    t0 = time.perf_counter()

    # Für jeden Key: Docs und Queries encodieren
    all_doc_states = []
    all_query_states = []
    kpt_instances = []

    for i, key in enumerate(keys):
        kpt = build_kpt(key, dim)
        kpt_instances.append(kpt)
        all_doc_states.append(kpt.encode_docs(raw_docs))
        all_query_states.append(kpt.encode_queries(raw_queries))
        if (i + 1) % 20 == 0:
            print(f"    {i+1}/{n_keys} Keys encodiert...")

    encode_time = time.perf_counter() - t0
    print(f"  Encoding fertig in {encode_time:.1f}s\n")

    # Score-Matrix: für jeden Key-Pair (i, j) den maximalen Score messen
    # i = Doc-Key, j = Query-Key
    print(f"  Berechne {n_keys}×{n_keys} Cross-Key Score-Matrix...")
    t0 = time.perf_counter()

    correct_scores = []
    wrong_scores = []
    worst_collision_score = 0.0
    worst_collision_keys = ("", "")
    best_correct_score = 1.0

    for doc_key_idx in range(n_keys):
        doc_state = all_doc_states[doc_key_idx]

        for query_key_idx in range(n_keys):
            query_state = all_query_states[query_key_idx]

            # Score: docs encoded with key_i, queries encoded with key_j
            scores = kpt_instances[doc_key_idx].score(doc_state, query_state)
            max_score = float(np.max(scores))
            mean_score = float(np.mean(scores))

            if doc_key_idx == query_key_idx:
                correct_scores.append(mean_score)
                best_correct_score = min(best_correct_score, mean_score)
            else:
                wrong_scores.append(max_score)
                if max_score > worst_collision_score:
                    worst_collision_score = max_score
                    worst_collision_keys = (keys[doc_key_idx][:20], keys[query_key_idx][:20])

        if (doc_key_idx + 1) % 10 == 0:
            print(f"    {doc_key_idx+1}/{n_keys} Doc-Keys getestet...")

    score_time = time.perf_counter() - t0
    print(f"  Scoring fertig in {score_time:.1f}s\n")

    # Ergebnisse
    correct_arr = np.array(correct_scores)
    wrong_arr = np.array(wrong_scores)

    print(f"  {'='*60}")
    print(f"  ERGEBNISSE")
    print(f"  {'='*60}\n")

    print(f"  Korrekt-Key Scores ({len(correct_scores)} Messungen):")
    print(f"    Mean:  {correct_arr.mean():.6f}")
    print(f"    Std:   {correct_arr.std():.6f}")
    print(f"    Min:   {correct_arr.min():.6f}")
    print(f"    Max:   {correct_arr.max():.6f}")

    print(f"\n  Fehl-Key Scores ({len(wrong_scores)} Messungen, jeweils MAX pro Key-Paar):")
    print(f"    Mean:  {wrong_arr.mean():.6f}")
    print(f"    Std:   {wrong_arr.std():.6f}")
    print(f"    Min:   {wrong_arr.min():.6f}")
    print(f"    Max:   {wrong_arr.max():.6f}")
    print(f"    P99:   {np.percentile(wrong_arr, 99):.6f}")
    print(f"    P99.9: {np.percentile(wrong_arr, 99.9):.6f}")

    gap = correct_arr.mean() - wrong_arr.max()
    gap_sigma = gap / wrong_arr.std() if wrong_arr.std() > 0 else float('inf')

    print(f"\n  Isolation:")
    print(f"    Gap (correct mean - worst collision): {gap:.6f}")
    print(f"    Gap in Sigma:                         {gap_sigma:.1f}σ")
    print(f"    Worst collision score:                {worst_collision_score:.6f}")
    print(f"    Worst collision keys:                 {worst_collision_keys}")
    print(f"    Best correct score:                   {best_correct_score:.6f}")

    # Threshold-Analyse
    thresholds = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50]
    print(f"\n  Kollisionen über Schwellwert:")
    total_wrong = len(wrong_scores)
    for t in thresholds:
        n_above = int(np.sum(wrong_arr > t))
        pct = n_above / total_wrong * 100
        print(f"    > {t:.2f}: {n_above:>6d} / {total_wrong} ({pct:.4f}%)")

    # Histogram
    print(f"\n  Score-Verteilung (Fehl-Key MAX scores):")
    bins = np.linspace(0, 0.06, 13)
    hist, edges = np.histogram(wrong_arr, bins=bins)
    for i in range(len(hist)):
        bar = "█" * (hist[i] * 60 // max(1, hist.max()))
        print(f"    [{edges[i]:.3f}-{edges[i+1]:.3f}]: {hist[i]:>5d} {bar}")

    print(f"\n  Total Cross-Key-Tests: {n_keys} × {n_keys} = {n_keys*n_keys}")
    print(f"    davon korrekt: {n_keys}")
    print(f"    davon fehl:    {n_keys * (n_keys - 1)}")

    if worst_collision_score < 0.05:
        print(f"\n  ERGEBNIS: KEINE KOLLISION. Kein Fehl-Key erreicht >0.05.")
        print(f"  Bei {n_keys*(n_keys-1)} Cross-Key-Tests ist der Worst Case {worst_collision_score:.6f}.")
    else:
        print(f"\n  WARNUNG: Worst collision bei {worst_collision_score:.6f}!")

    return {
        "n_keys": n_keys,
        "n_docs": n_docs,
        "n_queries": n_queries,
        "correct_mean": float(correct_arr.mean()),
        "correct_std": float(correct_arr.std()),
        "wrong_max": float(wrong_arr.max()),
        "wrong_mean": float(wrong_arr.mean()),
        "wrong_std": float(wrong_arr.std()),
        "gap_sigma": float(gap_sigma),
        "worst_collision_score": float(worst_collision_score),
        "worst_collision_keys": worst_collision_keys,
        "collisions_above_005": int(np.sum(wrong_arr > 0.05)),
        "collisions_above_010": int(np.sum(wrong_arr > 0.10)),
        "total_cross_key_tests": n_keys * (n_keys - 1),
    }


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="KPT Kollisionstest")
    parser.add_argument("--keys", type=int, default=100, help="Anzahl verschiedener Keys")
    parser.add_argument("--docs", type=int, default=500, help="Anzahl Dokumente")
    parser.add_argument("--queries", type=int, default=50, help="Anzahl Queries")
    parser.add_argument("--dim", type=int, default=384, help="Embedding-Dimension")
    parser.add_argument("--output", type=str, default="results/collision_test.json")
    args = parser.parse_args()

    results = run_collision_test(args.keys, args.docs, args.queries, args.dim)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Gespeichert: {args.output}")


if __name__ == "__main__":
    main()
