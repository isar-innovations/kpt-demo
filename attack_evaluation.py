"""Adversarial Evaluation: Reproduziere und quantifiziere die Red-Team-Angriffe.

Angriff 1: Score-Oracle → Gram-Matrix-Rekonstruktion
Angriff 2: Known-Plaintext → Neuronales Netz public → plaintext
Angriff 3: Statistical Leakage → pair_corr bei wachsendem N
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_lab import (
    METHOD_REGISTRY,
    _set_seed,
    _safe_normalize,
)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sentence_transformers import SentenceTransformer
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}")


CARRIER_PARAMS = {
    "modes": 8, "doc_top_k": 3, "query_top_k": 1,
    "route_temperature": 0.22, "route_scale": 1.25,
    "collapse_gain": 2.2, "phase_scale": 0.78,
    "envelope_gain": 0.45, "decoy_floor": 0.24,
    "coherence_weight": 0.46, "public_ratio": 0.18,
    "public_mask": 0.84, "public_chunk": 6,
}


def load_data(n_docs: int = 2000):
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    ds = load_dataset("ag_news", split="test")
    rng = np.random.default_rng(42)
    idx = rng.choice(len(ds), size=min(n_docs, len(ds)), replace=False)
    texts = [ds[int(i)]["text"] for i in idx]
    labels = [ds[int(i)]["label"] for i in idx]
    raw = _safe_normalize(sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32))
    return raw, labels, texts


def build_method(key: str, dim: int):
    params = {**CARRIER_PARAMS, "dim": dim, "hidden_dim": dim, "secret_key": key}
    rng = _set_seed(42)
    return METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"](rng, params)


# ── Angriff 1: Score-Oracle → Gram-Matrix-Rekonstruktion ──────────

def attack_score_oracle(raw, labels, n_queries: int = 100):
    """Angreifer hat Score-Zugang mit falschem Key.
    Kann er die Doc-Doc-Ähnlichkeitsmatrix rekonstruieren?"""
    print("\n" + "=" * 70)
    print("  ANGRIFF 1: Score-Oracle → Gram-Matrix-Rekonstruktion")
    print("=" * 70)

    dim = raw.shape[0]
    n_docs = raw.shape[0]
    method = build_method("alice-key", raw.shape[1])
    doc_state = method.encode_docs(raw)

    # Ground Truth: Plaintext-Ähnlichkeitsmatrix
    gt_sim = cosine_similarity(raw)

    # Angreifer: sendet n_queries zufällige Queries mit falschem Key
    method_attacker = build_method("attacker-key", raw.shape[1])
    rng = np.random.default_rng(99)
    query_vecs = _safe_normalize(rng.normal(size=(n_queries, raw.shape[1])).astype(np.float32))
    query_state = method_attacker.encode_queries(query_vecs)

    # Score-Matrix: (n_queries, n_docs)
    scores = method.score(doc_state, query_state)

    # Angreifer rekonstruiert Doc-Doc-Ähnlichkeit aus Score-Profilen
    # Idee: Dokumente mit ähnlichen Score-Profilen sind ähnlich
    score_profiles = scores.T  # (n_docs, n_queries) — jedes Dok hat ein Score-Profil
    reconstructed_sim = cosine_similarity(score_profiles)

    # Korrelation zwischen rekonstruierter und echter Ähnlichkeitsmatrix
    mask = np.triu(np.ones((n_docs, n_docs), dtype=bool), k=1)
    gt_flat = gt_sim[mask]
    recon_flat = reconstructed_sim[mask]
    correlation = float(np.corrcoef(gt_flat, recon_flat)[0, 1])

    # Ranking-Qualität: für jedes Dokument, wie gut stimmt das kNN-Ranking überein?
    k = 10
    knn_overlap = 0.0
    for i in range(n_docs):
        true_nbrs = set(np.argpartition(-gt_sim[i], k + 1)[:k + 1].tolist()) - {i}
        recon_nbrs = set(np.argpartition(-reconstructed_sim[i], k + 1)[:k + 1].tolist()) - {i}
        knn_overlap += len(true_nbrs & recon_nbrs) / k
    knn_overlap /= n_docs

    print(f"\n  Queries verwendet:      {n_queries}")
    print(f"  Gram-Matrix-Korrelation: {correlation:.4f}")
    print(f"  kNN@10-Überlappung:      {knn_overlap:.4f}")
    print(f"  Random Baseline kNN:     {k / n_docs:.4f}")

    # Sweep: wie viele Queries braucht der Angreifer?
    print(f"\n  Sweep: Queries → Korrelation")
    query_sweep = []
    for nq in [5, 10, 20, 50, 100, 200, 500]:
        if nq > n_queries:
            break
        sp = scores[:nq].T
        rs = cosine_similarity(sp)
        corr = float(np.corrcoef(gt_sim[mask], rs[mask])[0, 1])
        ko = 0.0
        for i in range(n_docs):
            t = set(np.argpartition(-gt_sim[i], k + 1)[:k + 1].tolist()) - {i}
            r = set(np.argpartition(-rs[i], k + 1)[:k + 1].tolist()) - {i}
            ko += len(t & r) / k
        ko /= n_docs
        query_sweep.append({"n_queries": nq, "correlation": round(corr, 4), "knn_overlap": round(ko, 4)})
        print(f"    nq={nq:>4}  corr={corr:.4f}  knn={ko:.4f}")

    return {
        "attack": "score_oracle_gram_matrix",
        "n_docs": n_docs,
        "n_queries": n_queries,
        "correlation": round(correlation, 4),
        "knn_overlap": round(knn_overlap, 4),
        "query_sweep": query_sweep,
    }


# ── Angriff 2: Known-Plaintext → Neural Net Inversion ────────────

def attack_known_plaintext(raw, labels, n_train: int = 1000):
    """Angreifer hat N (plaintext, public) Paare.
    Trainiert MLP: public → plaintext."""
    print("\n" + "=" * 70)
    print("  ANGRIFF 2: Known-Plaintext → Neural Net Inversion")
    print("=" * 70)

    method = build_method("alice-key", raw.shape[1])
    doc_state = method.encode_docs(raw)
    public = doc_state["public"]

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        public[:n_train], raw[:n_train], test_size=0.2, random_state=42
    )

    # OLS Baseline
    gram = X_train.T @ X_train + 1e-6 * np.eye(X_train.shape[1])
    ols_weights = np.linalg.solve(gram, X_train.T @ y_train).T
    ols_pred = _safe_normalize(X_test @ ols_weights.T)
    ols_cosines = np.sum(ols_pred * _safe_normalize(y_test), axis=1)
    ols_mean = float(np.mean(ols_cosines))
    ols_p95 = float(np.percentile(ols_cosines, 95))

    # MLP Attack
    mlp = MLPRegressor(
        hidden_layer_sizes=(256, 512, 384),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        random_state=42,
    )
    t0 = time.perf_counter()
    mlp.fit(X_train, y_train)
    dt = time.perf_counter() - t0

    mlp_pred = _safe_normalize(mlp.predict(X_test).astype(np.float32))
    mlp_cosines = np.sum(mlp_pred * _safe_normalize(y_test), axis=1)
    mlp_mean = float(np.mean(mlp_cosines))
    mlp_p95 = float(np.percentile(mlp_cosines, 95))

    # Kann der MLP semantische Klassen vorhersagen?
    labels_test = np.array(labels[:n_train])[- len(X_test):]
    if len(labels_test) == len(mlp_pred):
        gt_sim_test = cosine_similarity(_safe_normalize(y_test))
        pred_sim_test = cosine_similarity(mlp_pred)
        mask = np.triu(np.ones((len(X_test), len(X_test)), dtype=bool), k=1)
        sim_corr = float(np.corrcoef(gt_sim_test[mask], pred_sim_test[mask])[0, 1])
    else:
        sim_corr = 0.0

    print(f"\n  Training Paare:          {len(X_train)}")
    print(f"  Test Paare:              {len(X_test)}")
    print(f"  OLS Cosine Mean:         {ols_mean:.4f}")
    print(f"  OLS Cosine P95:          {ols_p95:.4f}")
    print(f"  MLP Cosine Mean:         {mlp_mean:.4f}")
    print(f"  MLP Cosine P95:          {mlp_p95:.4f}")
    print(f"  MLP Similarity Corr:     {sim_corr:.4f}")
    print(f"  MLP Training Time:       {dt:.1f}s")

    # Sweep: wie viele KPA-Paare braucht der Angreifer?
    print(f"\n  Sweep: Trainingsgröße → Cosine")
    kpa_sweep = []
    for nt in [50, 100, 200, 500, 800]:
        if nt > n_train:
            break
        mlp_s = MLPRegressor(hidden_layer_sizes=(256, 384), max_iter=300, early_stopping=True, random_state=42)
        mlp_s.fit(public[:nt], raw[:nt])
        holdout = _safe_normalize(mlp_s.predict(public[n_train:n_train + 200]).astype(np.float32))
        gt_holdout = _safe_normalize(raw[n_train:n_train + 200])
        cos_ho = float(np.mean(np.sum(holdout * gt_holdout, axis=1)))
        kpa_sweep.append({"n_train": nt, "cosine_mean": round(cos_ho, 4)})
        print(f"    n_train={nt:>4}  cosine={cos_ho:.4f}")

    return {
        "attack": "known_plaintext_mlp_inversion",
        "n_train": len(X_train),
        "ols_cosine_mean": round(ols_mean, 4),
        "ols_cosine_p95": round(ols_p95, 4),
        "mlp_cosine_mean": round(mlp_mean, 4),
        "mlp_cosine_p95": round(mlp_p95, 4),
        "mlp_similarity_corr": round(sim_corr, 4),
        "kpa_sweep": kpa_sweep,
    }


# ── Angriff 3: Statistical Leakage bei wachsendem N ──────────────

def attack_statistical_leakage(raw, labels):
    """Messe wie pair_corr und kNN-Overlap mit N wachsen."""
    print("\n" + "=" * 70)
    print("  ANGRIFF 3: Statistical Leakage bei wachsendem N")
    print("=" * 70)

    method = build_method("alice-key", raw.shape[1])
    doc_state = method.encode_docs(raw)
    public = doc_state["public"]

    gt_sim = cosine_similarity(raw)
    pub_sim = cosine_similarity(public)

    results = []
    for n in [50, 100, 200, 500, 1000, 1500, 2000]:
        if n > raw.shape[0]:
            break
        sub_gt = gt_sim[:n, :n]
        sub_pub = pub_sim[:n, :n]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pair_corr = float(np.corrcoef(sub_gt[mask], sub_pub[mask])[0, 1])

        # kNN overlap
        k = min(10, n - 1)
        overlap = 0.0
        for i in range(n):
            t = set(np.argpartition(-sub_gt[i], k + 1)[:k + 1].tolist()) - {i}
            p = set(np.argpartition(-sub_pub[i], k + 1)[:k + 1].tolist()) - {i}
            overlap += len(t & p) / k
        overlap /= n

        # Cluster-Erkennbarkeit: können wir Labels aus Public-kNN vorhersagen?
        label_arr = np.array(labels[:n])
        pub_graph_homophily = 0.0
        for i in range(n):
            nbrs = np.argpartition(-sub_pub[i], k + 1)[:k + 1].tolist()
            nbrs = [j for j in nbrs if j != i][:k]
            if nbrs:
                same_label = sum(1 for j in nbrs if label_arr[j] == label_arr[i]) / len(nbrs)
                pub_graph_homophily += same_label
        pub_graph_homophily /= n
        random_homophily = 1.0 / len(set(labels[:n]))

        row = {
            "n": n,
            "pair_corr": round(pair_corr, 4),
            "knn_overlap": round(overlap, 4),
            "pub_graph_homophily": round(pub_graph_homophily, 4),
            "random_homophily": round(random_homophily, 4),
            "homophily_lift": round(pub_graph_homophily / max(1e-9, random_homophily), 2),
        }
        results.append(row)
        print(f"    N={n:>5}  pair_corr={pair_corr:.4f}  knn={overlap:.4f}  homophily={pub_graph_homophily:.4f} ({row['homophily_lift']:.1f}× random)")

    return {"attack": "statistical_leakage_sweep", "results": results}


def main():
    print("=" * 70)
    print("  ADVERSARIAL EVALUATION")
    print("  Reproduziere und quantifiziere Red-Team-Angriffe")
    print("=" * 70)

    print("\nLade Daten (2000 docs, sentence-transformers)...")
    raw, labels, texts = load_data(n_docs=2000)
    print(f"  {raw.shape[0]} docs, dim={raw.shape[1]}")

    results = {}

    results["score_oracle"] = attack_score_oracle(raw, labels, n_queries=200)
    results["known_plaintext"] = attack_known_plaintext(raw, labels, n_train=1000)
    results["statistical_leakage"] = attack_statistical_leakage(raw, labels)

    out_path = "experiments/privacy_mathlab/results/attack_evaluation.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGespeichert: {out_path}")


if __name__ == "__main__":
    main()
