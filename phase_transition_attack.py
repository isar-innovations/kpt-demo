"""Phasenübergangs-Experiment: Gradient-Descent-Angriff auf embedding-native Verschlüsselung.

Kernfrage: Gibt es einen scharfen Threshold γ_c bei dem der Angriffserfolg
instantan von ~1 auf ~0 kollabiert? Wenn ja → Fyodorov-Framework anwendbar.

Methodik:
1. Fixiere keyed_wave_superpose_embedding_v0 als Carrier
2. Swepe public_mask von 0.0 (alles sichtbar) bis 0.95 (fast nichts sichtbar)
3. Pro γ-Punkt: generiere N Dokument-Embeddings, führe drei Angriffe durch:
   a) Linearer Rekonstruktionsangriff (OLS: public → plaintext)
   b) Score-Orakel-Angriff (Angreifer kann Scores abfragen, versucht Ranking zu rekonstruieren)
   c) kNN-Überlappungsangriff (Nachbarschaftsstruktur im Public-Raum vs. Plaintext-Raum)
4. Messe Angriffserfolg als Cosine-Similarity / Ranking-Korrelation
5. Plotte die Kurve → suche den Knick
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
from scipy.stats import spearmanr

# embedding_lab liegt im selben Verzeichnis
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_lab import (
    EmbeddingStateMethod,
    _safe_normalize,
    _set_seed,
    METHOD_REGISTRY,
)

try:
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}")


def generate_tfidf_docs(n_docs: int, seed: int = 42) -> Tuple[np.ndarray, List[int]]:
    """Lade AG News und erzeuge TF-IDF Vektoren."""
    ds = load_dataset("ag_news", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_docs, len(ds)), replace=False)
    texts = [ds[int(i)]["text"] for i in indices]
    labels = [ds[int(i)]["label"] for i in indices]
    vec = TfidfVectorizer(max_features=384, stop_words="english")
    X = vec.fit_transform(texts).toarray().astype(np.float32)
    return _safe_normalize(X), labels


def linear_reconstruction_attack(
    public: np.ndarray, plaintext: np.ndarray
) -> Dict[str, float]:
    """OLS-Angriff: lerne lineare Abbildung public → plaintext."""
    gram = public.T @ public + 1e-6 * np.eye(public.shape[1], dtype=np.float32)
    weights = np.linalg.solve(gram, public.T @ plaintext).T
    predicted = _safe_normalize(public @ weights.T)
    target = _safe_normalize(plaintext)
    cosines = np.sum(predicted * target, axis=1)
    return {
        "cosine_mean": float(np.mean(cosines)),
        "cosine_p50": float(np.median(cosines)),
        "cosine_p95": float(np.percentile(cosines, 95)),
    }


def knn_overlap_attack(
    public: np.ndarray, plaintext: np.ndarray, k: int = 10
) -> float:
    """Messe kNN-Überlappung zwischen Public-Raum und Plaintext-Raum."""
    n = public.shape[0]
    k = min(k, n - 1)
    sim_plain = cosine_similarity(plaintext)
    np.fill_diagonal(sim_plain, -np.inf)
    nbr_plain = np.argpartition(-sim_plain, k, axis=1)[:, :k]

    sim_pub = cosine_similarity(public)
    np.fill_diagonal(sim_pub, -np.inf)
    nbr_pub = np.argpartition(-sim_pub, k, axis=1)[:, :k]

    overlap = 0.0
    for i in range(n):
        overlap += len(set(nbr_plain[i].tolist()) & set(nbr_pub[i].tolist())) / k
    return overlap / n


def score_recall_attack(
    method: EmbeddingStateMethod,
    plaintext: np.ndarray,
    n_queries: int = 20,
    k: int = 10,
) -> Dict[str, float]:
    """Messe Recall@k und absolute Score-Werte.

    Recall misst ob das echte relevanteste Dokument im Top-k der Scores liegt.
    Absolute Scores messen die Magnitude — bei falschem Key sollten diese kollabieren.
    """
    n_docs = plaintext.shape[0]
    k = min(k, n_docs - 1)
    doc_state = method.encode_docs(plaintext)
    rng = np.random.default_rng(123)
    query_indices = rng.choice(n_docs, size=min(n_queries, n_docs), replace=False)
    query_plain = plaintext[query_indices]
    query_state = method.encode_queries(query_plain)

    plain_sim = cosine_similarity(query_plain, plaintext)
    score_matrix = method.score(doc_state, query_state)

    recalls = []
    score_magnitudes = []
    for i, qi in enumerate(query_indices):
        true_nbrs = set(np.argpartition(-plain_sim[i], k)[:k].tolist()) - {int(qi)}
        score_nbrs = set(np.argpartition(-score_matrix[i], k)[:k].tolist()) - {int(qi)}
        if true_nbrs:
            recalls.append(len(true_nbrs & score_nbrs) / len(true_nbrs))
        score_magnitudes.append(float(np.mean(np.sort(score_matrix[i])[-k:])))

    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "score_magnitude": float(np.mean(score_magnitudes)),
    }


def score_mismatch_attack(
    correct_method: EmbeddingStateMethod,
    wrong_method: EmbeddingStateMethod,
    plaintext: np.ndarray,
    n_queries: int = 20,
    k: int = 10,
) -> Dict[str, float]:
    """Der echte Angriff: Docs sind mit dem richtigen Key kodiert,
    Angreifer kodiert Queries mit dem falschen Key.

    Das ist das realistische Szenario — die DB enthält korrekt verschlüsselte
    Embeddings, der Angreifer versucht ohne den richtigen Key zu suchen.
    """
    n_docs = plaintext.shape[0]
    k = min(k, n_docs - 1)

    # Docs mit richtigem Key (wie in der DB)
    doc_state = correct_method.encode_docs(plaintext)

    rng = np.random.default_rng(123)
    query_indices = rng.choice(n_docs, size=min(n_queries, n_docs), replace=False)
    query_plain = plaintext[query_indices]

    # Queries mit FALSCHEM Key (Angreifer)
    query_state_wrong = wrong_method.encode_queries(query_plain)

    # Ground truth: Plaintext-Cosine-Ranking
    plain_sim = cosine_similarity(query_plain, plaintext)

    # Cross-Key-Score: richtige Docs × falsche Queries
    # Da die Score-Funktion auf internen States arbeitet, nutzen wir die
    # correct_method.score — der Mismatch liegt in den States, nicht im Scorer
    score_matrix = correct_method.score(doc_state, query_state_wrong)

    recalls = []
    score_magnitudes = []
    for i, qi in enumerate(query_indices):
        true_nbrs = set(np.argpartition(-plain_sim[i], k)[:k].tolist()) - {int(qi)}
        score_nbrs = set(np.argpartition(-score_matrix[i], k)[:k].tolist()) - {int(qi)}
        if true_nbrs:
            recalls.append(len(true_nbrs & score_nbrs) / len(true_nbrs))
        score_magnitudes.append(float(np.mean(np.sort(score_matrix[i])[-k:])))

    return {
        "recall_at_k": float(np.mean(recalls)) if recalls else 0.0,
        "score_magnitude": float(np.mean(score_magnitudes)),
    }


def run_gamma_sweep(
    n_docs: int = 200,
    gamma_points: int = 20,
    seed: int = 42,
) -> List[Dict]:
    """Swepe phase_scale von 0.0 bis 2.0 und messe Angriffserfolg.

    phase_scale steuert die Stärke der Phase-Rotation — der Kern des
    Verschlüsselungsmechanismus. Bei 0.0 gibt es keine Phasen-Scrambling,
    bei hohen Werten wird die kohärente Rekonstruktion ohne richtigen Key
    unmöglich.
    """
    print(f"Generating {n_docs} TF-IDF documents from AG News...")
    plaintext, labels = generate_tfidf_docs(n_docs, seed=seed)
    dim = plaintext.shape[1]
    print(f"Dimension: {dim}, Documents: {n_docs}")

    gammas = np.linspace(0.0, 2.0, gamma_points)
    results = []

    base_params = {
        "dim": dim,
        "hidden_dim": dim,
        "modes": 8,
        "doc_top_k": 3,
        "query_top_k": 1,
        "route_temperature": 0.22,
        "route_scale": 1.25,
        "collapse_gain": 2.2,
        "envelope_gain": 0.45,
        "decoy_floor": 0.24,
        "coherence_weight": 0.46,
        "public_ratio": 0.18,
        "public_mask": 0.84,
        "public_chunk": 6,
        "secret_key": "experiment-phase-transition-42",
    }

    build_fn = METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"]

    for i, gamma in enumerate(gammas):
        t0 = time.perf_counter()
        params = {**base_params, "phase_scale": float(gamma)}

        rng = _set_seed(seed)
        method = build_fn(rng, params)

        doc_state = method.encode_docs(plaintext)
        public = doc_state["public"]

        # Angriff 1: Lineare Rekonstruktion aus Public Layer
        linear = linear_reconstruction_attack(public, plaintext)

        # Angriff 2: kNN-Überlappung Public vs Plaintext
        knn_overlap = knn_overlap_attack(public, plaintext, k=10)

        # Angriff 3: Recall + Score-Magnitude mit richtigem Key
        correct_key = score_recall_attack(method, plaintext, n_queries=20, k=10)

        # Angriff 4: Recall mit KEY-MISMATCH — Docs mit richtigem Key, Queries mit falschem
        wrong_params = {**params, "secret_key": "wrong-key-attacker-99"}
        rng_wrong = _set_seed(seed)
        method_wrong = build_fn(rng_wrong, wrong_params)
        wrong_key = score_mismatch_attack(
            method, method_wrong, plaintext, n_queries=20, k=10
        )

        dt = time.perf_counter() - t0

        row = {
            "gamma": round(float(gamma), 4),
            "linear_cosine_mean": round(linear["cosine_mean"], 6),
            "knn_overlap": round(knn_overlap, 6),
            "correct_key_recall": round(correct_key["recall_at_k"], 6),
            "correct_key_magnitude": round(correct_key["score_magnitude"], 6),
            "wrong_key_recall": round(wrong_key["recall_at_k"], 6),
            "wrong_key_magnitude": round(wrong_key["score_magnitude"], 6),
            "recall_gap": round(correct_key["recall_at_k"] - wrong_key["recall_at_k"], 6),
            "magnitude_ratio": round(
                correct_key["score_magnitude"] / max(1e-9, wrong_key["score_magnitude"]), 4
            ),
            "elapsed_s": round(dt, 2),
        }
        results.append(row)

        arrow = "█" * int(30 * (i + 1) / gamma_points)
        print(
            f"  γ={gamma:.3f}  "
            f"recall_ok={correct_key['recall_at_k']:.3f}  "
            f"recall_wrong={wrong_key['recall_at_k']:.3f}  "
            f"mag_ok={correct_key['score_magnitude']:.4f}  "
            f"mag_wrong={wrong_key['score_magnitude']:.4f}  "
            f"linear={linear['cosine_mean']:.4f}  "
            f"[{arrow:<30}] {dt:.1f}s"
        )

    return results


def detect_threshold(results: List[Dict], metric: str = "linear_cosine_mean") -> float:
    """Finde den γ-Punkt mit dem stärksten Gradienten (= schärfster Knick)."""
    values = [r[metric] for r in results]
    gammas = [r["gamma"] for r in results]
    gradients = []
    for i in range(1, len(values)):
        grad = abs(values[i] - values[i - 1]) / max(1e-6, gammas[i] - gammas[i - 1])
        gradients.append((gammas[i], grad))
    if not gradients:
        return 0.0
    return max(gradients, key=lambda x: x[1])[0]


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Phase Transition Attack Experiment")
    parser.add_argument("--n-docs", type=int, default=200)
    parser.add_argument("--gamma-points", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/privacy_mathlab/results/phase_transition_attack.json",
    )
    args = parser.parse_args()

    print("=" * 72)
    print("PHASE TRANSITION ATTACK EXPERIMENT")
    print("Frage: Gibt es einen scharfen γ_c?")
    print("=" * 72)

    results = run_gamma_sweep(
        n_docs=args.n_docs,
        gamma_points=args.gamma_points,
        seed=args.seed,
    )

    # Threshold-Detection
    gamma_c_linear = detect_threshold(results, "linear_cosine_mean")
    gamma_c_knn = detect_threshold(results, "knn_overlap")

    summary = {
        "experiment": "phase_transition_attack",
        "carrier": "keyed_wave_superpose_embedding_v0",
        "n_docs": args.n_docs,
        "gamma_points": args.gamma_points,
        "seed": args.seed,
        "detected_threshold_linear": gamma_c_linear,
        "detected_threshold_knn": gamma_c_knn,
        "results": results,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nErgebnisse gespeichert: {args.output}")

    print(f"\n{'='*72}")
    print(f"ERGEBNIS:")
    print(f"  Geschätzter γ_c (linear):  {gamma_c_linear:.3f}")
    print(f"  Geschätzter γ_c (kNN):     {gamma_c_knn:.3f}")
    print(f"{'='*72}")

    # Kurzübersicht
    print(f"\n{'γ':>6}  {'recall_ok':>9}  {'recall_wr':>9}  {'gap':>6}  {'mag_ok':>8}  {'mag_wr':>8}  {'ratio':>7}  {'linear':>7}")
    print("-" * 75)
    for r in results:
        print(
            f"{r['gamma']:6.3f}  "
            f"{r['correct_key_recall']:9.4f}  "
            f"{r['wrong_key_recall']:9.4f}  "
            f"{r['recall_gap']:6.3f}  "
            f"{r['correct_key_magnitude']:8.4f}  "
            f"{r['wrong_key_magnitude']:8.4f}  "
            f"{r['magnitude_ratio']:7.2f}  "
            f"{r['linear_cosine_mean']:7.4f}"
        )


if __name__ == "__main__":
    main()
