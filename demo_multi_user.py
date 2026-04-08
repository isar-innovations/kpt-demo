"""Demo: Multi-User Encrypted Vector Search.

Zwei User, zwei Keys, eine Datenbank. Jeder sieht nur seine eigenen Ergebnisse.
Ein Angreifer ohne Key sieht nichts.

Baut einen echten kNN-Graph auf AG News und zeigt:
1. User A sucht mit Key A → findet relevante Dokumente
2. User B sucht mit Key B → findet relevante Dokumente
3. User A sucht mit Key B → findet NICHTS (falsche Ergebnisse)
4. Angreifer nutzt Public-Layer → findet NICHTS
"""

from __future__ import annotations

import sys
import os
import json
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_lab import (
    METHOD_REGISTRY,
    _set_seed,
    _safe_normalize,
    _method_seed,
)

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}")


def build_encrypted_index(texts, labels, secret_key, carrier_params):
    """Encode Dokumente mit einem bestimmten Key → verschlüsselter Index."""
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    raw_embeddings = sbert.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    raw_embeddings = _safe_normalize(raw_embeddings)
    dim = raw_embeddings.shape[1]

    params = {**carrier_params, "dim": dim, "hidden_dim": dim, "secret_key": secret_key}
    rng = _set_seed(42)
    method = METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"](rng, params)
    doc_state = method.encode_docs(raw_embeddings)

    return {
        "texts": texts,
        "labels": labels,
        "raw_embeddings": raw_embeddings,
        "doc_state": doc_state,
        "method": method,
        "key": secret_key,
    }


def search(index, query_text, key, k=5):
    """Suche im verschlüsselten Index mit einem bestimmten Key."""
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    raw_q = sbert.encode([query_text], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    raw_q = _safe_normalize(raw_q)
    dim = raw_q.shape[1]

    params = {
        **CARRIER_PARAMS, "dim": dim, "hidden_dim": dim, "secret_key": key,
    }
    rng = _set_seed(42)
    method = METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"](rng, params)
    query_state = method.encode_queries(raw_q)

    # Score: doc_state aus dem Index (Key des Index-Besitzers), query_state mit dem Suchenden-Key
    scores = index["method"].score(index["doc_state"], query_state)
    top_k = np.argsort(-scores[0])[:k]

    results = []
    for idx in top_k:
        results.append({
            "rank": len(results) + 1,
            "score": float(scores[0][idx]),
            "label": AG_NEWS_LABELS[int(index["labels"][idx])],
            "text": index["texts"][idx][:120] + "...",
        })
    return results


def public_layer_search(index, query_text, k=5):
    """Angreifer-Suche: nur Public-Layer, kein Key."""
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    raw_q = sbert.encode([query_text], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    raw_q = _safe_normalize(raw_q)
    dim = raw_q.shape[1]

    # Public-Layer des Index
    public_docs = index["doc_state"]["public"]

    # Angreifer hat keinen Key, nutzt rohes Embedding als Query gegen Public-Layer
    params = {**CARRIER_PARAMS, "dim": dim, "hidden_dim": dim, "secret_key": "attacker-no-key"}
    rng = _set_seed(42)
    method = METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"](rng, params)
    q_state = method.encode_queries(raw_q)
    public_query = q_state["public"]

    scores = cosine_similarity(public_query, public_docs)[0]
    top_k = np.argsort(-scores)[:k]

    results = []
    for idx in top_k:
        results.append({
            "rank": len(results) + 1,
            "score": float(scores[idx]),
            "label": AG_NEWS_LABELS[int(index["labels"][idx])],
            "text": index["texts"][idx][:120] + "...",
        })
    return results


def plaintext_search(raw_embeddings, texts, labels, query_text, k=5):
    """Referenz: unverschlüsselte Suche."""
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    raw_q = sbert.encode([query_text], convert_to_numpy=True, show_progress_bar=False).astype(np.float32)
    raw_q = _safe_normalize(raw_q)
    scores = cosine_similarity(raw_q, raw_embeddings)[0]
    top_k = np.argsort(-scores)[:k]
    results = []
    for idx in top_k:
        results.append({
            "rank": len(results) + 1,
            "score": float(scores[idx]),
            "label": AG_NEWS_LABELS[int(labels[idx])],
            "text": texts[idx][:120] + "...",
        })
    return results


def print_results(title, results):
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    for r in results:
        score_bar = "█" * int(max(0, r["score"]) * 30)
        print(f"  #{r['rank']}  [{r['label']:<12}]  score={r['score']:+.4f}  {score_bar}")
        print(f"      {r['text']}")
    print()


AG_NEWS_LABELS = {0: "World", 1: "Sports", 2: "Business", 3: "Sci/Tech"}

CARRIER_PARAMS = {
    "modes": 8, "doc_top_k": 3, "query_top_k": 1,
    "route_temperature": 0.22, "route_scale": 1.25,
    "collapse_gain": 2.2, "phase_scale": 0.78,
    "envelope_gain": 0.45, "decoy_floor": 0.24,
    "coherence_weight": 0.46, "public_ratio": 0.18,
    "public_mask": 0.84, "public_chunk": 6,
}


def main():
    print("=" * 70)
    print("  DEMO: Multi-User Encrypted Vector Search")
    print("  Zwei User, zwei Keys, eine Datenbank.")
    print("=" * 70)

    # Load AG News
    print("\nLade AG News + sentence-transformers...")
    ds = load_dataset("ag_news", split="test")
    rng = np.random.default_rng(42)
    indices = rng.choice(len(ds), size=500, replace=False)
    texts = [ds[int(i)]["text"] for i in indices]
    labels = [ds[int(i)]["label"] for i in indices]

    # Zwei User, zwei Keys
    KEY_ALICE = "alice-geheimer-schluessel-2026"
    KEY_BOB = "bob-sein-passwort-xyz"

    print(f"\n  Alice's Key: {KEY_ALICE[:20]}...")
    print(f"  Bob's Key:   {KEY_BOB[:20]}...")

    # Alice baut ihren Index
    print("\nAlice verschlüsselt die Datenbank mit ihrem Key...")
    t0 = time.perf_counter()
    alice_index = build_encrypted_index(texts, labels, KEY_ALICE, CARRIER_PARAMS)
    print(f"  → {len(texts)} Dokumente verschlüsselt in {time.perf_counter()-t0:.1f}s")

    # Queries
    queries = [
        ("Stock market crash fears grow amid trade tensions", "Business"),
        ("Champions League final draws record viewership", "Sports"),
        ("New AI model breaks benchmarks in reasoning tasks", "Sci/Tech"),
    ]

    for query_text, expected_topic in queries:
        print(f"\n{'='*70}")
        print(f"  QUERY: \"{query_text}\"")
        print(f"  Erwartetes Thema: {expected_topic}")
        print(f"{'='*70}")

        # 1. Referenz: unverschlüsselt
        ref_results = plaintext_search(alice_index["raw_embeddings"], texts, labels, query_text)
        print_results("REFERENZ (unverschlüsselt)", ref_results)

        # 2. Alice sucht mit ihrem Key (sollte funktionieren)
        alice_results = search(alice_index, query_text, KEY_ALICE)
        print_results("ALICE sucht mit ALICE's Key (✓ autorisiert)", alice_results)

        # 3. Bob sucht mit seinem Key (sollte scheitern)
        bob_results = search(alice_index, query_text, KEY_BOB)
        print_results("BOB sucht mit BOB's Key (✗ falscher Key)", bob_results)

        # 4. Angreifer nutzt Public-Layer
        attacker_results = public_layer_search(alice_index, query_text)
        print_results("ANGREIFER nutzt nur Public-Layer (✗ kein Key)", attacker_results)

    # Statistik
    print(f"\n{'='*70}")
    print("  ZUSAMMENFASSUNG")
    print(f"{'='*70}")

    # Berechne durchschnittliche Scores
    sbert = SentenceTransformer("all-MiniLM-L6-v2")
    all_queries = [q[0] for q in queries]
    raw_qs = _safe_normalize(sbert.encode(all_queries, convert_to_numpy=True, show_progress_bar=False).astype(np.float32))

    # Alice-Scores (autorisiert)
    params_a = {**CARRIER_PARAMS, "dim": 384, "hidden_dim": 384, "secret_key": KEY_ALICE}
    rng_a = _set_seed(42)
    method_a = METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"](rng_a, params_a)
    q_state_a = method_a.encode_queries(raw_qs)
    scores_a = method_a.score(alice_index["doc_state"], q_state_a)
    mean_top5_a = float(np.mean([np.sort(scores_a[i])[-5:].mean() for i in range(len(queries))]))

    # Bob-Scores (unautorisiert)
    params_b = {**CARRIER_PARAMS, "dim": 384, "hidden_dim": 384, "secret_key": KEY_BOB}
    rng_b = _set_seed(42)
    method_b = METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"](rng_b, params_b)
    q_state_b = method_b.encode_queries(raw_qs)
    scores_b = method_a.score(alice_index["doc_state"], q_state_b)
    mean_top5_b = float(np.mean([np.sort(scores_b[i])[-5:].mean() for i in range(len(queries))]))

    ratio = mean_top5_a / max(1e-12, abs(mean_top5_b))

    print(f"\n  Alice (autorisiert):     Top-5 Score = {mean_top5_a:.4f}")
    print(f"  Bob (unautorisiert):     Top-5 Score = {abs(mean_top5_b):.6f}")
    print(f"  Security Ratio:          {ratio:.0f}×")
    print(f"\n  → Alice sieht relevante Ergebnisse.")
    print(f"  → Bob sieht Rauschen.")
    print(f"  → Ein Angreifer ohne Key sieht nur den destrukturierten Public-Layer.")


if __name__ == "__main__":
    main()
