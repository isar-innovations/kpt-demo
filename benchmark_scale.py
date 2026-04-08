"""Production-Scale Benchmark: KPT auf 100k Docs.

Zeigt dass KPT bei production-relevanten Größen funktioniert.
Designed für GPU (sbert Encoding) + CPU (KPT/DCPE Scoring).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_lab import (
    METHOD_REGISTRY,
    _set_seed,
    _safe_normalize,
    _method_seed,
)

try:
    from sklearn.metrics.pairwise import cosine_similarity
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


def encode_texts(texts: List[str], batch_size: int = 512) -> np.ndarray:
    """Encode mit sbert auf GPU wenn verfügbar."""
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  sbert device: {device}")
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embs = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True)
    return _safe_normalize(embs.astype(np.float32))


def load_ag_news(n: int) -> List[str]:
    ds = load_dataset("ag_news", split="train")
    return ds["text"][:n]


def load_msmarco(n: int) -> List[str]:
    ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
    texts = []
    for sample in ds:
        for p in sample["passages"]["passage_text"]:
            p = p.strip()
            if len(p) > 30:
                texts.append(p)
            if len(texts) >= n:
                break
        if len(texts) >= n:
            break
    return texts[:n]


def build_kpt(key: str, dim: int):
    params = {**CARRIER_PARAMS, "dim": dim, "hidden_dim": dim, "secret_key": key}
    rng = _set_seed(42)
    return METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"](rng, params)


def build_dcpe(key: str, dim: int, noise_std: float = 0.05):
    params = {"dim": dim, "scale": 1.0, "noise_std": noise_std, "secret_key": key}
    rng = _set_seed(42)
    return METHOD_REGISTRY["dcpe_sap_v0"](rng, params)


def measure_at_scale(raw_docs, raw_queries, method, method_wrong, n_subset, k=10):
    """Messe Recall, Leakage, Latenz bei n_subset Docs."""
    docs = raw_docs[:n_subset]
    queries = raw_queries[:min(1000, n_subset // 10)]
    n_q = queries.shape[0]

    # Encode
    t0 = time.perf_counter()
    doc_state = method.encode_docs(docs)
    encode_time = time.perf_counter() - t0
    doc_ms = encode_time / n_subset * 1000

    query_state = method.encode_queries(queries)
    query_state_wrong = method_wrong.encode_queries(queries)

    # Score
    t0 = time.perf_counter()
    scores_ok = method.score(doc_state, query_state)
    score_time = time.perf_counter() - t0
    score_ms = score_time / n_q * 1000

    scores_wrong = method.score(doc_state, query_state_wrong)

    # Recall
    gt_sim = cosine_similarity(queries, docs)
    recalls = []
    for i in range(n_q):
        true_nbrs = set(np.argpartition(-gt_sim[i], k)[:k].tolist())
        pred_nbrs = set(np.argpartition(-scores_ok[i], k)[:k].tolist())
        recalls.append(len(true_nbrs & pred_nbrs) / k)
    recall = float(np.mean(recalls))

    # Leakage
    public = doc_state.get("public")
    if public is not None:
        n_sample = min(2000, n_subset)
        pub_sim = cosine_similarity(public[:n_sample])
        raw_sim = cosine_similarity(docs[:n_sample])
        mask = np.triu(np.ones((n_sample, n_sample), dtype=bool), k=1)
        pair_corr = float(np.corrcoef(raw_sim[mask], pub_sim[mask])[0, 1])
    else:
        pair_corr = 0.0

    # Score magnitude ratio
    mag_ok = float(np.mean(np.sort(scores_ok, axis=1)[:, -k:]))
    mag_wrong = float(np.mean(np.sort(np.abs(scores_wrong), axis=1)[:, -k:]))
    ratio = mag_ok / max(1e-12, mag_wrong)

    return {
        "n_docs": n_subset,
        "n_queries": n_q,
        "recall_at_10": round(recall, 4),
        "pair_corr": round(pair_corr, 4),
        "magnitude_ratio": round(ratio, 1),
        "doc_encode_ms": round(doc_ms, 2),
        "query_score_ms": round(score_ms, 2),
    }


def run_benchmark(dataset: str = "ag_news", n_docs: int = 100000, n_queries: int = 10000):
    print(f"\n{'='*70}")
    print(f"  PRODUCTION-SCALE BENCHMARK: {dataset}, {n_docs} docs")
    print(f"{'='*70}")

    # Load texts
    print(f"\nLade {dataset}...")
    t0 = time.perf_counter()
    if dataset == "ag_news":
        texts = load_ag_news(n_docs + n_queries)
    else:
        texts = load_msmarco(n_docs + n_queries)
    print(f"  {len(texts)} Texte in {time.perf_counter()-t0:.1f}s")

    # Encode mit sbert
    print(f"\nsbert-Encoding ({len(texts)} Texte)...")
    t0 = time.perf_counter()
    raw = encode_texts(texts)
    print(f"  {raw.shape[0]} Vektoren in {time.perf_counter()-t0:.1f}s, dim={raw.shape[1]}")

    raw_docs = raw[:n_docs]
    raw_queries = raw[n_docs:n_docs + n_queries]
    dim = raw.shape[1]

    # Build methods
    kpt = build_kpt("alice-production-key", dim)
    kpt_wrong = build_kpt("attacker-wrong-key", dim)
    dcpe = build_dcpe("alice-production-key", dim, noise_std=0.05)
    dcpe_wrong = build_dcpe("attacker-wrong-key", dim, noise_std=0.05)

    results = {"dataset": dataset, "total_docs": n_docs, "dim": dim, "scales": []}

    # Scale sweep
    for n in [1000, 5000, 10000, 50000, 100000]:
        if n > n_docs:
            break
        print(f"\n--- {n:,} docs ---")

        print(f"  KPT...")
        kpt_result = measure_at_scale(raw_docs, raw_queries, kpt, kpt_wrong, n)
        kpt_result["method"] = "KPT"
        print(f"    recall={kpt_result['recall_at_10']:.3f}  pair_corr={kpt_result['pair_corr']:.4f}  ratio={kpt_result['magnitude_ratio']:.0f}×  encode={kpt_result['doc_encode_ms']:.1f}ms/doc  score={kpt_result['query_score_ms']:.1f}ms/query")

        print(f"  DCPE...")
        dcpe_result = measure_at_scale(raw_docs, raw_queries, dcpe, dcpe_wrong, n)
        dcpe_result["method"] = "DCPE"
        print(f"    recall={dcpe_result['recall_at_10']:.3f}  pair_corr={dcpe_result['pair_corr']:.4f}  ratio={dcpe_result['magnitude_ratio']:.0f}×  encode={dcpe_result['doc_encode_ms']:.1f}ms/doc  score={dcpe_result['query_score_ms']:.1f}ms/query")

        results["scales"].append({"n": n, "kpt": kpt_result, "dcpe": dcpe_result})

    # Information bound: PCA on public layer
    print(f"\n--- Information Bound (PCA auf Public Layer) ---")
    doc_state = kpt.encode_docs(raw_docs[:10000])
    public = doc_state["public"]
    from sklearn.decomposition import PCA
    pca = PCA(n_components=min(public.shape[1], 50))
    pca.fit(public)
    cumvar = np.cumsum(pca.explained_variance_ratio_)
    eff_dim_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    eff_dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1
    eff_dim_99 = int(np.searchsorted(cumvar, 0.99)) + 1

    results["information_bound"] = {
        "public_dim": public.shape[1],
        "effective_dim_90pct": eff_dim_90,
        "effective_dim_95pct": eff_dim_95,
        "effective_dim_99pct": eff_dim_99,
        "plaintext_dim": dim,
        "information_ratio": round(eff_dim_95 / dim, 4),
    }
    print(f"  Public dim: {public.shape[1]}")
    print(f"  Effective dim (90% var): {eff_dim_90}")
    print(f"  Effective dim (95% var): {eff_dim_95}")
    print(f"  Effective dim (99% var): {eff_dim_99}")
    print(f"  Plaintext dim: {dim}")
    print(f"  Information ratio (95%): {eff_dim_95}/{dim} = {eff_dim_95/dim:.1%}")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["ag_news", "msmarco"], default="ag_news")
    parser.add_argument("--n-docs", type=int, default=100000)
    parser.add_argument("--n-queries", type=int, default=10000)
    parser.add_argument("--output", type=str, default="results/benchmark_scale.json")
    args = parser.parse_args()

    results = run_benchmark(args.dataset, args.n_docs, args.n_queries)

    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nGespeichert: {args.output}")


if __name__ == "__main__":
    main()
