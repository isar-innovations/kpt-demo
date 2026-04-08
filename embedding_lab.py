"""Embedding-Lab für zustandsbasierte Retrieval-Repräsentationen.

Dieses Lab verlässt das klassische Muster "Vektor + nachgelagerte Transformation".
Stattdessen wird ein eigenes Embedding-Objekt erzeugt, bestehend aus:

- öffentlicher Beobachtung (`public`) für Leakage-/Topologie-Metriken
- autorisiertem Score-Raum (`score_*`) für keyed Retrieval

Die Idee: Ohne Schlüssel bleibt nur eine grobe, destrukturierte Beobachtung sichtbar.
Mit Schlüssel kann die Query über einen spezifischen Score-Operator kohärent auslesen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Dict, List, Tuple

import numpy as np

try:
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception as exc:  # pragma: no cover - dependency-guard
    raise SystemExit(f"Missing sklearn dependency: {exc}")

try:
    from datasets import load_dataset
except Exception:  # pragma: no cover - optional dependency
    load_dataset = None


StateMap = Dict[str, np.ndarray]

_VECTORIZER_KIND: str = "tfidf"


def _set_seed(seed: int) -> np.random.Generator:
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def _safe_normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm = np.where(norm == 0.0, 1.0, norm)
    return x / norm


def _method_seed(method_name: str, secret_key: str, dim: int, offset: int = 0) -> int:
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    ikm = secret_key.encode("utf-8")
    info = f"{method_name}|{dim}|{offset}".encode("utf-8")
    key_material = HKDF(
        algorithm=hashes.SHA512(),
        length=64,
        salt=b"kpt-v1",
        info=info,
    ).derive(ikm)
    return int.from_bytes(key_material, "big")


def _normalized_key_material(secret_key: str, key_bits: int, variant: str) -> str:
    if key_bits <= 0:
        return f"{variant}|{secret_key}"
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF
    from cryptography.hazmat.primitives import hashes
    key_byte_len = max(1, (key_bits + 7) // 8)
    material = HKDF(
        algorithm=hashes.SHA512(),
        length=key_byte_len,
        salt=b"kpt-key-material",
        info=f"{variant}|{key_bits}".encode("utf-8"),
    ).derive(secret_key.encode("utf-8"))
    return f"{variant}|{key_bits}|{material.hex()}"


def _qr_orthogonal(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    matrix = rng.normal(size=(rows, cols))
    q, _ = np.linalg.qr(matrix)
    return q.astype(np.float32)


def _mask_public_observation(
    method_name: str,
    secret_key: str,
    public: np.ndarray,
    strength: float,
    chunk: int,
) -> np.ndarray:
    y = np.array(public, dtype=np.float32, copy=True)
    if y.shape[1] == 0:
        return y
    local_rng = np.random.default_rng(
        _method_seed(f"{method_name}_public", secret_key, y.shape[1])
    )
    mix, _ = np.linalg.qr(local_rng.normal(size=(y.shape[1], y.shape[1])))
    mixed = y @ mix.T.astype(np.float32)
    drift = local_rng.normal(0.0, max(1e-4, 0.03 * strength), size=mixed.shape).astype(
        np.float32
    )
    mixed = mixed + drift
    mixed = np.tanh(mixed * (1.0 + 1.8 * strength))
    if chunk > 1:
        use_dim = (mixed.shape[1] // chunk) * chunk
        if use_dim > 0:
            head = mixed[:, :use_dim].reshape(mixed.shape[0], use_dim // chunk, chunk)
            pooled = np.mean(head, axis=2)
            tail = mixed[:, use_dim:]
            mixed = np.hstack([pooled, tail]).astype(np.float32)
    return _safe_normalize(mixed)


def _topk_soft_assign(logits: np.ndarray, top_k: int, temperature: float) -> np.ndarray:
    if logits.shape[1] == 0:
        return logits.astype(np.float32, copy=True)
    keep = max(1, min(top_k, logits.shape[1]))
    idx = np.argpartition(-logits, keep - 1, axis=1)[:, :keep]
    out = np.zeros_like(logits, dtype=np.float32)
    gathered = np.take_along_axis(logits, idx, axis=1)
    shifted = gathered - np.max(gathered, axis=1, keepdims=True)
    weights = np.exp(shifted / max(1e-4, temperature)).astype(np.float32)
    weights_sum = np.sum(weights, axis=1, keepdims=True)
    weights_sum = np.where(weights_sum == 0.0, 1.0, weights_sum)
    weights = weights / weights_sum
    np.put_along_axis(out, idx, weights, axis=1)
    return out


def _relative_uncertainty_gate(
    candidate_scores: np.ndarray, uncertainty_width: float
) -> np.ndarray:
    gate = np.ones((candidate_scores.shape[0], 1), dtype=np.float32)
    if candidate_scores.shape[1] <= 1:
        return gate
    if uncertainty_width <= 0.0:
        return gate
    sorted_scores = np.sort(candidate_scores, axis=1)
    top1 = sorted_scores[:, -1:]
    top2 = sorted_scores[:, -2:-1]
    relative_margin = (top1 - top2) / np.maximum(1e-6, top1)
    return np.clip(
        1.0 - relative_margin / max(1e-4, uncertainty_width),
        0.0,
        1.0,
    ).astype(np.float32)


@dataclass(frozen=True)
class EmbeddingStateMethod:
    method_name: str
    family: str
    params: Dict[str, float]
    encode_docs: Callable[[np.ndarray], StateMap]
    encode_queries: Callable[[np.ndarray], StateMap]
    score: Callable[[StateMap, StateMap], np.ndarray]
    aux_score: Callable[[StateMap, StateMap], np.ndarray] | None = None


def _generate_dataset(
    rng: np.random.Generator,
    n_docs: int,
    n_queries: int,
    dim: int,
    n_clusters: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    centers = _safe_normalize(rng.normal(size=(n_clusters, dim)).astype(np.float32))
    style_basis = _safe_normalize(
        rng.normal(size=(max(4, n_clusters), dim)).astype(np.float32)
    )
    background = _safe_normalize(rng.normal(size=(3, dim)).astype(np.float32))

    docs = np.zeros((n_docs, dim), dtype=np.float32)
    doc_labels = np.zeros(n_docs, dtype=np.int64)
    for idx in range(n_docs):
        primary = int(rng.integers(0, n_clusters))
        secondary = int(rng.integers(0, n_clusters - 1))
        if secondary >= primary:
            secondary += 1
        style = int(rng.integers(0, style_basis.shape[0]))
        overlap = float(rng.uniform(0.0, 0.35))
        bg_mix = float(rng.uniform(0.05, 0.15))
        sample = centers[primary] * (1.0 - overlap)
        sample = sample + centers[secondary] * overlap
        sample = sample + style_basis[style] * rng.uniform(0.08, 0.18)
        sample = sample + background[idx % background.shape[0]] * bg_mix
        sample = sample + 0.20 * rng.normal(size=dim).astype(np.float32)
        docs[idx] = sample
        doc_labels[idx] = primary

    queries = np.zeros((n_queries, dim), dtype=np.float32)
    query_labels = np.zeros(n_queries, dtype=np.int64)
    for idx in range(n_queries):
        primary = int(rng.integers(0, n_clusters))
        secondary = int(rng.integers(0, n_clusters - 1))
        if secondary >= primary:
            secondary += 1
        style = int(rng.integers(0, style_basis.shape[0]))
        overlap = float(rng.uniform(0.0, 0.28))
        bg_mix = float(rng.uniform(0.02, 0.10))
        sample = centers[primary] * (1.0 - overlap)
        sample = sample + centers[secondary] * overlap
        sample = sample + style_basis[style] * rng.uniform(0.06, 0.14)
        sample = sample + background[idx % background.shape[0]] * bg_mix
        sample = sample + 0.16 * rng.normal(size=dim).astype(np.float32)
        queries[idx] = sample
        query_labels[idx] = primary

    return _safe_normalize(docs), _safe_normalize(queries), doc_labels, query_labels


def _topic_lexicon(topic_idx: int) -> List[str]:
    bases = [
        ["privacy", "cipher", "key", "secret", "vector", "search", "recall", "noise"],
        [
            "biology",
            "cell",
            "protein",
            "enzyme",
            "motif",
            "evolution",
            "signal",
            "amino",
        ],
        ["physics", "quantum", "wave", "phase", "field", "spin", "loop", "holonomy"],
        ["finance", "market", "risk", "alpha", "credit", "yield", "liquidity", "hedge"],
        [
            "law",
            "contract",
            "court",
            "statute",
            "evidence",
            "appeal",
            "liability",
            "claim",
        ],
        [
            "medicine",
            "patient",
            "dose",
            "therapy",
            "diagnosis",
            "symptom",
            "immune",
            "trial",
        ],
        [
            "climate",
            "carbon",
            "ocean",
            "storm",
            "emission",
            "forest",
            "warming",
            "energy",
        ],
        [
            "robotics",
            "sensor",
            "control",
            "motion",
            "planner",
            "actuator",
            "trajectory",
            "mapping",
        ],
        [
            "linguistics",
            "syntax",
            "morphology",
            "phonology",
            "semantics",
            "corpus",
            "utterance",
            "grammar",
        ],
        [
            "security",
            "exploit",
            "sandbox",
            "malware",
            "attack",
            "hardening",
            "audit",
            "forensics",
        ],
        [
            "graphics",
            "shader",
            "render",
            "texture",
            "mesh",
            "contour",
            "lighting",
            "surface",
        ],
        [
            "education",
            "teacher",
            "lesson",
            "curriculum",
            "student",
            "assessment",
            "feedback",
            "practice",
        ],
        [
            "music",
            "harmony",
            "rhythm",
            "melody",
            "chord",
            "cadence",
            "voice",
            "counterpoint",
        ],
        [
            "history",
            "archive",
            "empire",
            "treaty",
            "chronicle",
            "artifact",
            "dynasty",
            "conflict",
        ],
        [
            "agriculture",
            "soil",
            "crop",
            "yield",
            "harvest",
            "irrigation",
            "seed",
            "fertility",
        ],
        [
            "networks",
            "router",
            "latency",
            "packet",
            "throughput",
            "switch",
            "protocol",
            "topology",
        ],
    ]
    return bases[topic_idx % len(bases)]


def _style_lexicon(style_idx: int) -> List[str]:
    styles = [
        ["overview", "summary", "brief", "note", "report", "outline"],
        ["deep", "detailed", "analysis", "mechanism", "theory", "evidence"],
        ["practical", "deployment", "system", "operator", "workflow", "integration"],
        ["debate", "question", "critique", "hypothesis", "counterpoint", "discussion"],
        ["metric", "benchmark", "evaluation", "precision", "recall", "variance"],
        ["creative", "metaphor", "analogy", "crossdomain", "novel", "transfer"],
    ]
    return styles[style_idx % len(styles)]


def _common_lexicon() -> List[str]:
    return [
        "context",
        "pattern",
        "signal",
        "structure",
        "system",
        "model",
        "topic",
        "cluster",
        "query",
        "document",
        "memory",
        "reasoning",
        "result",
        "method",
        "measure",
        "latent",
    ]


def _build_text_sample(
    rng: np.random.Generator,
    primary_words: List[str],
    secondary_words: List[str],
    style_words: List[str],
    common_words: List[str],
    max_len: int,
    overlap: float,
) -> str:
    parts: List[str] = []
    primary_count = max(4, int(round(max_len * (0.45 - 0.15 * overlap))))
    secondary_count = max(1, int(round(max_len * overlap * 0.35)))
    style_count = max(2, int(round(max_len * 0.18)))
    common_count = max(3, max_len - primary_count - secondary_count - style_count)
    for _ in range(primary_count):
        parts.append(primary_words[int(rng.integers(0, len(primary_words)))])
    for _ in range(secondary_count):
        parts.append(secondary_words[int(rng.integers(0, len(secondary_words)))])
    for _ in range(style_count):
        parts.append(style_words[int(rng.integers(0, len(style_words)))])
    for _ in range(common_count):
        parts.append(common_words[int(rng.integers(0, len(common_words)))])
    rng.shuffle(parts)
    if len(parts) >= 4:
        bigram = f"{parts[0]}_{parts[1]}"
        trigram = f"{parts[2]}_{parts[3]}"
        parts.append(bigram)
        parts.append(trigram)
    return " ".join(parts)


def _vectorize_text_corpus(
    docs_text: List[str],
    queries_text: List[str],
    dim: int,
    seed: int,
    vectorizer_kind: str = "tfidf",
) -> Tuple[np.ndarray, np.ndarray]:
    if vectorizer_kind == "sbert":
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("all-MiniLM-L6-v2")
        docs_emb = model.encode(
            docs_text, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        queries_emb = model.encode(
            queries_text, convert_to_numpy=True, show_progress_bar=False
        ).astype(np.float32)
        return _safe_normalize(docs_emb), _safe_normalize(queries_emb)

    vectorizer = TfidfVectorizer(
        max_features=max(4096, dim * 8), ngram_range=(1, 2), min_df=1
    )
    joined = docs_text + queries_text
    tfidf = vectorizer.fit_transform(joined)
    effective_dim = min(dim, max(8, tfidf.shape[1] - 1))
    if tfidf.shape[1] <= effective_dim:
        dense = tfidf.toarray().astype(np.float32)
        docs = dense[: len(docs_text)]
        queries = dense[len(docs_text) :]
        return _safe_normalize(docs), _safe_normalize(queries)
    svd = TruncatedSVD(n_components=effective_dim, random_state=seed)
    dense = svd.fit_transform(tfidf).astype(np.float32)
    docs = dense[: len(docs_text)]
    queries = dense[len(docs_text) :]
    return _safe_normalize(docs), _safe_normalize(queries)


def _generate_synthetic_text_dataset(
    rng: np.random.Generator,
    n_docs: int,
    n_queries: int,
    dim: int,
    n_clusters: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    common_words = _common_lexicon()
    docs_text: List[str] = []
    doc_labels = np.zeros(n_docs, dtype=np.int64)
    for idx in range(n_docs):
        primary = int(rng.integers(0, n_clusters))
        secondary = int(rng.integers(0, n_clusters - 1))
        if secondary >= primary:
            secondary += 1
        style = int(rng.integers(0, 6))
        overlap = float(rng.uniform(0.0, 0.35))
        docs_text.append(
            _build_text_sample(
                rng=rng,
                primary_words=_topic_lexicon(primary),
                secondary_words=_topic_lexicon(secondary),
                style_words=_style_lexicon(style),
                common_words=common_words,
                max_len=int(rng.integers(18, 34)),
                overlap=overlap,
            )
        )
        doc_labels[idx] = primary

    queries_text: List[str] = []
    query_labels = np.zeros(n_queries, dtype=np.int64)
    for idx in range(n_queries):
        primary = int(rng.integers(0, n_clusters))
        secondary = int(rng.integers(0, n_clusters - 1))
        if secondary >= primary:
            secondary += 1
        style = int(rng.integers(0, 6))
        overlap = float(rng.uniform(0.0, 0.22))
        queries_text.append(
            _build_text_sample(
                rng=rng,
                primary_words=_topic_lexicon(primary),
                secondary_words=_topic_lexicon(secondary),
                style_words=_style_lexicon(style),
                common_words=common_words,
                max_len=int(rng.integers(10, 18)),
                overlap=overlap,
            )
        )
        query_labels[idx] = primary

    docs, queries = _vectorize_text_corpus(
        docs_text, queries_text, dim=dim, seed=seed, vectorizer_kind=_VECTORIZER_KIND
    )
    return docs, queries, doc_labels, query_labels


@lru_cache(maxsize=8)
def _load_hf_split(dataset_name: str, config_name: str, split_name: str):
    if load_dataset is None:
        raise SystemExit(
            "Missing datasets dependency: install `datasets` for --dataset-kind hf_text"
        )
    if config_name:
        return load_dataset(dataset_name, config_name, split=split_name)
    return load_dataset(dataset_name, split=split_name)


def _guess_text_field(sample: Dict[str, object]) -> str:
    for candidate in (
        "text",
        "content",
        "sentence",
        "document",
        "body",
        "article",
        "description",
    ):
        value = sample.get(candidate)
        if isinstance(value, str) and value.strip():
            return candidate
    best_field = ""
    best_score = -1
    for key, value in sample.items():
        if not isinstance(value, str):
            continue
        score = len(value.split())
        if score <= best_score:
            continue
        best_field = key
        best_score = score
    if best_field:
        return best_field
    raise ValueError("Could not infer text field from HuggingFace sample")


def _guess_label_field(sample: Dict[str, object]) -> str:
    for candidate in ("label", "labels", "topic", "category", "class"):
        value = sample.get(candidate)
        if isinstance(value, (int, np.integer)) and not isinstance(value, bool):
            return candidate
    for key, value in sample.items():
        if not isinstance(value, (int, np.integer)):
            continue
        if isinstance(value, bool):
            continue
        return key
    raise ValueError("Could not infer label field from HuggingFace sample")


def _normalize_split_name(split_name: str, pool_size: int) -> str:
    if pool_size <= 0:
        return split_name
    if "[" in split_name:
        return split_name
    return f"{split_name}[:{pool_size}]"


def _extract_hf_text_and_labels(
    dataset_name: str,
    config_name: str,
    split_name: str,
    text_field: str,
    label_field: str,
) -> Tuple[List[str], np.ndarray]:
    split = _load_hf_split(dataset_name, config_name, split_name)
    if len(split) == 0:
        raise ValueError(f"HuggingFace split is empty: {dataset_name} {split_name}")
    sample = split[0]
    resolved_text_field = text_field or _guess_text_field(sample)
    resolved_label_field = label_field or _guess_label_field(sample)
    texts: List[str] = []
    labels: List[int] = []
    for row in split:
        text_value = row.get(resolved_text_field)
        label_value = row.get(resolved_label_field)
        if not isinstance(text_value, str):
            continue
        if not isinstance(label_value, (int, np.integer)):
            continue
        if isinstance(label_value, bool):
            continue
        text = " ".join(text_value.split())
        if not text:
            continue
        texts.append(text)
        labels.append(int(label_value))
    if len(texts) == 0:
        raise ValueError(
            f"No usable text rows found in {dataset_name} {split_name} with fields "
            f"{resolved_text_field}/{resolved_label_field}"
        )
    return texts, np.asarray(labels, dtype=np.int64)


def _balanced_sample_indices(
    labels: np.ndarray, target_size: int, rng: np.random.Generator
) -> np.ndarray:
    if target_size > labels.shape[0]:
        raise ValueError(
            f"Requested {target_size} rows from only {labels.shape[0]} available labels"
        )
    by_label: Dict[int, List[int]] = {}
    for idx, label in enumerate(labels.tolist()):
        by_label.setdefault(int(label), []).append(idx)
    label_order = list(by_label.keys())
    rng.shuffle(label_order)
    for label in label_order:
        rng.shuffle(by_label[label])
    cursors = {label: 0 for label in label_order}
    picked: List[int] = []
    while len(picked) < target_size:
        progressed = False
        for label in label_order:
            cursor = cursors[label]
            if cursor >= len(by_label[label]):
                continue
            picked.append(by_label[label][cursor])
            cursors[label] = cursor + 1
            progressed = True
            if len(picked) >= target_size:
                break
        if progressed:
            continue
        raise ValueError(
            "Balanced sampling exhausted all classes before target size was reached"
        )
    picked_array = np.asarray(picked, dtype=np.int64)
    rng.shuffle(picked_array)
    return picked_array


def _generate_hf_text_dataset(
    rng: np.random.Generator,
    n_docs: int,
    n_queries: int,
    dim: int,
    seed: int,
    dataset_name: str,
    config_name: str,
    doc_split: str,
    query_split: str,
    text_field: str,
    label_field: str,
    doc_pool: int,
    query_pool: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    resolved_doc_split = _normalize_split_name(doc_split, doc_pool)
    resolved_query_split = _normalize_split_name(query_split, query_pool)
    doc_texts_all, doc_labels_all = _extract_hf_text_and_labels(
        dataset_name=dataset_name,
        config_name=config_name,
        split_name=resolved_doc_split,
        text_field=text_field,
        label_field=label_field,
    )
    query_texts_all, query_labels_all = _extract_hf_text_and_labels(
        dataset_name=dataset_name,
        config_name=config_name,
        split_name=resolved_query_split,
        text_field=text_field,
        label_field=label_field,
    )
    doc_idx = _balanced_sample_indices(doc_labels_all, n_docs, rng)
    query_idx = _balanced_sample_indices(query_labels_all, n_queries, rng)
    docs_text = [doc_texts_all[int(idx)] for idx in doc_idx]
    queries_text = [query_texts_all[int(idx)] for idx in query_idx]
    docs, queries = _vectorize_text_corpus(
        docs_text, queries_text, dim=dim, seed=seed, vectorizer_kind=_VECTORIZER_KIND
    )
    return docs, queries, doc_labels_all[doc_idx], query_labels_all[query_idx]


def _generate_msmarco_dataset(
    rng: np.random.Generator,
    n_docs: int,
    n_queries: int,
    dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ds = load_dataset("ms_marco", "v1.1", split="train", streaming=True)
    passages, queries_text = [], []
    passage_labels, query_labels = [], []
    label_counter = 0
    for i, sample in enumerate(ds):
        if len(passages) >= n_docs + n_queries:
            break
        q = sample["query"].strip()
        for j, p in enumerate(sample["passages"]["passage_text"]):
            p = p.strip()
            if len(p) < 30:
                continue
            if len(passages) < n_docs:
                passages.append(p)
                passage_labels.append(label_counter)
            elif len(queries_text) < n_queries:
                queries_text.append(q)
                query_labels.append(label_counter)
                break
        label_counter += 1
    docs_vec, queries_vec = _vectorize_text_corpus(
        passages[:n_docs],
        queries_text[:n_queries],
        dim=dim,
        seed=seed,
        vectorizer_kind=_VECTORIZER_KIND,
    )
    return (
        docs_vec,
        queries_vec,
        np.array(passage_labels[:n_docs]),
        np.array(query_labels[:n_queries]),
    )


def _generate_20newsgroups_dataset(
    rng: np.random.Generator,
    n_docs: int,
    n_queries: int,
    dim: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from sklearn.datasets import fetch_20newsgroups

    train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"))
    test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"))
    train_labels = np.array(train.target)
    test_labels = np.array(test.target)
    doc_idx = _balanced_sample_indices(train_labels, n_docs, rng)
    query_idx = _balanced_sample_indices(test_labels, n_queries, rng)
    docs_text = [train.data[int(i)] for i in doc_idx]
    queries_text = [test.data[int(i)] for i in query_idx]
    docs, queries = _vectorize_text_corpus(
        docs_text, queries_text, dim=dim, seed=seed, vectorizer_kind=_VECTORIZER_KIND
    )
    return docs, queries, train_labels[doc_idx], test_labels[query_idx]


def _build_dataset(
    kind: str,
    rng: np.random.Generator,
    n_docs: int,
    n_queries: int,
    dim: int,
    n_clusters: int,
    seed: int,
    hf_dataset: str,
    hf_config: str,
    hf_doc_split: str,
    hf_query_split: str,
    hf_text_field: str,
    hf_label_field: str,
    hf_doc_pool: int,
    hf_query_pool: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if kind == "latent":
        return _generate_dataset(rng, n_docs, n_queries, dim, n_clusters)
    if kind == "synthetic_text":
        return _generate_synthetic_text_dataset(
            rng, n_docs, n_queries, dim, n_clusters, seed
        )
    if kind == "hf_text":
        return _generate_hf_text_dataset(
            rng=rng,
            n_docs=n_docs,
            n_queries=n_queries,
            dim=dim,
            seed=seed,
            dataset_name=hf_dataset,
            config_name=hf_config,
            doc_split=hf_doc_split,
            query_split=hf_query_split,
            text_field=hf_text_field,
            label_field=hf_label_field,
            doc_pool=hf_doc_pool,
            query_pool=hf_query_pool,
        )
    if kind == "sklearn_20ng":
        return _generate_20newsgroups_dataset(rng, n_docs, n_queries, dim, seed)
    if kind == "msmarco":
        return _generate_msmarco_dataset(rng, n_docs, n_queries, dim, seed)
    raise ValueError(f"Unknown dataset kind: {kind}")


def _ground_truth_topk(docs: np.ndarray, queries: np.ndarray, k: int) -> np.ndarray:
    sim = cosine_similarity(queries, docs)
    return np.argpartition(-sim, k - 1, axis=1)[:, :k]


def _recall_at_k(scores: np.ndarray, gt_topk: np.ndarray, k: int) -> float:
    if scores.shape[0] == 0:
        return 0.0
    pred = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    total = 0.0
    for idx in range(gt_topk.shape[0]):
        total += len(
            set(gt_topk[idx].tolist()).intersection(set(pred[idx].tolist()))
        ) / float(k)
    return float(total / gt_topk.shape[0])


def _per_query_recall_at_k(
    scores: np.ndarray, gt_topk: np.ndarray, k: int
) -> np.ndarray:
    if scores.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if scores.shape[1] == 0:
        return np.zeros((scores.shape[0],), dtype=np.float32)
    keep = min(max(1, k), scores.shape[1], gt_topk.shape[1])
    pred = np.argpartition(-scores, keep - 1, axis=1)[:, :keep]
    hits = np.zeros((gt_topk.shape[0],), dtype=np.float32)
    for idx in range(gt_topk.shape[0]):
        hits[idx] = len(
            set(gt_topk[idx, :keep].tolist()).intersection(set(pred[idx].tolist()))
        ) / float(keep)
    return hits


def _top1_margin(scores: np.ndarray) -> np.ndarray:
    if scores.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    if scores.shape[1] <= 1:
        return np.ones((scores.shape[0],), dtype=np.float32)
    sorted_scores = np.sort(scores, axis=1)
    top1 = sorted_scores[:, -1]
    top2 = sorted_scores[:, -2]
    denom = np.maximum(np.abs(top1), 1e-6)
    return np.clip((top1 - top2) / denom, 0.0, None).astype(np.float32)


def _query_label_purity(gt_topk: np.ndarray, doc_labels: np.ndarray) -> np.ndarray:
    if gt_topk.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    purity = np.zeros((gt_topk.shape[0],), dtype=np.float32)
    for idx in range(gt_topk.shape[0]):
        labels = doc_labels[gt_topk[idx]]
        _, counts = np.unique(labels, return_counts=True)
        purity[idx] = float(np.max(counts) / max(1, labels.shape[0]))
    return purity


def _observer_query_ambiguity(
    doc_state: StateMap,
    query_state: StateMap,
    carrier_scores: np.ndarray | None,
    params: Dict[str, float],
) -> Dict[str, object]:
    doc_observer = doc_state.get("observer_channels")
    query_observer = query_state.get("observer_channels")
    if carrier_scores is None:
        return {"available": 0.0}
    if doc_observer is None or query_observer is None:
        return {"available": 0.0}
    if doc_observer.shape[0] != carrier_scores.shape[1]:
        return {"available": 0.0}
    if query_observer.shape[0] != carrier_scores.shape[0]:
        return {"available": 0.0}
    channel_count = int(query_observer.shape[1])
    if channel_count <= 0:
        return {"available": 0.0}
    support_width = min(
        max(1, int(params.get("support_width", 8))), carrier_scores.shape[1]
    )
    if support_width <= 0:
        return {"available": 0.0}
    support_temperature = float(params.get("support_temperature", 0.14))
    support_idx = np.argpartition(-carrier_scores, support_width - 1, axis=1)[
        :, :support_width
    ]
    support_scores = np.take_along_axis(carrier_scores, support_idx, axis=1)
    shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
    support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
        np.float32
    )
    weight_sum = np.sum(support_weights, axis=1, keepdims=True)
    weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
    support_weights = support_weights / weight_sum
    support_observer = doc_observer[support_idx]
    channel_response = np.maximum(
        np.sum(query_observer[:, None, :, :] * support_observer, axis=3), 0.0
    )
    channel_weight = np.sum(
        channel_response * support_weights[:, :, None], axis=1, dtype=np.float32
    )
    observer_gate = query_state.get("observer_gate")
    if observer_gate is None:
        observer_gate = doc_state.get("observer_gate")
    if observer_gate is not None:
        observer_gate = np.ravel(observer_gate).astype(np.float32, copy=False)
    if observer_gate is not None and observer_gate.shape[0] == channel_count:
        channel_weight = channel_weight * observer_gate[None, :]
    if channel_count == 1:
        ambiguity = np.zeros((channel_weight.shape[0],), dtype=np.float32)
        effective_modes = np.ones((channel_weight.shape[0],), dtype=np.float32)
        return {
            "available": 1.0,
            "ambiguity_mean": float(np.mean(ambiguity)),
            "ambiguity_std": float(np.std(ambiguity)),
            "effective_modes_mean": float(np.mean(effective_modes)),
            "per_query_ambiguity": ambiguity,
        }
    channel_total = np.sum(channel_weight, axis=1, keepdims=True)
    normalized = np.full(
        channel_weight.shape,
        1.0 / float(channel_count),
        dtype=np.float32,
    )
    positive = channel_total[:, 0] > 1e-8
    normalized[positive] = channel_weight[positive] / np.maximum(
        channel_total[positive], 1e-8
    )
    raw_entropy = -np.sum(
        normalized * np.log(np.maximum(normalized, 1e-8)), axis=1, dtype=np.float32
    )
    ambiguity = (raw_entropy / math.log(float(channel_count))).astype(np.float32)
    effective_modes = np.exp(raw_entropy).astype(np.float32)
    return {
        "available": 1.0,
        "ambiguity_mean": float(np.mean(ambiguity)),
        "ambiguity_std": float(np.std(ambiguity)),
        "effective_modes_mean": float(np.mean(effective_modes)),
        "per_query_ambiguity": ambiguity,
    }


def _carrier_similarity_scores(
    doc_state: StateMap, query_state: StateMap
) -> np.ndarray | None:
    required = {"wave_real", "wave_imag"}
    if not required.issubset(doc_state.keys()):
        return None
    if not required.issubset(query_state.keys()):
        return None
    query_real = query_state["wave_real"]
    doc_real = doc_state["wave_real"]
    query_imag = query_state["wave_imag"]
    doc_imag = doc_state["wave_imag"]
    if query_real.ndim != doc_real.ndim:
        return None
    if query_imag.ndim != doc_imag.ndim:
        return None
    if query_real.ndim == 2:
        overlap = query_real @ doc_real.T
        overlap = overlap + query_imag @ doc_imag.T
        return (overlap**2).astype(np.float32)
    if query_real.ndim < 2:
        return None
    if query_real.shape[1:] != doc_real.shape[1:]:
        return None
    if query_imag.shape[1:] != doc_imag.shape[1:]:
        return None
    component_dim = int(np.prod(query_real.shape[1:-1]))
    query_real_flat = query_real.reshape(
        query_real.shape[0], component_dim, query_real.shape[-1]
    )
    doc_real_flat = doc_real.reshape(
        doc_real.shape[0], component_dim, doc_real.shape[-1]
    )
    query_imag_flat = query_imag.reshape(
        query_imag.shape[0], component_dim, query_imag.shape[-1]
    )
    doc_imag_flat = doc_imag.reshape(
        doc_imag.shape[0], component_dim, doc_imag.shape[-1]
    )
    component_overlap = np.einsum(
        "qcd,ncd->qnc", query_real_flat, doc_real_flat, dtype=np.float32
    )
    component_overlap = component_overlap + np.einsum(
        "qcd,ncd->qnc", query_imag_flat, doc_imag_flat, dtype=np.float32
    )
    return np.mean(component_overlap**2, axis=2, dtype=np.float32)


def _semantic_query_focus(
    doc_state: StateMap,
    query_state: StateMap,
    carrier_scores: np.ndarray | None,
    method_params: Dict[str, float],
) -> Dict[str, object]:
    default = {
        "available": 0.0,
        "focus_mean": 0.0,
        "focus_std": 0.0,
        "effective_codes_mean": 0.0,
        "per_query_focus": None,
    }
    semantic_codes = doc_state.get("semantic_codes")
    semantic_centers = doc_state.get("semantic_centers")
    semantic_config = doc_state.get("semantic_code_config")
    if semantic_codes is None:
        return default
    if semantic_centers is None:
        return default
    if semantic_config is None:
        return default
    if carrier_scores is None:
        return default
    if "energy" not in query_state:
        return default
    if semantic_codes.shape[0] == 0:
        return default
    if semantic_centers.shape[0] == 0:
        return default
    if semantic_codes.shape[0] != carrier_scores.shape[1]:
        return default

    code_top_k = min(
        semantic_centers.shape[0],
        max(1, int(round(float(semantic_config[0])))),
    )
    code_temperature = max(1e-4, float(semantic_config[1]))
    query_codes = _topk_soft_assign(
        query_state["energy"] @ semantic_centers.T,
        code_top_k,
        code_temperature,
    )
    raw_entropy = -np.sum(
        query_codes * np.log(np.maximum(query_codes, 1e-8)),
        axis=1,
        dtype=np.float32,
    )
    effective_codes = np.exp(raw_entropy).astype(np.float32)
    support_width = min(
        max(1, int(method_params.get("support_width", 8))),
        carrier_scores.shape[1],
    )
    support_temperature = max(
        1e-4, float(method_params.get("support_temperature", 0.14))
    )
    support_idx = np.argpartition(-carrier_scores, support_width - 1, axis=1)[
        :, :support_width
    ]
    support_scores = np.take_along_axis(carrier_scores, support_idx, axis=1)
    shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
    support_weights = np.exp(shifted / support_temperature).astype(np.float32)
    weight_sum = np.sum(support_weights, axis=1, keepdims=True)
    weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
    support_weights = support_weights / weight_sum
    support_codes = semantic_codes[support_idx]
    support_context = np.sum(
        support_codes * support_weights[:, :, None],
        axis=1,
        dtype=np.float32,
    )
    support_context = _safe_normalize(support_context)
    query_peak = np.max(query_codes, axis=1)
    family_alignment = np.sum(
        query_codes * support_context,
        axis=1,
        dtype=np.float32,
    )
    focus = np.sqrt(
        np.clip(query_peak * np.maximum(family_alignment, 0.0), 0.0, 1.0)
    ).astype(np.float32)
    return {
        "available": 1.0,
        "focus_mean": float(np.mean(focus)),
        "focus_std": float(np.std(focus)),
        "effective_codes_mean": float(np.mean(effective_codes)),
        "per_query_focus": focus,
    }


def _cohort_summary_row(
    mask: np.ndarray,
    recall_hits: np.ndarray,
    wrong_recall_hits: np.ndarray,
    carrier_hits: np.ndarray | None,
    observer_ambiguity: np.ndarray | None,
    semantic_focus: np.ndarray | None,
) -> Dict[str, float]:
    query_count = int(mask.shape[0])
    cohort_count = int(np.sum(mask))
    if cohort_count <= 0:
        return {
            "query_count": 0.0,
            "query_share": 0.0,
            "recall_at_k": 0.0,
            "wrong_key_recall_at_k": 0.0,
            "wrong_key_recall_ratio": 0.0,
            "carrier_recall_at_k": 0.0,
            "delta_vs_carrier": 0.0,
            "observer_ambiguity_mean": 0.0,
            "semantic_focus_mean": 0.0,
        }
    cohort_recall = float(np.mean(recall_hits[mask]))
    cohort_wrong_recall = float(np.mean(wrong_recall_hits[mask]))
    carrier_recall = cohort_recall
    if carrier_hits is not None:
        carrier_recall = float(np.mean(carrier_hits[mask]))
    observer_ambiguity_mean = 0.0
    if observer_ambiguity is not None:
        observer_ambiguity_mean = float(np.mean(observer_ambiguity[mask]))
    semantic_focus_mean = 0.0
    if semantic_focus is not None:
        semantic_focus_mean = float(np.mean(semantic_focus[mask]))
    return {
        "query_count": float(cohort_count),
        "query_share": float(cohort_count / max(1, query_count)),
        "recall_at_k": cohort_recall,
        "wrong_key_recall_at_k": cohort_wrong_recall,
        "wrong_key_recall_ratio": float(cohort_wrong_recall / max(cohort_recall, 1e-8)),
        "carrier_recall_at_k": carrier_recall,
        "delta_vs_carrier": float(cohort_recall - carrier_recall),
        "observer_ambiguity_mean": observer_ambiguity_mean,
        "semantic_focus_mean": semantic_focus_mean,
    }


def _query_cohort_metrics(
    scores: np.ndarray,
    wrong_scores: np.ndarray,
    gt_topk: np.ndarray,
    doc_labels: np.ndarray,
    top_k: int,
    carrier_scores: np.ndarray | None,
    observer_ambiguity: np.ndarray | None,
    semantic_focus: np.ndarray | None,
) -> Dict[str, object]:
    recall_hits = _per_query_recall_at_k(scores, gt_topk, top_k)
    wrong_recall_hits = _per_query_recall_at_k(wrong_scores, gt_topk, top_k)
    carrier_hits = None
    margin_scores = scores
    margin_source = "final"
    if carrier_scores is not None:
        carrier_hits = _per_query_recall_at_k(carrier_scores, gt_topk, top_k)
        margin_scores = carrier_scores
        margin_source = "carrier"
    margins = _top1_margin(margin_scores)
    purity = _query_label_purity(gt_topk, doc_labels)
    if recall_hits.shape[0] == 0:
        return {
            "available": 0.0,
            "margin_source": margin_source,
            "margin_threshold": 0.0,
            "purity_threshold": 0.0,
            "observer_ambiguity_available": 0.0,
            "observer_ambiguity_threshold": 0.0,
            "semantic_focus_available": 0.0,
            "semantic_focus_threshold": 0.0,
            "cohorts": {},
        }
    margin_threshold = float(np.median(margins))
    purity_threshold = float(np.median(purity))
    hard_mask = margins <= margin_threshold
    easy_mask = np.logical_not(hard_mask)
    diffuse_mask = purity <= purity_threshold
    coherent_mask = np.logical_not(diffuse_mask)
    cohorts = {
        "all": np.ones((recall_hits.shape[0],), dtype=bool),
        "hard": hard_mask,
        "easy": easy_mask,
        "diffuse": diffuse_mask,
        "coherent": coherent_mask,
        "hard_coherent": np.logical_and(hard_mask, coherent_mask),
    }
    observer_ambiguity_available = 0.0
    observer_ambiguity_threshold = 0.0
    if (
        observer_ambiguity is not None
        and observer_ambiguity.shape[0] == recall_hits.shape[0]
    ):
        observer_ambiguity_available = 1.0
        observer_ambiguity_threshold = float(np.median(observer_ambiguity))
        focused_mask = observer_ambiguity <= observer_ambiguity_threshold
        cohorts["observer_focused"] = focused_mask
        cohorts["observer_ambiguous"] = np.logical_not(focused_mask)
    semantic_focus_available = 0.0
    semantic_focus_threshold = 0.0
    if semantic_focus is not None and semantic_focus.shape[0] == recall_hits.shape[0]:
        semantic_focus_available = 1.0
        semantic_focus_threshold = float(np.median(semantic_focus))
        focused_mask = semantic_focus >= semantic_focus_threshold
        cohorts["semantic_focused"] = focused_mask
        cohorts["semantic_unfocused"] = np.logical_not(focused_mask)
    return {
        "available": 1.0,
        "margin_source": margin_source,
        "margin_threshold": margin_threshold,
        "purity_threshold": purity_threshold,
        "observer_ambiguity_available": observer_ambiguity_available,
        "observer_ambiguity_threshold": observer_ambiguity_threshold,
        "semantic_focus_available": semantic_focus_available,
        "semantic_focus_threshold": semantic_focus_threshold,
        "cohorts": {
            name: _cohort_summary_row(
                mask,
                recall_hits,
                wrong_recall_hits,
                carrier_hits,
                observer_ambiguity,
                semantic_focus,
            )
            for name, mask in cohorts.items()
        },
    }


def _community_alignment(
    docs_plain: np.ndarray, docs_public: np.ndarray, n_clusters: int
) -> Dict[str, float]:
    n_clusters = min(n_clusters, docs_plain.shape[0], docs_public.shape[0], 20)
    if n_clusters <= 1:
        return {"ari": 1.0, "nmi": 1.0}
    base = KMeans(n_clusters=n_clusters, n_init=8, random_state=0, max_iter=200).fit(
        docs_plain
    )
    transformed = KMeans(
        n_clusters=n_clusters, n_init=8, random_state=0, max_iter=200
    ).fit(docs_public)
    return {
        "ari": float(adjusted_rand_score(base.labels_, transformed.labels_)),
        "nmi": float(normalized_mutual_info_score(base.labels_, transformed.labels_)),
    }


def _label_alignment(
    features: np.ndarray, labels: np.ndarray, n_clusters: int
) -> Dict[str, float]:
    if features.shape[0] == 0:
        return {"ari": 0.0, "nmi": 0.0}
    unique_labels = np.unique(labels)
    if unique_labels.shape[0] <= 1:
        return {"ari": 1.0, "nmi": 1.0}
    cluster_count = min(n_clusters, features.shape[0], unique_labels.shape[0], 20)
    if cluster_count <= 1:
        return {"ari": 1.0, "nmi": 1.0}
    normalized = _safe_normalize(features.astype(np.float32, copy=False))
    clustered = KMeans(
        n_clusters=cluster_count, n_init=8, random_state=0, max_iter=200
    ).fit(normalized)
    return {
        "ari": float(adjusted_rand_score(labels, clustered.labels_)),
        "nmi": float(normalized_mutual_info_score(labels, clustered.labels_)),
    }


def _slice_state_rows(state: StateMap, row_idx: np.ndarray) -> StateMap:
    row_count = state["public"].shape[0]
    sliced: StateMap = {}
    for name, value in state.items():
        if value.shape[0] != row_count:
            sliced[name] = value
            continue
        sliced[name] = value[row_idx]
    return sliced


def _score_profile_alignment(
    method: EmbeddingStateMethod,
    doc_state: StateMap,
    doc_labels: np.ndarray,
    n_clusters: int,
    rng: np.random.Generator,
) -> Dict[str, float]:
    doc_count = doc_labels.shape[0]
    if doc_count == 0:
        return {"ari": 0.0, "nmi": 0.0}
    unique_labels = np.unique(doc_labels)
    if unique_labels.shape[0] <= 1:
        return {"ari": 1.0, "nmi": 1.0}
    anchor_count = min(doc_count, max(16, n_clusters * 16), 96)
    anchor_idx = np.sort(rng.choice(doc_count, size=anchor_count, replace=False))
    anchor_state = _slice_state_rows(doc_state, anchor_idx)
    score_profiles = method.score(doc_state, anchor_state).T.astype(np.float32)
    return _label_alignment(score_profiles, doc_labels, n_clusters)


def _topk_neighbor_index(
    scores: np.ndarray, k: int, exclude_self: bool = False
) -> np.ndarray:
    if scores.shape[0] == 0 or scores.shape[1] == 0:
        return np.zeros((scores.shape[0], 0), dtype=np.int32)
    work = np.array(scores, dtype=np.float32, copy=True)
    candidate_count = work.shape[1]
    if exclude_self and work.shape[0] == work.shape[1]:
        np.fill_diagonal(work, -np.inf)
        candidate_count = max(0, candidate_count - 1)
    if candidate_count <= 0:
        return np.zeros((work.shape[0], 0), dtype=np.int32)
    keep = min(max(1, k), candidate_count)
    return np.argpartition(-work, keep - 1, axis=1)[:, :keep]


def _label_homophily_from_scores(
    scores: np.ndarray, labels: np.ndarray, k: int, exclude_self: bool = False
) -> float:
    neighbors = _topk_neighbor_index(scores, k, exclude_self=exclude_self)
    if neighbors.shape[1] == 0:
        return 0.0
    same_label = labels[neighbors] == labels[:, None]
    return float(np.mean(same_label))


def _neighbor_overlap(
    base_scores: np.ndarray,
    probe_scores: np.ndarray,
    k: int,
    exclude_self: bool = False,
) -> float:
    base_neighbors = _topk_neighbor_index(base_scores, k, exclude_self=exclude_self)
    probe_neighbors = _topk_neighbor_index(probe_scores, k, exclude_self=exclude_self)
    if base_neighbors.shape[1] == 0:
        return 0.0
    total = 0.0
    for idx in range(base_neighbors.shape[0]):
        total += len(
            set(base_neighbors[idx].tolist()).intersection(
                set(probe_neighbors[idx].tolist())
            )
        ) / float(base_neighbors.shape[1])
    return float(total / base_neighbors.shape[0])


def _relational_graph_metrics(
    method: EmbeddingStateMethod,
    doc_state: StateMap,
    doc_query_state: StateMap,
    wrong_doc_query_state: StateMap,
    public_docs: np.ndarray,
    doc_labels: np.ndarray,
    n_clusters: int,
) -> Dict[str, float]:
    if doc_labels.shape[0] == 0:
        return {
            "public_label_homophily": 0.0,
            "score_label_homophily": 0.0,
            "wrong_key_label_homophily": 0.0,
            "wrong_key_homophily_ratio": 0.0,
            "wrong_key_edge_overlap": 0.0,
        }
    graph_k = min(max(2, n_clusters), max(1, doc_labels.shape[0] - 1), 12)
    public_scores = cosine_similarity(public_docs, public_docs)
    score_graph = method.score(doc_state, doc_query_state)
    wrong_score_graph = method.score(doc_state, wrong_doc_query_state)
    public_homophily = _label_homophily_from_scores(
        public_scores, doc_labels, graph_k, exclude_self=True
    )
    score_homophily = _label_homophily_from_scores(
        score_graph, doc_labels, graph_k, exclude_self=True
    )
    wrong_homophily = _label_homophily_from_scores(
        wrong_score_graph, doc_labels, graph_k, exclude_self=True
    )
    return {
        "public_label_homophily": public_homophily,
        "score_label_homophily": score_homophily,
        "wrong_key_label_homophily": wrong_homophily,
        "wrong_key_homophily_ratio": float(
            wrong_homophily / max(score_homophily, 1e-8)
        ),
        "wrong_key_edge_overlap": _neighbor_overlap(
            score_graph, wrong_score_graph, graph_k, exclude_self=True
        ),
    }


def _semantic_code_observable_metrics(
    doc_state: StateMap,
    doc_query_state: StateMap,
    wrong_doc_query_state: StateMap,
    doc_labels: np.ndarray,
    n_clusters: int,
) -> Dict[str, float]:
    default = {
        "available": 0.0,
        "label_ari": 0.0,
        "label_nmi": 0.0,
        "graph_homophily": 0.0,
        "wrong_key_graph_homophily": 0.0,
        "wrong_key_graph_ratio": 0.0,
        "wrong_key_graph_overlap": 0.0,
    }
    semantic_codes = doc_state.get("semantic_codes")
    semantic_centers = doc_state.get("semantic_centers")
    semantic_config = doc_state.get("semantic_code_config")
    if semantic_codes is None:
        return default
    if semantic_centers is None:
        return default
    if semantic_config is None:
        return default
    if "energy" not in doc_query_state:
        return default
    if "energy" not in wrong_doc_query_state:
        return default
    if semantic_codes.shape[0] == 0:
        return default
    if semantic_centers.shape[0] == 0:
        return default
    if semantic_codes.shape[0] != doc_labels.shape[0]:
        return default

    code_top_k = min(
        semantic_centers.shape[0],
        max(1, int(round(float(semantic_config[0])))),
    )
    code_temperature = max(1e-4, float(semantic_config[1]))
    observed_codes = semantic_codes.astype(np.float32, copy=False)
    observable_alignment = _label_alignment(observed_codes, doc_labels, n_clusters)
    observable_scores = (
        _topk_soft_assign(
            doc_query_state["energy"] @ semantic_centers.T,
            code_top_k,
            code_temperature,
        )
        @ observed_codes.T
    )
    wrong_observable_scores = (
        _topk_soft_assign(
            wrong_doc_query_state["energy"] @ semantic_centers.T,
            code_top_k,
            code_temperature,
        )
        @ observed_codes.T
    )
    graph_k = min(max(2, n_clusters), max(1, doc_labels.shape[0] - 1), 12)
    observable_homophily = _label_homophily_from_scores(
        observable_scores,
        doc_labels,
        graph_k,
        exclude_self=True,
    )
    wrong_observable_homophily = _label_homophily_from_scores(
        wrong_observable_scores,
        doc_labels,
        graph_k,
        exclude_self=True,
    )
    return {
        "available": 1.0,
        "label_ari": float(observable_alignment["ari"]),
        "label_nmi": float(observable_alignment["nmi"]),
        "graph_homophily": observable_homophily,
        "wrong_key_graph_homophily": wrong_observable_homophily,
        "wrong_key_graph_ratio": float(
            wrong_observable_homophily / max(observable_homophily, 1e-8)
        ),
        "wrong_key_graph_overlap": _neighbor_overlap(
            observable_scores,
            wrong_observable_scores,
            graph_k,
            exclude_self=True,
        ),
    }


def _aux_operator_scores(
    doc_state: StateMap,
    query_state: StateMap,
    aux_score: Callable[[StateMap, StateMap], np.ndarray] | None = None,
) -> np.ndarray | None:
    if aux_score is not None:
        return aux_score(doc_state, query_state)
    doc_profile = doc_state.get("aux_operator_profile")
    query_profile = query_state.get("aux_operator_profile")
    if doc_profile is None:
        return None
    if query_profile is None:
        return None
    if doc_profile.shape[1] == 0:
        return None
    if query_profile.shape[1] != doc_profile.shape[1]:
        return None
    return query_profile @ doc_profile.T


def _aux_operator_metrics(
    doc_state: StateMap,
    query_state: StateMap,
    doc_query_state: StateMap,
    wrong_doc_query_state: StateMap,
    carrier_scores: np.ndarray | None,
    doc_labels: np.ndarray,
    n_clusters: int,
    top_k: int,
    aux_score: Callable[[StateMap, StateMap], np.ndarray] | None = None,
) -> Dict[str, float]:
    default = {
        "available": 0.0,
        "label_ari": 0.0,
        "graph_homophily": 0.0,
        "wrong_key_graph_homophily": 0.0,
        "wrong_key_graph_ratio": 0.0,
        "wrong_key_graph_overlap": 0.0,
        "carrier_graph_overlap": 0.0,
        "query_rank_corr": 0.0,
        "query_topk_overlap": 0.0,
    }
    aux_profiles = doc_state.get("aux_operator_profile")
    if aux_profiles is None:
        return default
    if aux_profiles.shape[0] != doc_labels.shape[0]:
        return default

    aux_query_scores = _aux_operator_scores(doc_state, query_state, aux_score)
    aux_doc_scores = _aux_operator_scores(doc_state, doc_query_state, aux_score)
    wrong_aux_doc_scores = _aux_operator_scores(
        doc_state, wrong_doc_query_state, aux_score
    )
    if aux_query_scores is None:
        return default
    if aux_doc_scores is None:
        return default
    if wrong_aux_doc_scores is None:
        return default

    graph_k = min(max(2, n_clusters), max(1, doc_labels.shape[0] - 1), 12)
    aux_alignment = _label_alignment(
        aux_profiles.astype(np.float32, copy=False), doc_labels, n_clusters
    )
    aux_graph_homophily = _label_homophily_from_scores(
        aux_doc_scores,
        doc_labels,
        graph_k,
        exclude_self=True,
    )
    wrong_aux_graph_homophily = _label_homophily_from_scores(
        wrong_aux_doc_scores,
        doc_labels,
        graph_k,
        exclude_self=True,
    )

    query_rank_corr = 0.0
    query_topk_overlap = 0.0
    carrier_graph_overlap = 0.0
    if carrier_scores is not None:
        carrier_centered = carrier_scores - np.mean(
            carrier_scores, axis=1, keepdims=True
        )
        aux_centered = aux_query_scores - np.mean(
            aux_query_scores, axis=1, keepdims=True
        )
        numer = np.sum(carrier_centered * aux_centered, axis=1, dtype=np.float32)
        denom = np.linalg.norm(carrier_centered, axis=1) * np.linalg.norm(
            aux_centered, axis=1
        )
        valid = denom > 1e-8
        if np.any(valid):
            query_rank_corr = float(np.mean(numer[valid] / denom[valid]))
        query_topk_overlap = _neighbor_overlap(
            carrier_scores,
            aux_query_scores,
            min(top_k, carrier_scores.shape[1]),
            exclude_self=False,
        )
        carrier_doc_scores = _carrier_similarity_scores(doc_state, doc_query_state)
        if carrier_doc_scores is not None:
            carrier_graph_overlap = _neighbor_overlap(
                carrier_doc_scores,
                aux_doc_scores,
                graph_k,
                exclude_self=True,
            )

    return {
        "available": 1.0,
        "label_ari": float(aux_alignment["ari"]),
        "graph_homophily": aux_graph_homophily,
        "wrong_key_graph_homophily": wrong_aux_graph_homophily,
        "wrong_key_graph_ratio": float(
            wrong_aux_graph_homophily / max(aux_graph_homophily, 1e-8)
        ),
        "wrong_key_graph_overlap": _neighbor_overlap(
            aux_doc_scores,
            wrong_aux_doc_scores,
            graph_k,
            exclude_self=True,
        ),
        "carrier_graph_overlap": carrier_graph_overlap,
        "query_rank_corr": query_rank_corr,
        "query_topk_overlap": query_topk_overlap,
    }


def _topology_leakage(
    orig: np.ndarray, transformed: np.ndarray, rng: np.random.Generator
) -> Dict[str, float]:
    sample_n = min(1500, orig.shape[0])
    sample_idx = rng.choice(orig.shape[0], size=sample_n, replace=False)
    s_orig = orig[sample_idx]
    s_trans = transformed[sample_idx]
    centroid_o = np.mean(s_orig, axis=0)
    centroid_t = np.mean(s_trans, axis=0)
    dist_o = np.linalg.norm(s_orig - centroid_o, axis=1)
    dist_t = np.linalg.norm(s_trans - centroid_t, axis=1)
    centroid_corr = float(np.corrcoef(dist_o, dist_t)[0, 1])
    pair_n = min(2000, sample_n * (sample_n - 1) // 2)
    idx1 = rng.integers(0, sample_n, size=pair_n)
    idx2 = rng.integers(0, sample_n, size=pair_n)
    keep = idx1 != idx2
    pair_orig = np.sum(s_orig[idx1[keep]] * s_orig[idx2[keep]], axis=1)
    pair_trans = np.sum(s_trans[idx1[keep]] * s_trans[idx2[keep]], axis=1)
    return {
        "centroid_distance_corr": centroid_corr,
        "pairwise_similarity_corr": float(np.corrcoef(pair_orig, pair_trans)[0, 1]),
    }


def _attack_proxies(orig: np.ndarray, transformed: np.ndarray) -> Dict[str, float]:
    n = min(orig.shape[0], 2500)
    x = transformed[:n]
    y = orig[:n]
    gram = x.T @ x + 1e-6 * np.eye(x.shape[1], dtype=np.float32)
    weights = np.linalg.solve(gram, x.T @ y).T
    y_hat = _safe_normalize(x @ weights.T)
    y_norm = _safe_normalize(y)
    cosine = np.sum(y_hat * y_norm, axis=1)
    mse = np.mean((y_hat - y_norm) ** 2)
    return {
        "linear_recon_cosine_mean": float(np.mean(cosine)),
        "linear_recon_cosine_p95": float(np.percentile(cosine, 95)),
        "linear_recon_mse": float(mse),
    }


def _nn_overlap(orig: np.ndarray, transformed: np.ndarray) -> float:
    n = orig.shape[0]
    k = min(15, max(2, n - 1))
    sim_orig = cosine_similarity(orig, orig)
    np.fill_diagonal(sim_orig, -np.inf)
    nbr_orig = np.argpartition(-sim_orig, k, axis=1)[:, :k]
    sim_trans = cosine_similarity(transformed, transformed)
    np.fill_diagonal(sim_trans, -np.inf)
    nbr_trans = np.argpartition(-sim_trans, k, axis=1)[:, :k]
    total = 0.0
    for idx in range(n):
        total += len(
            set(nbr_orig[idx].tolist()).intersection(set(nbr_trans[idx].tolist()))
        ) / float(k)
    return float(total / n)


def _runtime_ms(fn: Callable[..., object], x: object, reps: int = 3) -> float:
    if reps <= 0:
        return 0.0
    start = time.perf_counter()
    for _ in range(reps):
        _ = fn(x)
    elapsed = (time.perf_counter() - start) / reps
    return float(elapsed * 1000.0)


def _entropy_of_components(x: np.ndarray) -> float:
    normed = np.square(x)
    normed = normed / np.sum(normed, axis=0, keepdims=True)
    entropy = -np.sum(normed * np.log2(np.where(normed > 0, normed, 1.0)))
    return float(entropy)


def _baseline_vector_build(
    _: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))

    def _encode(x: np.ndarray) -> StateMap:
        public = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        return {"public": public, "score": public}

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return cosine_similarity(query_state["score"], doc_state["score"])

    return EmbeddingStateMethod(
        method_name="baseline_vector_state",
        family="baseline_state",
        params={"dim": dim},
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _complex_wavepacket_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    bands = int(params.get("bands", 4))
    phase_scale = float(params.get("phase_scale", 0.9))
    envelope_gain = float(params.get("envelope_gain", 0.55))
    public_ratio = float(params.get("public_ratio", 0.5))
    public_mask = float(params.get("public_mask", 0.42))
    public_chunk = int(params.get("public_chunk", 2))
    public_dim = max(8, int(dim * public_ratio))
    public_band_chunk = max(4, public_dim // max(1, bands))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("complex_wavepacket_v0", secret_key, dim)
    )
    real_proj = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(bands)])
    imag_proj = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(bands)])
    public_mix = np.stack(
        [
            local_rng.normal(size=(dim, public_band_chunk)).astype(np.float32)
            for _ in range(bands)
        ]
    )
    band_weights = np.abs(local_rng.normal(size=(bands,)).astype(np.float32))
    band_weights = band_weights / np.sum(band_weights)
    band_phase = local_rng.uniform(0.0, 2.0 * math.pi, size=(bands, dim)).astype(
        np.float32
    )

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = np.einsum("nd,bdk->nbk", y, real_proj)
        imag = np.einsum("nd,bdk->nbk", y, imag_proj)
        phase = phase_scale * real + band_phase[None, :, :]
        amplitude = 1.0 + envelope_gain * np.tanh(imag)
        wave_real = amplitude * np.cos(phase)
        wave_imag = amplitude * np.sin(phase)
        wave_norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=2, keepdims=True))
        wave_norm = np.where(wave_norm == 0.0, 1.0, wave_norm)
        wave_real = wave_real / wave_norm
        wave_imag = wave_imag / wave_norm
        amp_hidden = np.sqrt(np.maximum(amplitude, 1e-6)).astype(np.float32)
        amp_norm = np.linalg.norm(amp_hidden, axis=2, keepdims=True)
        amp_norm = np.where(amp_norm == 0.0, 1.0, amp_norm)
        amp_hidden = amp_hidden / amp_norm
        public_chunks = []
        for band in range(bands):
            energy = np.sqrt(wave_real[:, band] ** 2 + wave_imag[:, band] ** 2)
            chunk = energy @ public_mix[band]
            public_chunks.append(chunk.astype(np.float32))
        public = _safe_normalize(np.hstack(public_chunks))
        public = _mask_public_observation(
            "complex_wavepacket_v0", secret_key, public, public_mask, public_chunk
        )
        return {
            "public": public,
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "amp_hidden": amp_hidden,
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        sim = np.zeros(
            (query_state["public"].shape[0], doc_state["public"].shape[0]),
            dtype=np.float32,
        )
        for band in range(bands):
            q_real = query_state["wave_real"][:, band]
            d_real = doc_state["wave_real"][:, band]
            q_imag = query_state["wave_imag"][:, band]
            d_imag = doc_state["wave_imag"][:, band]
            coherence = q_real @ d_real.T
            coherence = coherence + q_imag @ d_imag.T
            amp_q = query_state["amp_hidden"][:, band]
            amp_d = doc_state["amp_hidden"][:, band]
            amplitude = amp_q @ amp_d.T
            band_sim = 0.65 * coherence + 0.35 * amplitude
            sim = sim + band_weights[band] * band_sim
        return sim / float(max(1, bands))

    return EmbeddingStateMethod(
        method_name="complex_wavepacket_v0",
        family="state_wavepacket",
        params={
            "dim": dim,
            "bands": bands,
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _complex_wavepacket_v1_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    bands = int(params.get("bands", 5))
    phase_scale = float(params.get("phase_scale", 0.80))
    envelope_gain = float(params.get("envelope_gain", 0.45))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.82))
    public_chunk = int(params.get("public_chunk", 6))
    codebook_size = int(params.get("codebook_size", 48))
    code_top_k = int(params.get("code_top_k", 6))
    code_weight = float(params.get("code_weight", 0.06))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("complex_wavepacket_v1", secret_key, dim)
    )
    public_dim = max(8, int(dim * public_ratio))
    public_band_chunk = max(3, public_dim // max(1, bands))
    real_proj = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(bands)])
    imag_proj = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(bands)])
    public_mix = np.stack(
        [
            local_rng.normal(size=(dim, public_band_chunk)).astype(np.float32)
            for _ in range(bands)
        ]
    )
    codebook = _safe_normalize(
        local_rng.normal(size=(codebook_size, dim)).astype(np.float32)
    )
    band_weights = np.abs(local_rng.normal(size=(bands,)).astype(np.float32))
    band_weights = band_weights / np.sum(band_weights)
    band_phase = local_rng.uniform(0.0, 2.0 * math.pi, size=(bands, dim)).astype(
        np.float32
    )

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = np.einsum("nd,bdk->nbk", y, real_proj)
        imag = np.einsum("nd,bdk->nbk", y, imag_proj)
        phase = phase_scale * real + band_phase[None, :, :]
        amplitude = 1.0 + envelope_gain * np.tanh(imag)
        wave_real = amplitude * np.cos(phase)
        wave_imag = amplitude * np.sin(phase)
        wave_norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=2, keepdims=True))
        wave_norm = np.where(wave_norm == 0.0, 1.0, wave_norm)
        wave_real = wave_real / wave_norm
        wave_imag = wave_imag / wave_norm
        amp_hidden = np.sqrt(np.maximum(amplitude, 1e-6)).astype(np.float32)
        amp_norm = np.linalg.norm(amp_hidden, axis=2, keepdims=True)
        amp_norm = np.where(amp_norm == 0.0, 1.0, amp_norm)
        amp_hidden = amp_hidden / amp_norm
        code_logits = y @ codebook.T
        code_hidden = _topk_soft_assign(code_logits, code_top_k, 0.40)
        public_chunks = []
        for band in range(bands):
            energy = np.sqrt(wave_real[:, band] ** 2 + wave_imag[:, band] ** 2)
            public_features = energy @ public_mix[band]
            public_chunks.append(public_features.astype(np.float32))
        public = _safe_normalize(np.hstack(public_chunks))
        public = _mask_public_observation(
            "complex_wavepacket_v1", secret_key, public, public_mask, public_chunk
        )
        return {
            "public": public,
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "amp_hidden": amp_hidden,
            "code_hidden": code_hidden.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        sim = np.zeros(
            (query_state["public"].shape[0], doc_state["public"].shape[0]),
            dtype=np.float32,
        )
        for band in range(bands):
            q_real = query_state["wave_real"][:, band]
            d_real = doc_state["wave_real"][:, band]
            q_imag = query_state["wave_imag"][:, band]
            d_imag = doc_state["wave_imag"][:, band]
            coherence = q_real @ d_real.T
            coherence = coherence + q_imag @ d_imag.T
            amp_q = query_state["amp_hidden"][:, band]
            amp_d = doc_state["amp_hidden"][:, band]
            amplitude_score = amp_q @ amp_d.T
            band_score = 0.68 * coherence + 0.32 * amplitude_score
            sim = sim + band_weights[band] * band_score
        code_score = query_state["code_hidden"] @ doc_state["code_hidden"].T
        return (1.0 - code_weight) * (
            sim / float(max(1, bands))
        ) + code_weight * code_score

    return EmbeddingStateMethod(
        method_name="complex_wavepacket_v1",
        family="state_wavepacket",
        params={
            "dim": dim,
            "bands": bands,
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
            "codebook_size": float(codebook_size),
            "code_top_k": float(code_top_k),
            "code_weight": code_weight,
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _complex_observer_subgraph_transport_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "complex_wavepacket_v0"
    dim = int(params.get("dim", 0))
    bands = int(params.get("bands", 4))
    phase_scale = float(params.get("phase_scale", 0.9))
    envelope_gain = float(params.get("envelope_gain", 0.55))
    graph_k = max(4, int(params.get("graph_k", 8)))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    relation_slots = max(8, int(params.get("relation_slots", 16)))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 2))))
    relation_temperature = float(params.get("relation_temperature", 0.08))
    transport_mix = float(params.get("transport_mix", 0.68))
    support_width = max(graph_k, int(params.get("support_width", 8)))
    support_temperature = float(params.get("support_temperature", 0.10))
    observer_channels = max(2, int(params.get("observer_channels", 4)))
    observer_dim = max(12, int(params.get("observer_dim", 24)))
    observer_gain = float(params.get("observer_gain", 0.45))
    observer_temperature = float(params.get("observer_temperature", 0.16))
    observer_edge_floor = float(params.get("observer_edge_floor", 0.18))
    profile_mix = float(params.get("profile_mix", 0.35))
    public_ratio = float(params.get("public_ratio", 0.5))
    public_mask = float(params.get("public_mask", 0.42))
    public_chunk = int(params.get("public_chunk", 2))
    public_dim = max(8, int(dim * public_ratio))
    public_band_chunk = max(4, public_dim // max(1, bands))
    relation_dim = max(1, bands) * dim
    secret_key = str(params.get("secret_key", ""))

    carrier_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    local_rng = np.random.default_rng(
        _method_seed(
            "complex_observer_subgraph_transport_observable_v0", secret_key, dim
        )
    )
    real_proj = np.stack([_qr_orthogonal(carrier_rng, dim, dim) for _ in range(bands)])
    imag_proj = np.stack([_qr_orthogonal(carrier_rng, dim, dim) for _ in range(bands)])
    public_mix = np.stack(
        [
            carrier_rng.normal(size=(dim, public_band_chunk)).astype(np.float32)
            for _ in range(bands)
        ]
    )
    band_weights = np.abs(carrier_rng.normal(size=(bands,)).astype(np.float32))
    band_weights = band_weights / np.maximum(1e-6, np.sum(band_weights))
    band_phase = carrier_rng.uniform(0.0, 2.0 * math.pi, size=(bands, dim)).astype(
        np.float32
    )
    relation_basis = _safe_normalize(
        local_rng.normal(size=(relation_slots, relation_dim)).astype(np.float32)
    )
    observer_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, observer_dim) for _ in range(observer_channels)]
    )
    observer_mod_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, observer_dim) for _ in range(observer_channels)]
    )
    observer_bias = local_rng.uniform(
        -math.pi, math.pi, size=(observer_channels, observer_dim)
    ).astype(np.float32)
    observer_gate = np.abs(
        local_rng.normal(size=(observer_channels,)).astype(np.float32)
    )
    observer_gate = observer_gate / np.maximum(1e-6, np.sum(observer_gate))

    def _encode_wave(
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = np.einsum("nd,bdk->nbk", y, real_proj)
        imag = np.einsum("nd,bdk->nbk", y, imag_proj)
        phase = phase_scale * real + band_phase[None, :, :]
        amplitude = 1.0 + envelope_gain * np.tanh(imag)
        wave_real = amplitude * np.cos(phase)
        wave_imag = amplitude * np.sin(phase)
        wave_norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=2, keepdims=True))
        wave_norm = np.where(wave_norm == 0.0, 1.0, wave_norm)
        wave_real = wave_real / wave_norm
        wave_imag = wave_imag / wave_norm
        amp_hidden = np.sqrt(np.maximum(amplitude, 1e-6)).astype(np.float32)
        amp_norm = np.linalg.norm(amp_hidden, axis=2, keepdims=True)
        amp_norm = np.where(amp_norm == 0.0, 1.0, amp_norm)
        amp_hidden = amp_hidden / amp_norm
        public_chunks = []
        for band in range(bands):
            band_energy = np.sqrt(wave_real[:, band] ** 2 + wave_imag[:, band] ** 2)
            chunk = band_energy @ public_mix[band]
            public_chunks.append(chunk.astype(np.float32))
        public = _safe_normalize(np.hstack(public_chunks))
        public = _mask_public_observation(
            base_method_name, secret_key, public, public_mask, public_chunk
        )
        band_energy = (wave_real**2 + wave_imag**2).astype(np.float32)
        energy = (np.sqrt(band_weights)[None, :, None] * band_energy).reshape(
            wave_real.shape[0], relation_dim
        )
        energy = _safe_normalize(energy.astype(np.float32, copy=False))
        return (
            public,
            wave_real.astype(np.float32),
            wave_imag.astype(np.float32),
            amp_hidden.astype(np.float32),
            energy,
        )

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        amp_hidden_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
        amp_hidden_b: np.ndarray,
    ) -> np.ndarray:
        sim = np.zeros((wave_real_a.shape[0], wave_real_b.shape[0]), dtype=np.float32)
        for band in range(bands):
            coherence = wave_real_a[:, band] @ wave_real_b[:, band].T
            coherence = coherence + wave_imag_a[:, band] @ wave_imag_b[:, band].T
            amplitude = amp_hidden_a[:, band] @ amp_hidden_b[:, band].T
            band_sim = 0.65 * coherence + 0.35 * amplitude
            sim = sim + band_weights[band] * band_sim
        return sim / float(max(1, bands))

    def _relation_profile(energy: np.ndarray) -> np.ndarray:
        relation_logits = energy @ relation_basis.T
        return _topk_soft_assign(relation_logits, relation_top_k, relation_temperature)

    def _observer_hidden(x: np.ndarray) -> np.ndarray:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=False))
        channels = []
        for channel in range(observer_channels):
            base = y @ observer_proj[channel]
            mod = np.sin(y @ observer_mod_proj[channel] + observer_bias[channel])
            hidden = _safe_normalize(base + observer_gain * mod)
            channels.append(hidden.astype(np.float32))
        return np.stack(channels, axis=1)

    def _observer_summary(observer_hidden: np.ndarray) -> np.ndarray:
        summary = np.mean(np.maximum(observer_hidden, 0.0), axis=2)
        summary = summary * observer_gate[None, :]
        return summary.astype(np.float32, copy=False)

    def _aux_profile(
        relation_profile: np.ndarray,
        observer_hidden: np.ndarray,
        transport_graph: np.ndarray | None,
    ) -> np.ndarray:
        mixed_profile = np.array(relation_profile, dtype=np.float32, copy=True)
        if (
            transport_graph is not None
            and transport_graph.shape[0] == relation_profile.shape[0]
        ):
            transported = transport_graph @ relation_profile
            mixed_profile = (
                1.0 - transport_mix
            ) * mixed_profile + transport_mix * transported
        observer_profile = _observer_summary(observer_hidden)
        merged = np.concatenate(
            [mixed_profile, profile_mix * observer_profile], axis=1
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _channel_weights(
        query_observer: np.ndarray,
        support_observer: np.ndarray,
        support_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        support_response = np.maximum(
            np.sum(query_observer[:, None, :, :] * support_observer, axis=3), 0.0
        ).astype(np.float32)
        channel_weight = np.sum(
            support_response * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        channel_weight = channel_weight * observer_gate[None, :]
        shifted = channel_weight - np.max(channel_weight, axis=1, keepdims=True)
        channel_weight = np.exp(shifted / max(1e-4, observer_temperature)).astype(
            np.float32
        )
        total = np.sum(channel_weight, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        channel_weight = channel_weight / total
        return support_response, channel_weight

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, amp_hidden, energy = _encode_wave(x)
        relation_profile = _relation_profile(energy).astype(np.float32)
        observer_hidden = _observer_hidden(x)
        doc_count = wave_real.shape[0]
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        transport_graph = support_graph
        if doc_count > 1:
            graph_scores = _carrier_score(
                wave_real, wave_imag, amp_hidden, wave_real, wave_imag, amp_hidden
            )
            np.fill_diagonal(graph_scores, -np.inf)
            keep = min(graph_k, doc_count - 1)
            if keep > 0:
                neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[
                    :, :keep
                ]
                neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
                shifted = neighbor_scores - np.max(
                    neighbor_scores, axis=1, keepdims=True
                )
                neighbor_weights = np.exp(
                    shifted / max(1e-4, graph_temperature)
                ).astype(np.float32)
                weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
                weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
                neighbor_weights = neighbor_weights / weight_sum
                np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
                transport_graph = 0.5 * (support_graph + support_graph.T)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "amp_hidden": amp_hidden,
            "energy": energy,
            "support_graph": support_graph,
            "transport_graph": transport_graph,
            "relation_profile": relation_profile,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
            "aux_operator_profile": _aux_profile(
                relation_profile, observer_hidden, transport_graph
            ),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, amp_hidden, energy = _encode_wave(x)
        relation_profile = _relation_profile(energy).astype(np.float32)
        observer_hidden = _observer_hidden(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "amp_hidden": amp_hidden,
            "energy": energy,
            "relation_profile": relation_profile,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
            "aux_operator_profile": _aux_profile(
                relation_profile, observer_hidden, None
            ),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            query_state["amp_hidden"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
            doc_state["amp_hidden"],
        )

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_scores = _score(doc_state, query_state)
        doc_count = base_scores.shape[1]
        if doc_count == 0:
            return base_scores
        transport_graph = doc_state["transport_graph"]
        doc_observer = doc_state["observer_channels"]
        if transport_graph.shape != (doc_count, doc_count):
            return base_scores
        if doc_observer.shape[0] != doc_count:
            return base_scores
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return base_scores
        support_idx = np.argpartition(-base_scores, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(base_scores, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_rows = transport_graph[support_idx]
        support_observer = doc_observer[support_idx]
        query_observer = query_state["observer_channels"]
        support_response, channel_weight = _channel_weights(
            query_observer, support_observer, support_weights
        )
        candidate_response = np.maximum(
            np.sum(query_observer[:, None, :, :] * doc_observer[None, :, :, :], axis=3),
            0.0,
        ).astype(np.float32)
        weighted_support = np.sqrt(np.maximum(support_response, 0.0)) * np.sqrt(
            channel_weight[:, None, :]
        )
        weighted_candidate = np.sqrt(np.maximum(candidate_response, 0.0)) * np.sqrt(
            channel_weight[:, None, :]
        )
        observer_filter = np.einsum(
            "qsc,qdc->qsd", weighted_support, weighted_candidate, dtype=np.float32
        )
        observer_filter = np.clip(observer_filter, 0.0, 1.0)
        filtered_rows = support_rows * (
            observer_edge_floor + (1.0 - observer_edge_floor) * observer_filter
        )
        transport_signal = np.sum(
            filtered_rows * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        relation_signal = (
            query_state["relation_profile"] @ doc_state["relation_profile"].T
        ).astype(np.float32)
        aux_scores = (
            1.0 - transport_mix
        ) * relation_signal + transport_mix * transport_signal
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return aux_scores / scale

    return EmbeddingStateMethod(
        method_name="complex_observer_subgraph_transport_observable_v0",
        family="state_complex_observer_subgraph_transport_observable",
        params={
            "dim": dim,
            "bands": float(bands),
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "graph_k": float(graph_k),
            "graph_temperature": graph_temperature,
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "transport_mix": transport_mix,
            "support_width": float(support_width),
            "support_temperature": support_temperature,
            "observer_channels": float(observer_channels),
            "observer_dim": float(observer_dim),
            "observer_gain": observer_gain,
            "observer_temperature": observer_temperature,
            "observer_edge_floor": observer_edge_floor,
            "profile_mix": profile_mix,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _complex_observer_prototype_support_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    codebook_size = max(8, int(params.get("codebook_size", 16)))
    code_top_k = min(codebook_size, max(1, int(params.get("code_top_k", 2))))
    code_temperature = float(params.get("code_temperature", 0.10))
    code_support_gain = float(params.get("code_support_gain", 0.30))
    code_focus_floor = float(params.get("code_focus_floor", 0.18))
    code_profile_mix = float(params.get("code_profile_mix", 0.20))
    prototype_refine_steps = max(0, int(params.get("prototype_refine_steps", 1)))
    dim = int(params.get("dim", 0))
    secret_key = str(params.get("secret_key", ""))
    operator_method = _complex_observer_subgraph_transport_observable_build(rng, params)
    aux_score = operator_method.aux_score

    if aux_score is None:
        return EmbeddingStateMethod(
            method_name="complex_observer_prototype_support_observable_v0",
            family="state_complex_observer_prototype_support_observable",
            params={
                **operator_method.params,
                "codebook_size": float(codebook_size),
                "code_top_k": float(code_top_k),
                "code_temperature": code_temperature,
                "code_support_gain": code_support_gain,
                "code_focus_floor": code_focus_floor,
                "code_profile_mix": code_profile_mix,
                "prototype_refine_steps": float(prototype_refine_steps),
            },
            encode_docs=operator_method.encode_docs,
            encode_queries=operator_method.encode_queries,
            score=operator_method.score,
        )

    def _semantic_codes(energy: np.ndarray, centers: np.ndarray) -> np.ndarray:
        if centers.shape[0] == 0:
            return np.zeros((energy.shape[0], 0), dtype=np.float32)
        local_top_k = min(code_top_k, centers.shape[0])
        logits = energy @ centers.T
        return _topk_soft_assign(logits, local_top_k, code_temperature)

    def _merge_aux_profile(
        base_profile: np.ndarray | None, semantic_codes: np.ndarray
    ) -> np.ndarray:
        if base_profile is None:
            return _safe_normalize(semantic_codes.astype(np.float32, copy=False))
        if base_profile.shape[0] != semantic_codes.shape[0]:
            return base_profile.astype(np.float32, copy=False)
        merged = np.concatenate(
            [
                base_profile.astype(np.float32, copy=False),
                code_profile_mix * semantic_codes.astype(np.float32, copy=False),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _prototype_centers(
        energy: np.ndarray, effective_codebook_size: int
    ) -> np.ndarray:
        if effective_codebook_size <= 0:
            return np.zeros((0, energy.shape[1]), dtype=np.float32)
        prototype_rng = np.random.default_rng(
            _method_seed(
                "complex_observer_prototype_support_observable_v0_prototype_bank",
                secret_key,
                dim,
                offset=energy.shape[0],
            )
        )
        prototype_idx = np.sort(
            prototype_rng.choice(
                energy.shape[0], size=effective_codebook_size, replace=False
            )
        )
        centers = np.asarray(energy[prototype_idx], dtype=np.float32)
        centers = _safe_normalize(centers)
        if prototype_refine_steps <= 0:
            return centers
        for _ in range(prototype_refine_steps):
            logits = energy @ centers.T
            assign_idx = np.argmax(logits, axis=1)
            updated_centers = np.array(centers, copy=True)
            for center_idx in range(effective_codebook_size):
                member_mask = assign_idx == center_idx
                if not np.any(member_mask):
                    continue
                updated_centers[center_idx] = np.mean(
                    energy[member_mask], axis=0, dtype=np.float32
                )
            centers = _safe_normalize(updated_centers.astype(np.float32, copy=False))
        return centers

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(operator_method.encode_docs(x))
        energy = state["energy"]
        doc_count = energy.shape[0]
        if doc_count == 0:
            empty_codes = np.zeros((0, 0), dtype=np.float32)
            empty_centers = np.zeros((0, energy.shape[1]), dtype=np.float32)
            state["semantic_codes"] = empty_codes
            state["semantic_centers"] = empty_centers
            state["semantic_code_config"] = np.asarray(
                [float(code_top_k), code_temperature], dtype=np.float32
            )
            state["aux_operator_profile"] = empty_codes
            return state
        effective_codebook_size = min(codebook_size, doc_count)
        semantic_centers = _prototype_centers(energy, effective_codebook_size)
        semantic_codes = _semantic_codes(energy, semantic_centers).astype(np.float32)
        state["semantic_codes"] = semantic_codes
        state["semantic_centers"] = semantic_centers.astype(np.float32)
        state["semantic_code_config"] = np.asarray(
            [float(code_top_k), code_temperature], dtype=np.float32
        )
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), semantic_codes
        )
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        return dict(operator_method.encode_queries(x))

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_aux = aux_score(doc_state, query_state)
        semantic_codes = doc_state.get("semantic_codes")
        semantic_centers = doc_state.get("semantic_centers")
        semantic_config = doc_state.get("semantic_code_config")
        if semantic_codes is None:
            return base_aux
        if semantic_centers is None:
            return base_aux
        if semantic_config is None:
            return base_aux
        if "energy" not in query_state:
            return base_aux
        doc_count = base_aux.shape[1]
        if semantic_codes.shape[0] != doc_count:
            return base_aux
        if semantic_centers.shape[0] == 0:
            return base_aux

        local_top_k = min(
            semantic_centers.shape[0],
            max(1, int(round(float(semantic_config[0])))),
        )
        local_temperature = max(1e-4, float(semantic_config[1]))
        query_codes = _topk_soft_assign(
            query_state["energy"] @ semantic_centers.T,
            local_top_k,
            local_temperature,
        )
        carrier = operator_method.score(doc_state, query_state)
        support_width = min(
            max(1, int(round(float(operator_method.params.get("support_width", 8))))),
            doc_count,
        )
        if support_width <= 0:
            return base_aux
        support_idx = np.argpartition(-carrier, support_width - 1, axis=1)[
            :, :support_width
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        support_temperature = max(
            1e-4, float(operator_method.params.get("support_temperature", 0.10))
        )
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_codes = semantic_codes[support_idx]
        support_context = np.sum(
            support_codes * support_weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        support_context = _safe_normalize(support_context)
        support_alignment = np.sum(
            query_codes[:, None, :] * support_codes,
            axis=2,
            dtype=np.float32,
        )
        candidate_alignment = np.sum(
            support_context[:, None, :] * semantic_codes[None, :, :],
            axis=2,
            dtype=np.float32,
        )
        community_support = np.sum(
            support_weights[:, :, None]
            * np.sqrt(np.clip(np.maximum(support_alignment, 0.0), 0.0, 1.0))[:, :, None]
            * np.sqrt(np.clip(np.maximum(candidate_alignment, 0.0), 0.0, 1.0))[
                :, None, :
            ],
            axis=1,
            dtype=np.float32,
        )
        community_support = community_support - np.mean(
            community_support, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(community_support), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        community_support = community_support / signal_scale
        query_peak = np.max(query_codes, axis=1, keepdims=True)
        family_alignment = np.sum(
            query_codes * support_context,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        focus = np.sqrt(
            np.clip(query_peak * np.maximum(family_alignment, 0.0), 0.0, 1.0)
        ).astype(np.float32)
        focus_gate = np.clip(
            (focus - code_focus_floor) / max(1e-4, 1.0 - code_focus_floor),
            0.0,
            1.0,
        )
        combined = base_aux + code_support_gain * focus_gate * community_support
        combined = combined - np.mean(combined, axis=1, keepdims=True)
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return combined / combined_scale

    return EmbeddingStateMethod(
        method_name="complex_observer_prototype_support_observable_v0",
        family="state_complex_observer_prototype_support_observable",
        params={
            **operator_method.params,
            "codebook_size": float(codebook_size),
            "code_top_k": float(code_top_k),
            "code_temperature": code_temperature,
            "code_support_gain": code_support_gain,
            "code_focus_floor": code_focus_floor,
            "code_profile_mix": code_profile_mix,
            "prototype_refine_steps": float(prototype_refine_steps),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=operator_method.score,
        aux_score=_aux_score,
    )


def _build_complex_holonomy_observable(
    method_name: str, rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "complex_wavepacket_v0"
    dim = int(params.get("dim", 0))
    bands = int(params.get("bands", 4))
    phase_scale = float(params.get("phase_scale", 0.9))
    envelope_gain = float(params.get("envelope_gain", 0.55))
    holonomy_loops = max(2, int(params.get("holonomy_loops", 8)))
    holonomy_curvature = float(params.get("holonomy_curvature", 0.82))
    residual_mix = float(params.get("residual_mix", 0.0))
    public_ratio = float(params.get("public_ratio", 0.5))
    public_mask = float(params.get("public_mask", 0.42))
    public_chunk = int(params.get("public_chunk", 2))
    public_dim = max(8, int(dim * public_ratio))
    public_band_chunk = max(4, public_dim // max(1, bands))
    relation_dim = max(1, bands) * dim
    aux_dim = holonomy_loops * relation_dim
    secret_key = str(params.get("secret_key", ""))

    carrier_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    real_proj = np.stack([_qr_orthogonal(carrier_rng, dim, dim) for _ in range(bands)])
    imag_proj = np.stack([_qr_orthogonal(carrier_rng, dim, dim) for _ in range(bands)])
    public_mix = np.stack(
        [
            carrier_rng.normal(size=(dim, public_band_chunk)).astype(np.float32)
            for _ in range(bands)
        ]
    )
    band_weights = np.abs(carrier_rng.normal(size=(bands,)).astype(np.float32))
    band_weights = band_weights / np.maximum(1e-6, np.sum(band_weights))
    band_phase = carrier_rng.uniform(0.0, 2.0 * math.pi, size=(bands, dim)).astype(
        np.float32
    )

    holonomy_rng = np.random.default_rng(
        _method_seed("holonomy_loop_embedding_v0", secret_key, dim)
    )
    holonomy_proj = np.stack(
        [_qr_orthogonal(holonomy_rng, dim, dim) for _ in range(holonomy_loops)]
    )
    holonomy_phase_bias = holonomy_rng.uniform(
        -math.pi, math.pi, size=(holonomy_loops, dim)
    ).astype(np.float32)
    shared_bridge = np.zeros((relation_dim, aux_dim), dtype=np.float32)
    if residual_mix > 0.0:
        residual_rng = np.random.default_rng(_method_seed(method_name, secret_key, dim))
        shared_bridge = residual_rng.normal(size=(relation_dim, aux_dim)).astype(
            np.float32
        )

    def _encode_wave(
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = np.einsum("nd,bdk->nbk", y, real_proj)
        imag = np.einsum("nd,bdk->nbk", y, imag_proj)
        phase = phase_scale * real + band_phase[None, :, :]
        amplitude = 1.0 + envelope_gain * np.tanh(imag)
        wave_real = amplitude * np.cos(phase)
        wave_imag = amplitude * np.sin(phase)
        wave_norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=2, keepdims=True))
        wave_norm = np.where(wave_norm == 0.0, 1.0, wave_norm)
        wave_real = wave_real / wave_norm
        wave_imag = wave_imag / wave_norm
        amp_hidden = np.sqrt(np.maximum(amplitude, 1e-6)).astype(np.float32)
        amp_norm = np.linalg.norm(amp_hidden, axis=2, keepdims=True)
        amp_norm = np.where(amp_norm == 0.0, 1.0, amp_norm)
        amp_hidden = amp_hidden / amp_norm
        public_chunks = []
        for band in range(bands):
            band_energy = np.sqrt(wave_real[:, band] ** 2 + wave_imag[:, band] ** 2)
            chunk = band_energy @ public_mix[band]
            public_chunks.append(chunk.astype(np.float32))
        public = _safe_normalize(np.hstack(public_chunks))
        public = _mask_public_observation(
            base_method_name, secret_key, public, public_mask, public_chunk
        )
        band_energy = (wave_real**2 + wave_imag**2).astype(np.float32)
        return (
            public,
            wave_real.astype(np.float32),
            wave_imag.astype(np.float32),
            amp_hidden.astype(np.float32),
            band_energy,
        )

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        amp_hidden_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
        amp_hidden_b: np.ndarray,
    ) -> np.ndarray:
        sim = np.zeros((wave_real_a.shape[0], wave_real_b.shape[0]), dtype=np.float32)
        for band in range(bands):
            coherence = wave_real_a[:, band] @ wave_real_b[:, band].T
            coherence = coherence + wave_imag_a[:, band] @ wave_imag_b[:, band].T
            amplitude = amp_hidden_a[:, band] @ amp_hidden_b[:, band].T
            band_sim = 0.65 * coherence + 0.35 * amplitude
            sim = sim + band_weights[band] * band_sim
        return sim / float(max(1, bands))

    def _holonomy_aux_profile(band_energy: np.ndarray) -> np.ndarray:
        loop_features = []
        for loop_idx in range(holonomy_loops):
            band_features = []
            for band in range(bands):
                base = band_energy[:, band] @ holonomy_proj[loop_idx]
                shifted = np.roll(base, loop_idx + band + 1, axis=1)
                holonomy = np.cos(
                    holonomy_curvature * base + holonomy_phase_bias[loop_idx]
                )
                holonomy = holonomy + np.sin(holonomy_curvature * shifted)
                holonomy = _safe_normalize(holonomy)
                band_features.append(holonomy.astype(np.float32))
            loop_features.append(np.hstack(band_features))
        aux_profile = _safe_normalize(np.hstack(loop_features).astype(np.float32))
        if residual_mix <= 0.0:
            return aux_profile
        shared_projection = _safe_normalize(
            band_energy.reshape(band_energy.shape[0], -1) @ shared_bridge
        )
        aux_profile = aux_profile - residual_mix * shared_projection
        aux_profile = aux_profile - np.mean(aux_profile, axis=1, keepdims=True)
        return _safe_normalize(aux_profile.astype(np.float32, copy=False))

    def _encode(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, amp_hidden, band_energy = _encode_wave(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "amp_hidden": amp_hidden,
            "aux_operator_profile": _holonomy_aux_profile(band_energy),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            query_state["amp_hidden"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
            doc_state["amp_hidden"],
        )

    return EmbeddingStateMethod(
        method_name=method_name,
        family="state_complex_parallel_holonomy_observable",
        params={
            "dim": dim,
            "bands": float(bands),
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "holonomy_loops": float(holonomy_loops),
            "holonomy_curvature": holonomy_curvature,
            "residual_mix": residual_mix,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _complex_parallel_holonomy_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    return _build_complex_holonomy_observable(
        "complex_parallel_holonomy_observable_v0",
        rng,
        params,
    )


def _complex_transport_holonomy_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    graph_k = max(4, int(params.get("graph_k", 8)))
    graph_temperature = float(params.get("graph_temperature", 0.16))
    support_width = max(graph_k, int(params.get("support_width", 10)))
    support_temperature = float(params.get("support_temperature", 0.12))
    transport_mix = float(params.get("transport_mix", 0.58))
    symmetry_mix = float(params.get("symmetry_mix", 0.35))
    profile_mix = float(params.get("profile_mix", 0.40))
    carrier_graph_mix = float(params.get("carrier_graph_mix", 0.0))
    base_method = _build_complex_holonomy_observable(
        "complex_transport_holonomy_observable_v0",
        rng,
        params,
    )

    def _centered_profile(profile: np.ndarray) -> np.ndarray:
        centered = np.array(profile, dtype=np.float32, copy=True)
        centered = centered - np.mean(centered, axis=1, keepdims=True)
        return _safe_normalize(centered.astype(np.float32, copy=False))

    def _support_graph(
        profile: np.ndarray, carrier_scores: np.ndarray | None
    ) -> np.ndarray:
        doc_count = profile.shape[0]
        graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        if doc_count <= 1:
            return graph
        scores = profile @ profile.T
        if carrier_scores is not None and carrier_scores.shape == scores.shape:
            carrier_centered = carrier_scores - np.mean(
                carrier_scores, axis=1, keepdims=True
            )
            carrier_scale = np.max(np.abs(carrier_centered), axis=1, keepdims=True)
            carrier_scale = np.where(carrier_scale == 0.0, 1.0, carrier_scale)
            carrier_centered = carrier_centered / carrier_scale
            scores = (1.0 - carrier_graph_mix) * scores
            scores = scores + carrier_graph_mix * carrier_centered
        np.fill_diagonal(scores, -np.inf)
        keep = min(graph_k, doc_count - 1)
        if keep <= 0:
            return graph
        neighbor_idx = np.argpartition(-scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        np.put_along_axis(graph, neighbor_idx, neighbor_weights, axis=1)
        return ((1.0 - symmetry_mix) * graph + symmetry_mix * graph.T).astype(
            np.float32
        )

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_docs(x))
        base_profile = _centered_profile(state["aux_operator_profile"])
        carrier_scores = base_method.score(state, state)
        transport_graph = _support_graph(base_profile, carrier_scores)
        transported_profile = transport_graph @ base_profile
        mixed_profile = (1.0 - profile_mix) * base_profile
        mixed_profile = mixed_profile + profile_mix * transported_profile
        state["holonomy_profile"] = base_profile
        state["transport_graph"] = transport_graph
        state["aux_operator_profile"] = _centered_profile(mixed_profile)
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_queries(x))
        base_profile = _centered_profile(state["aux_operator_profile"])
        state["holonomy_profile"] = base_profile
        state["aux_operator_profile"] = base_profile
        return state

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        doc_profile = doc_state.get("aux_operator_profile")
        query_profile = query_state.get("aux_operator_profile")
        transport_graph = doc_state.get("transport_graph")
        if doc_profile is None:
            return base_method.score(doc_state, query_state)
        if query_profile is None:
            return base_method.score(doc_state, query_state)
        if transport_graph is None:
            return query_profile @ doc_profile.T
        doc_count = doc_profile.shape[0]
        if query_profile.shape[1] != doc_profile.shape[1]:
            return base_method.score(doc_state, query_state)
        if transport_graph.shape != (doc_count, doc_count):
            return query_profile @ doc_profile.T
        base_aux = query_profile @ doc_profile.T
        base_aux = base_aux - np.mean(base_aux, axis=1, keepdims=True)
        base_scale = np.max(np.abs(base_aux), axis=1, keepdims=True)
        base_scale = np.where(base_scale == 0.0, 1.0, base_scale)
        base_aux = base_aux / base_scale
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return base_aux
        support_idx = np.argpartition(-base_aux, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(base_aux, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_vector = np.zeros_like(base_aux, dtype=np.float32)
        np.put_along_axis(support_vector, support_idx, support_weights, axis=1)
        transported_scores = support_vector @ transport_graph
        transported_scores = transported_scores - np.mean(
            transported_scores, axis=1, keepdims=True
        )
        transport_scale = np.max(np.abs(transported_scores), axis=1, keepdims=True)
        transport_scale = np.where(transport_scale == 0.0, 1.0, transport_scale)
        transported_scores = transported_scores / transport_scale
        combined = (1.0 - transport_mix) * base_aux + transport_mix * transported_scores
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return combined / combined_scale

    return EmbeddingStateMethod(
        method_name="complex_transport_holonomy_observable_v0",
        family="state_complex_transport_holonomy_observable",
        params={
            **base_method.params,
            "graph_k": float(graph_k),
            "graph_temperature": graph_temperature,
            "support_width": float(support_width),
            "support_temperature": support_temperature,
            "transport_mix": transport_mix,
            "symmetry_mix": symmetry_mix,
            "profile_mix": profile_mix,
            "carrier_graph_mix": carrier_graph_mix,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base_method.score,
        aux_score=_aux_score,
    )


def _observer_resonance_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    channels = int(params.get("channels", 6))
    channel_gain = float(params.get("channel_gain", 0.60))
    public_ratio = float(params.get("public_ratio", 0.33))
    public_mask = float(params.get("public_mask", 0.48))
    public_chunk = int(params.get("public_chunk", 2))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("observer_resonance_v0", secret_key, dim)
    )
    channel_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, dim) for _ in range(channels)]
    )
    mod_proj = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(channels)])
    public_mix = local_rng.normal(size=(channels, dim, public_dim)).astype(np.float32)
    gate = _safe_normalize(
        np.abs(local_rng.normal(size=(1, channels)).astype(np.float32))
    ).reshape(-1)
    bias = local_rng.uniform(-math.pi, math.pi, size=(channels, dim)).astype(np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        channels_out = []
        public_chunks = []
        for channel in range(channels):
            base = y @ channel_proj[channel]
            mod = np.sin(y @ mod_proj[channel] + bias[channel])
            hidden = _safe_normalize(base + channel_gain * mod)
            channels_out.append(hidden.astype(np.float32))
            public_features = np.abs(hidden) @ public_mix[channel]
            public_chunks.append(public_features.astype(np.float32))
        hidden = np.stack(channels_out, axis=1)
        public = _safe_normalize(np.mean(np.stack(public_chunks, axis=1), axis=1))
        public = _mask_public_observation(
            "observer_resonance_v0", secret_key, public, public_mask, public_chunk
        )
        return {"public": public, "channels": hidden}

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        sim = np.zeros(
            (query_state["public"].shape[0], doc_state["public"].shape[0]),
            dtype=np.float32,
        )
        for channel in range(channels):
            q_hidden = query_state["channels"][:, channel]
            d_hidden = doc_state["channels"][:, channel]
            sim = sim + gate[channel] * (q_hidden @ d_hidden.T)
        return sim / float(max(1, channels))

    return EmbeddingStateMethod(
        method_name="observer_resonance_v0",
        family="state_observer",
        params={
            "dim": dim,
            "channels": channels,
            "channel_gain": channel_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _tensor_graph_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    slots = int(params.get("slots", 6))
    slot_dim = max(4, dim // max(1, slots))
    public_ratio = float(params.get("public_ratio", 0.5))
    public_dim = max(8, int(dim * public_ratio))
    relation_gain = float(params.get("relation_gain", 0.45))
    public_mask = float(params.get("public_mask", 0.35))
    public_chunk = int(params.get("public_chunk", 3))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed("tensor_graph_v0", secret_key, dim))
    slot_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, slot_dim) for _ in range(slots)]
    )
    public_mix = local_rng.normal(size=(slots, slot_dim, public_dim // 2)).astype(
        np.float32
    )

    tri_upper = np.triu_indices(slots, k=1)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        factors = np.einsum("nd,sdh->nsh", y, slot_proj)
        factor_norm = np.linalg.norm(factors, axis=2, keepdims=True)
        factor_norm = np.where(factor_norm == 0.0, 1.0, factor_norm)
        factors = factors / factor_norm
        relations = np.einsum("nsh,nth->nst", factors, factors) / float(slot_dim)
        rel_flat = relations[:, tri_upper[0], tri_upper[1]]
        rel_sorted = np.sort(rel_flat, axis=1)
        public_nodes = []
        for slot in range(slots):
            public_nodes.append(np.abs(factors[:, slot]) @ public_mix[slot])
        public = np.hstack(public_nodes + [rel_sorted.astype(np.float32)])
        public = _safe_normalize(public)
        public = _mask_public_observation(
            "tensor_graph_v0", secret_key, public, public_mask, public_chunk
        )
        return {
            "public": public,
            "factors": factors.astype(np.float32),
            "relations": rel_flat.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        slot_score = np.zeros(
            (query_state["public"].shape[0], doc_state["public"].shape[0]),
            dtype=np.float32,
        )
        for slot in range(slots):
            slot_score = slot_score + (
                query_state["factors"][:, slot] @ doc_state["factors"][:, slot].T
            )
        slot_score = slot_score / float(max(1, slots))
        relation_score = query_state["relations"] @ doc_state["relations"].T
        relation_score = relation_score / float(
            max(1, query_state["relations"].shape[1])
        )
        return (1.0 - relation_gain) * slot_score + relation_gain * relation_score

    return EmbeddingStateMethod(
        method_name="tensor_graph_v0",
        family="state_tensor_graph",
        params={
            "dim": dim,
            "slots": slots,
            "slot_dim": slot_dim,
            "public_ratio": public_ratio,
            "relation_gain": relation_gain,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _holonomy_loop_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    loops = int(params.get("loops", 8))
    public_ratio = float(params.get("public_ratio", 0.24))
    public_mask = float(params.get("public_mask", 0.66))
    public_chunk = int(params.get("public_chunk", 4))
    curvature = float(params.get("curvature", 0.82))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("holonomy_loop_embedding_v0", secret_key, dim)
    )
    public_dim = max(8, int(dim * public_ratio))
    loop_proj = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(loops)])
    phase_bias = local_rng.uniform(-math.pi, math.pi, size=(loops, dim)).astype(
        np.float32
    )
    public_mix = np.stack(
        [
            local_rng.normal(size=(dim, max(2, public_dim // loops + 1))).astype(
                np.float32
            )
            for _ in range(loops)
        ]
    )

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        loop_states = []
        public_chunks = []
        for loop_idx in range(loops):
            base = y @ loop_proj[loop_idx]
            shifted = np.roll(base, loop_idx + 1, axis=1)
            holonomy = np.cos(curvature * base + phase_bias[loop_idx])
            holonomy = holonomy + np.sin(curvature * shifted)
            holonomy = _safe_normalize(holonomy)
            loop_states.append(holonomy.astype(np.float32))
            public_features = np.abs(holonomy) @ public_mix[loop_idx]
            public_chunks.append(public_features.astype(np.float32))
        hidden = np.stack(loop_states, axis=1)
        public = _safe_normalize(np.hstack(public_chunks))
        public = _mask_public_observation(
            "holonomy_loop_embedding_v0", secret_key, public, public_mask, public_chunk
        )
        return {"public": public, "loops": hidden}

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        sim = np.zeros(
            (query_state["public"].shape[0], doc_state["public"].shape[0]),
            dtype=np.float32,
        )
        for loop_idx in range(loops):
            sim = sim + (
                query_state["loops"][:, loop_idx] @ doc_state["loops"][:, loop_idx].T
            )
        return sim / float(max(1, loops))

    return EmbeddingStateMethod(
        method_name="holonomy_loop_embedding_v0",
        family="state_holonomy",
        params={
            "dim": dim,
            "loops": float(loops),
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
            "curvature": curvature,
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _codebook_superposition_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    codebook_size = int(params.get("codebook_size", 64))
    top_k = int(params.get("top_k", 5))
    residual_weight = float(params.get("residual_weight", 0.34))
    public_ratio = float(params.get("public_ratio", 0.20))
    public_mask = float(params.get("public_mask", 0.78))
    public_chunk = int(params.get("public_chunk", 4))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("codebook_superposition_embedding_v0", secret_key, dim)
    )
    codebook = _safe_normalize(
        local_rng.normal(size=(codebook_size, dim)).astype(np.float32)
    )
    residual_proj, _ = np.linalg.qr(local_rng.normal(size=(dim, dim)))
    public_dim = max(8, int(dim * public_ratio))
    public_mix = local_rng.normal(size=(codebook_size, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        logits = y @ codebook.T
        sparse = _topk_soft_assign(logits, top_k, 0.18)
        recon = sparse @ codebook
        residual = _safe_normalize((y - recon) @ residual_proj.T.astype(np.float32))
        public = sparse @ public_mix
        public = _safe_normalize(public)
        public = _mask_public_observation(
            "codebook_superposition_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public,
            "codes": sparse.astype(np.float32),
            "residual": residual.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        code_score = query_state["codes"] @ doc_state["codes"].T
        residual_score = query_state["residual"] @ doc_state["residual"].T
        return (1.0 - residual_weight) * code_score + residual_weight * residual_score

    return EmbeddingStateMethod(
        method_name="codebook_superposition_embedding_v0",
        family="state_codebook",
        params={
            "dim": dim,
            "codebook_size": float(codebook_size),
            "top_k": float(top_k),
            "residual_weight": residual_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _keyed_wave_superpose_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    modes = max(2, int(params.get("modes", 8)))
    doc_top_k = max(1, min(modes, int(params.get("doc_top_k", 3))))
    query_top_k = max(1, min(modes, int(params.get("query_top_k", 1))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.22)))
    route_scale = float(params.get("route_scale", 1.25))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 2.2)))
    phase_scale = float(params.get("phase_scale", 0.78))
    envelope_gain = float(params.get("envelope_gain", 0.45))
    decoy_floor = float(params.get("decoy_floor", 0.24))
    decoy_floor = min(max(decoy_floor, 0.0), 0.85)
    coherence_weight = float(params.get("coherence_weight", 0.46))
    coherence_weight = min(max(coherence_weight, 0.0), 1.0)
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.84))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("keyed_wave_superpose_embedding_v0", secret_key, dim)
    )
    route_bank = _safe_normalize(
        local_rng.normal(size=(modes, hidden_dim)).astype(np.float32)
    )
    route_bias = local_rng.uniform(-0.35, 0.35, size=(modes,)).astype(np.float32)
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    mode_phase_shift = local_rng.uniform(
        -math.pi, math.pi, size=(modes, hidden_dim)
    ).astype(np.float32)
    public_mix = local_rng.normal(size=(modes, public_dim)).astype(np.float32)
    mode_cos = np.cos(mode_phase_shift).astype(np.float32)
    mode_sin = np.sin(mode_phase_shift).astype(np.float32)

    def _scramble_from_seed(seed: int, size: int):
        drng = np.random.default_rng(seed)
        perm = drng.permutation(size)
        signs = drng.choice([-1.0, 1.0], size=size).astype(np.float32)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(size)
        return perm, signs, inv

    def _scramble_waves(state: StateMap) -> StateMap:
        n = state["wave_real"].shape[0]
        wave_r = state["wave_real"].reshape(n, -1).copy()
        wave_i = state["wave_imag"].reshape(n, -1).copy()
        base_r = state["base_wave_real"].copy()
        base_i = state["base_wave_imag"].copy()
        mw = state["mode_weight"]
        wave_size = wave_r.shape[1]
        base_size = base_r.shape[1]
        for i in range(n):
            # Seed from mode_weight (not scrambled, stable across encode/decode)
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            from cryptography.hazmat.primitives import hashes
            ikm = secret_key.encode("utf-8") + mw[i].tobytes()
            km_w = HKDF(algorithm=hashes.SHA512(), length=64, salt=b"kpt-scramble-wave", info=b"").derive(ikm)
            km_b = HKDF(algorithm=hashes.SHA512(), length=64, salt=b"kpt-scramble-base", info=b"").derive(ikm)
            seed_w = int.from_bytes(km_w, "big")
            seed_b = int.from_bytes(km_b, "big")
            wp, ws, _ = _scramble_from_seed(seed_w, wave_size)
            wave_r[i] = wave_r[i, wp] * ws
            wave_i[i] = wave_i[i, wp] * ws
            bp, bs, _ = _scramble_from_seed(seed_b, base_size)
            base_r[i] = base_r[i, bp] * bs
            base_i[i] = base_i[i, bp] * bs
        return {
            **state,
            "wave_real": wave_r.reshape(n, modes, hidden_dim).astype(np.float32),
            "wave_imag": wave_i.reshape(n, modes, hidden_dim).astype(np.float32),
            "base_wave_real": base_r.astype(np.float32),
            "base_wave_imag": base_i.astype(np.float32),
        }

    def _unscramble_waves(state: StateMap) -> StateMap:
        n = state["wave_real"].shape[0]
        wave_r = state["wave_real"].reshape(n, -1).copy()
        wave_i = state["wave_imag"].reshape(n, -1).copy()
        base_r = state["base_wave_real"].copy()
        base_i = state["base_wave_imag"].copy()
        mw = state["mode_weight"]
        wave_size = wave_r.shape[1]
        base_size = base_r.shape[1]
        for i in range(n):
            from cryptography.hazmat.primitives.kdf.hkdf import HKDF
            from cryptography.hazmat.primitives import hashes
            ikm = secret_key.encode("utf-8") + mw[i].tobytes()
            km_w = HKDF(algorithm=hashes.SHA512(), length=64, salt=b"kpt-scramble-wave", info=b"").derive(ikm)
            km_b = HKDF(algorithm=hashes.SHA512(), length=64, salt=b"kpt-scramble-base", info=b"").derive(ikm)
            seed_w = int.from_bytes(km_w, "big")
            seed_b = int.from_bytes(km_b, "big")
            _, ws, wi = _scramble_from_seed(seed_w, wave_size)
            wave_r[i] = (wave_r[i] * ws)[wi]
            wave_i[i] = (wave_i[i] * ws)[wi]
            _, bs, bi = _scramble_from_seed(seed_b, base_size)
            base_r[i] = (base_r[i] * bs)[bi]
            base_i[i] = (base_i[i] * bs)[bi]
        return {
            **state,
            "wave_real": wave_r.reshape(n, modes, hidden_dim).astype(np.float32),
            "wave_imag": wave_i.reshape(n, modes, hidden_dim).astype(np.float32),
            "base_wave_real": base_r.astype(np.float32),
            "base_wave_imag": base_i.astype(np.float32),
        }

    def _encode(x: np.ndarray, top_k: int, scale: float) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        carrier_real = y @ real_proj
        carrier_imag = y @ imag_proj
        carrier_phase = phase_scale * carrier_real + phase_bias[None, :]
        carrier_amp = np.sqrt(
            np.maximum(1e-6, 1.0 + envelope_gain * np.tanh(carrier_imag))
        ).astype(np.float32)
        base_wave_real = carrier_amp * np.cos(carrier_phase)
        base_wave_imag = carrier_amp * np.sin(carrier_phase)
        base_norm = np.sqrt(
            np.sum(base_wave_real**2 + base_wave_imag**2, axis=1, keepdims=True)
        )
        base_norm = np.where(base_norm == 0.0, 1.0, base_norm)
        base_wave_real = base_wave_real / base_norm
        base_wave_imag = base_wave_imag / base_norm
        carrier_energy = (base_wave_real**2 + base_wave_imag**2).astype(np.float32)
        route_logits = (
            carrier_energy @ route_bank.T + route_bias[None, :]
        ) / math.sqrt(max(1.0, float(hidden_dim)))
        sparse_weight = _topk_soft_assign(
            route_logits * scale, top_k, route_temperature
        )
        mode_weight = decoy_floor / float(modes) + (1.0 - decoy_floor) * sparse_weight
        mode_total = np.sum(mode_weight, axis=1, keepdims=True)
        mode_total = np.where(mode_total == 0.0, 1.0, mode_total)
        mode_weight = mode_weight / mode_total
        mode_scale = np.sqrt(np.maximum(mode_weight, 1e-6)).astype(np.float32)[
            :, :, None
        ]
        rotated_real = (
            base_wave_real[:, None, :] * mode_cos[None, :, :]
            - base_wave_imag[:, None, :] * mode_sin[None, :, :]
        )
        rotated_imag = (
            base_wave_real[:, None, :] * mode_sin[None, :, :]
            + base_wave_imag[:, None, :] * mode_cos[None, :, :]
        )
        wave_real = mode_scale * rotated_real
        wave_imag = mode_scale * rotated_imag
        wave_norm = np.sqrt(
            np.sum(wave_real**2 + wave_imag**2, axis=(1, 2), keepdims=True)
        )
        wave_norm = np.where(wave_norm == 0.0, 1.0, wave_norm)
        wave_real = wave_real / wave_norm
        wave_imag = wave_imag / wave_norm
        mode_energy = mode_weight * np.maximum(route_logits, 0.0)
        public = _safe_normalize(np.tanh(mode_weight + 0.35 * mode_energy) @ public_mix)
        public = _mask_public_observation(
            "keyed_wave_superpose_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        raw_state = {
            "public": public.astype(np.float32),
            "base_wave_real": base_wave_real.astype(np.float32),
            "base_wave_imag": base_wave_imag.astype(np.float32),
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "mode_weight": mode_weight.astype(np.float32),
            "mode_energy": mode_energy.astype(np.float32),
        }
        return _scramble_waves(raw_state)

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _encode(x, doc_top_k, route_scale)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _encode(x, query_top_k, route_scale * collapse_gain)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        doc_clear = _unscramble_waves(doc_state)
        query_clear = _unscramble_waves(query_state)
        base_overlap = query_clear["base_wave_real"] @ doc_clear["base_wave_real"].T
        base_overlap = base_overlap + (
            query_clear["base_wave_imag"] @ doc_clear["base_wave_imag"].T
        )
        component_overlap = np.einsum(
            "qmh,dmh->qdm",
            query_clear["wave_real"],
            doc_clear["wave_real"],
            dtype=np.float32,
        )
        component_overlap = component_overlap + np.einsum(
            "qmh,dmh->qdm",
            query_clear["wave_imag"],
            doc_clear["wave_imag"],
            dtype=np.float32,
        )
        mode_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qm,dm->qdm",
                    query_state["mode_weight"],
                    doc_state["mode_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        energy_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qm,dm->qdm",
                    query_state["mode_energy"],
                    doc_state["mode_energy"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        coherence = (component_overlap**2) * (0.30 + 0.70 * mode_gate)
        coherence_score = np.mean(coherence, axis=2, dtype=np.float32)
        energy_score = np.mean(energy_gate, axis=2, dtype=np.float32)
        mode_support = query_state["mode_weight"] @ doc_state["mode_weight"].T
        superposed_score = (
            0.80 * coherence_score + 0.10 * energy_score + 0.10 * mode_support
        )
        return coherence_weight * superposed_score + (1.0 - coherence_weight) * (
            base_overlap**2
        )

    return EmbeddingStateMethod(
        method_name="keyed_wave_superpose_embedding_v0",
        family="state_wave_superposition",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "modes": float(modes),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "decoy_floor": decoy_floor,
            "coherence_weight": coherence_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _dual_carrier_superposition_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    bands = max(2, int(params.get("bands", 5)))
    summary_bands = max(2, int(params.get("summary_bands", bands)))
    doc_top_k = max(1, min(2, int(params.get("doc_top_k", 2))))
    query_top_k = max(1, min(2, int(params.get("query_top_k", 1))))
    collapse_temperature = max(1e-4, float(params.get("collapse_temperature", 0.12)))
    collapse_scale = float(params.get("collapse_scale", 1.20))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 2.20)))
    collapse_floor = float(params.get("collapse_floor", 0.18))
    collapse_floor = min(max(collapse_floor, 0.0), 0.75)
    agreement_weight = float(params.get("agreement_weight", 0.16))
    agreement_weight = min(max(agreement_weight, 0.0), 0.60)
    public_ratio = float(params.get("public_ratio", 0.16))
    public_mask = float(params.get("public_mask", 0.90))
    public_chunk = int(params.get("public_chunk", 8))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))

    wave_params = {
        "dim": dim,
        "secret_key": secret_key,
        "bands": bands,
        "phase_scale": float(params.get("wave_phase_scale", 0.80)),
        "envelope_gain": float(params.get("wave_envelope_gain", 0.45)),
        "public_ratio": float(params.get("wave_public_ratio", 0.25)),
        "public_mask": float(params.get("wave_public_mask", 0.72)),
        "public_chunk": int(params.get("wave_public_chunk", 5)),
    }
    projective_params = {
        "dim": dim,
        "secret_key": secret_key,
        "hidden_dim": max(12, int(params.get("hidden_dim", dim))),
        "phase_scale": float(params.get("projective_phase_scale", 0.75)),
        "public_ratio": float(params.get("projective_public_ratio", 0.18)),
        "public_mask": float(params.get("projective_public_mask", 0.80)),
        "public_chunk": int(params.get("projective_public_chunk", 5)),
    }
    wave_method = _complex_wavepacket_build(rng, wave_params)
    projective_method = _projective_hilbert_build(rng, projective_params)

    local_rng = np.random.default_rng(
        _method_seed("dual_carrier_superposition_v0", secret_key, dim)
    )
    collapse_bank = _safe_normalize(
        local_rng.normal(size=(2, bands + summary_bands)).astype(np.float32)
    )
    collapse_bias = local_rng.uniform(-0.25, 0.25, size=(2,)).astype(np.float32)
    observer_gate = np.abs(local_rng.normal(size=(2,)).astype(np.float32))
    observer_gate = observer_gate / np.maximum(1e-6, np.sum(observer_gate))
    semantic_centers = np.eye(2, dtype=np.float32)
    semantic_code_config = np.array([1.0, collapse_temperature], dtype=np.float32)

    def _projective_summary(state: StateMap) -> np.ndarray:
        energy = state["wave_real"] ** 2 + state["wave_imag"] ** 2
        chunks = np.array_split(np.arange(energy.shape[1]), summary_bands)
        summary_parts = []
        for chunk_idx in chunks:
            if chunk_idx.size == 0:
                continue
            summary_parts.append(np.mean(energy[:, chunk_idx], axis=1, keepdims=True))
        summary = np.hstack(summary_parts).astype(np.float32)
        return _safe_normalize(summary)

    def _carrier_state(state: StateMap, prefix: str) -> StateMap:
        out: StateMap = {"public": state[f"{prefix}_public"]}
        for key in ("wave_real", "wave_imag", "amp_hidden"):
            prefixed_key = f"{prefix}_{key}"
            if prefixed_key in state:
                out[key] = state[prefixed_key]
        return out

    def _encode(
        x: np.ndarray,
        top_k: int,
        scale: float,
        encoder_kind: str,
    ) -> StateMap:
        if encoder_kind == "docs":
            wave_state = wave_method.encode_docs(x)
            projective_state = projective_method.encode_docs(x)
        else:
            wave_state = wave_method.encode_queries(x)
            projective_state = projective_method.encode_queries(x)
        wave_summary = np.mean(wave_state["amp_hidden"], axis=2, dtype=np.float32)
        wave_summary = _safe_normalize(wave_summary.astype(np.float32))
        projective_summary = _projective_summary(projective_state)
        collapse_features = _safe_normalize(
            np.hstack([wave_summary, projective_summary]).astype(np.float32)
        )
        collapse_logits = collapse_features @ collapse_bank.T + collapse_bias[None, :]
        sparse = _topk_soft_assign(collapse_logits * scale, top_k, collapse_temperature)
        carrier_weight = collapse_floor / 2.0 + (1.0 - collapse_floor) * sparse
        carrier_total = np.sum(carrier_weight, axis=1, keepdims=True)
        carrier_total = np.where(carrier_total == 0.0, 1.0, carrier_total)
        carrier_weight = carrier_weight / carrier_total

        public_context = np.hstack(
            [
                carrier_weight[:, :1] * wave_state["public"],
                carrier_weight[:, 1:2] * projective_state["public"],
                collapse_features,
                carrier_weight,
            ]
        ).astype(np.float32)
        public_rng = np.random.default_rng(
            _method_seed(
                "dual_carrier_superposition_v0_public",
                secret_key,
                public_context.shape[1],
            )
        )
        public_mix = public_rng.normal(
            size=(public_context.shape[1], public_dim)
        ).astype(np.float32)
        public = _safe_normalize(np.tanh(public_context @ public_mix))
        public = _mask_public_observation(
            "dual_carrier_superposition_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )

        wave_gate = np.sqrt(np.maximum(carrier_weight[:, 0], 1e-6)).astype(np.float32)[
            :, None, None
        ]
        projective_gate = np.sqrt(np.maximum(carrier_weight[:, 1], 1e-6)).astype(
            np.float32
        )[:, None, None]
        combined_wave_real = np.concatenate(
            [
                wave_state["wave_real"] * wave_gate,
                projective_state["wave_real"][:, None, :] * projective_gate,
            ],
            axis=1,
        )
        combined_wave_imag = np.concatenate(
            [
                wave_state["wave_imag"] * wave_gate,
                projective_state["wave_imag"][:, None, :] * projective_gate,
            ],
            axis=1,
        )
        aux_profile = _safe_normalize(
            np.hstack([collapse_features, carrier_weight]).astype(np.float32)
        )
        return {
            "public": public.astype(np.float32),
            "wave_real": combined_wave_real.astype(np.float32),
            "wave_imag": combined_wave_imag.astype(np.float32),
            "carrier_weight": carrier_weight.astype(np.float32),
            "collapse_features": collapse_features.astype(np.float32),
            "energy": carrier_weight.astype(np.float32),
            "semantic_codes": carrier_weight.astype(np.float32),
            "semantic_centers": semantic_centers,
            "semantic_code_config": semantic_code_config,
            "observer_channels": carrier_weight[:, :, None].astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
            "complex_public": wave_state["public"].astype(np.float32),
            "complex_wave_real": wave_state["wave_real"].astype(np.float32),
            "complex_wave_imag": wave_state["wave_imag"].astype(np.float32),
            "complex_amp_hidden": wave_state["amp_hidden"].astype(np.float32),
            "projective_public": projective_state["public"].astype(np.float32),
            "projective_wave_real": projective_state["wave_real"].astype(np.float32),
            "projective_wave_imag": projective_state["wave_imag"].astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _encode(x, doc_top_k, collapse_scale, "docs")

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _encode(x, query_top_k, collapse_scale * collapse_gain, "queries")

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        wave_scores = wave_method.score(
            _carrier_state(doc_state, "complex"),
            _carrier_state(query_state, "complex"),
        )
        projective_scores = projective_method.score(
            _carrier_state(doc_state, "projective"),
            _carrier_state(query_state, "projective"),
        )
        pair_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qc,dc->qdc",
                    query_state["carrier_weight"],
                    doc_state["carrier_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        blend_weight = np.sum(pair_gate, axis=2)
        blend_weight = np.where(blend_weight == 0.0, 1.0, blend_weight)
        blended = (
            pair_gate[:, :, 0] * wave_scores + pair_gate[:, :, 1] * projective_scores
        ) / blend_weight
        agreement = np.sqrt(np.maximum(wave_scores * projective_scores, 0.0))
        return (1.0 - agreement_weight) * blended + agreement_weight * agreement

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        wave_scores = wave_method.score(
            _carrier_state(doc_state, "complex"),
            _carrier_state(query_state, "complex"),
        )
        projective_scores = projective_method.score(
            _carrier_state(doc_state, "projective"),
            _carrier_state(query_state, "projective"),
        )
        wave_support = (
            query_state["carrier_weight"][:, :1] @ doc_state["carrier_weight"][:, :1].T
        )
        projective_support = (
            query_state["carrier_weight"][:, 1:2]
            @ doc_state["carrier_weight"][:, 1:2].T
        )
        aux_scores = (wave_support - projective_support) * (
            wave_scores - projective_scores
        )
        aux_scores = aux_scores + 0.20 * (
            query_state["collapse_features"] @ doc_state["collapse_features"].T
        )
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return (aux_scores / scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="dual_carrier_superposition_v0",
        family="state_dual_carrier_superposition",
        params={
            "dim": dim,
            "bands": float(bands),
            "summary_bands": float(summary_bands),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "collapse_temperature": collapse_temperature,
            "collapse_scale": collapse_scale,
            "collapse_gain": collapse_gain,
            "collapse_floor": collapse_floor,
            "agreement_weight": agreement_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
            "wave_phase_scale": wave_params["phase_scale"],
            "wave_envelope_gain": wave_params["envelope_gain"],
            "projective_phase_scale": projective_params["phase_scale"],
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _spin_glass_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    spins = max(3, int(params.get("spins", 10)))
    doc_top_k = max(1, min(spins, int(params.get("doc_top_k", 4))))
    query_top_k = max(1, min(spins, int(params.get("query_top_k", 2))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.20)))
    route_scale = float(params.get("route_scale", 1.20))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 2.10)))
    phase_scale = float(params.get("phase_scale", 0.78))
    envelope_gain = float(params.get("envelope_gain", 0.44))
    frustration_gain = float(params.get("frustration_gain", 0.22))
    frustration_gain = min(max(frustration_gain, 0.0), 1.50)
    decoy_floor = float(params.get("decoy_floor", 0.22))
    decoy_floor = min(max(decoy_floor, 0.0), 0.85)
    coherence_weight = float(params.get("coherence_weight", 0.48))
    coherence_weight = min(max(coherence_weight, 0.0), 1.0)
    support_weight = float(params.get("support_weight", 0.14))
    support_weight = min(max(support_weight, 0.0), 0.40)
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.84))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("spin_glass_embedding_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    residual_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    route_bank = _safe_normalize(
        local_rng.normal(size=(spins, hidden_dim)).astype(np.float32)
    )
    route_bias = local_rng.uniform(-0.25, 0.25, size=(spins,)).astype(np.float32)
    spin_codes = local_rng.choice([-1.0, 1.0], size=(spins, hidden_dim)).astype(
        np.float32
    )
    mode_phase_shift = local_rng.uniform(
        -math.pi, math.pi, size=(spins, hidden_dim)
    ).astype(np.float32)
    coupling = local_rng.normal(size=(spins, spins)).astype(np.float32)
    coupling = 0.5 * (coupling + coupling.T)
    np.fill_diagonal(coupling, 0.0)
    coupling = coupling / max(1.0, np.sqrt(float(spins)))
    frustrated_codes = spin_codes + frustration_gain * (coupling @ spin_codes)
    frustrated_codes = _safe_normalize(frustrated_codes)
    mode_cos = np.cos(mode_phase_shift).astype(np.float32)
    mode_sin = np.sin(mode_phase_shift).astype(np.float32)
    public_mix = local_rng.normal(size=(spins, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray, top_k: int, scale: float) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        carrier_real = y @ real_proj
        carrier_imag = y @ imag_proj
        residual_hidden = _safe_normalize(np.tanh(y @ residual_proj))
        carrier_phase = phase_scale * carrier_real + phase_bias[None, :]
        carrier_amp = np.sqrt(
            np.maximum(1e-6, 1.0 + envelope_gain * np.tanh(carrier_imag))
        ).astype(np.float32)
        base_wave_real = carrier_amp * np.cos(carrier_phase)
        base_wave_imag = carrier_amp * np.sin(carrier_phase)
        base_norm = np.sqrt(
            np.sum(base_wave_real**2 + base_wave_imag**2, axis=1, keepdims=True)
        )
        base_norm = np.where(base_norm == 0.0, 1.0, base_norm)
        base_wave_real = base_wave_real / base_norm
        base_wave_imag = base_wave_imag / base_norm
        carrier_energy = (base_wave_real**2 + base_wave_imag**2).astype(np.float32)
        route_logits = carrier_energy @ route_bank.T
        route_logits = route_logits + 0.20 * (
            np.abs(residual_hidden) @ np.abs(route_bank).T
        )
        route_logits = route_logits + route_bias[None, :]
        route_logits = route_logits / math.sqrt(max(1.0, float(hidden_dim)))
        sparse_weight = _topk_soft_assign(
            route_logits * scale, top_k, route_temperature
        )
        mode_weight = decoy_floor / float(spins) + (1.0 - decoy_floor) * sparse_weight
        mode_total = np.sum(mode_weight, axis=1, keepdims=True)
        mode_total = np.where(mode_total == 0.0, 1.0, mode_total)
        mode_weight = mode_weight / mode_total
        local_field = route_logits + frustration_gain * (mode_weight @ coupling)
        glass_field = np.tanh(local_field).astype(np.float32)
        glass_weight = np.abs(glass_field)
        glass_total = np.sum(glass_weight, axis=1, keepdims=True)
        glass_total = np.where(glass_total == 0.0, 1.0, glass_total)
        glass_weight = glass_weight / glass_total
        spin_sign = np.where(glass_field >= 0.0, 1.0, -1.0).astype(np.float32)
        spin_scale = np.sqrt(np.maximum(glass_weight, 1e-6)).astype(np.float32)[
            :, :, None
        ]
        rotated_real = (
            base_wave_real[:, None, :] * mode_cos[None, :, :]
            - base_wave_imag[:, None, :] * mode_sin[None, :, :]
        )
        rotated_imag = (
            base_wave_real[:, None, :] * mode_sin[None, :, :]
            + base_wave_imag[:, None, :] * mode_cos[None, :, :]
        )
        wave_real = (
            spin_scale
            * spin_sign[:, :, None]
            * rotated_real
            * frustrated_codes[None, :, :]
        )
        wave_imag = spin_scale * rotated_imag * frustrated_codes[None, :, :]
        wave_norm = np.sqrt(
            np.sum(wave_real**2 + wave_imag**2, axis=(1, 2), keepdims=True)
        )
        wave_norm = np.where(wave_norm == 0.0, 1.0, wave_norm)
        wave_real = wave_real / wave_norm
        wave_imag = wave_imag / wave_norm
        glass_energy = glass_weight * np.maximum(local_field, 0.0)
        public_profile = mode_weight + 0.30 * glass_weight + 0.18 * np.abs(glass_field)
        public = _safe_normalize(np.tanh(public_profile) @ public_mix)
        public = _mask_public_observation(
            "spin_glass_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public.astype(np.float32),
            "base_wave_real": base_wave_real.astype(np.float32),
            "base_wave_imag": base_wave_imag.astype(np.float32),
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "mode_weight": mode_weight.astype(np.float32),
            "glass_weight": glass_weight.astype(np.float32),
            "glass_energy": glass_energy.astype(np.float32),
            "spin_sign": spin_sign.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _encode(x, doc_top_k, route_scale)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _encode(x, query_top_k, route_scale * collapse_gain)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier_overlap = query_state["base_wave_real"] @ doc_state["base_wave_real"].T
        carrier_overlap = carrier_overlap + (
            query_state["base_wave_imag"] @ doc_state["base_wave_imag"].T
        )
        spin_overlap = np.einsum(
            "qmh,dmh->qdm",
            query_state["wave_real"],
            doc_state["wave_real"],
            dtype=np.float32,
        )
        spin_overlap = spin_overlap + np.einsum(
            "qmh,dmh->qdm",
            query_state["wave_imag"],
            doc_state["wave_imag"],
            dtype=np.float32,
        )
        mode_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qm,dm->qdm",
                    query_state["glass_weight"],
                    doc_state["glass_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        polarity_gate = query_state["spin_sign"] @ doc_state["spin_sign"].T
        polarity_gate = polarity_gate / float(max(1, spins))
        polarity_gate = np.clip(0.5 + 0.5 * polarity_gate, 0.0, 1.0).astype(np.float32)
        glass_score = np.mean(
            (spin_overlap**2) * (0.25 + 0.75 * mode_gate), axis=2, dtype=np.float32
        )
        glass_score = glass_score * (0.25 + 0.75 * polarity_gate)
        mode_support = query_state["mode_weight"] @ doc_state["mode_weight"].T
        energy_support = query_state["glass_energy"] @ doc_state["glass_energy"].T
        support_score = (
            0.50 * mode_support + 0.30 * energy_support + 0.20 * polarity_gate
        )
        residual_weight = max(0.0, 1.0 - coherence_weight - support_weight)
        return (
            coherence_weight * glass_score
            + support_weight * support_score
            + residual_weight * (carrier_overlap**2)
        )

    return EmbeddingStateMethod(
        method_name="spin_glass_embedding_v0",
        family="state_spin_glass",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "spins": float(spins),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "frustration_gain": frustration_gain,
            "decoy_floor": decoy_floor,
            "coherence_weight": coherence_weight,
            "support_weight": support_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _spread_spectrum_carrier_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    carriers = max(2, int(params.get("carriers", 12)))
    doc_top_k = max(1, min(carriers, int(params.get("doc_top_k", 4))))
    query_top_k = max(1, min(carriers, int(params.get("query_top_k", 1))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.20)))
    route_scale = float(params.get("route_scale", 1.35))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 2.4)))
    chip_residual = float(params.get("chip_residual", 0.22))
    chip_residual = min(max(chip_residual, 0.0), 0.95)
    decoy_floor = float(params.get("decoy_floor", 0.18))
    decoy_floor = min(max(decoy_floor, 0.0), 0.85)
    match_weight = float(params.get("match_weight", 0.62))
    match_weight = min(max(match_weight, 0.05), 0.90)
    support_weight = float(params.get("support_weight", 0.16))
    support_weight = min(max(support_weight, 0.0), 0.40)
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.84))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("spread_spectrum_carrier_v0", secret_key, dim)
    )
    base_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    residual_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    route_bank = _safe_normalize(
        local_rng.normal(size=(carriers, hidden_dim)).astype(np.float32)
    )
    route_bias = local_rng.uniform(-0.25, 0.25, size=(carriers,)).astype(np.float32)
    chip_codes = local_rng.choice([-1.0, 1.0], size=(carriers, hidden_dim)).astype(
        np.float32
    )
    residual_codes = local_rng.choice([-1.0, 1.0], size=(carriers, hidden_dim)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(carriers, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray, top_k: int, scale: float) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        base_hidden = _safe_normalize(y @ base_proj)
        residual_hidden = _safe_normalize(np.tanh(y @ residual_proj))
        route_logits = (base_hidden @ route_bank.T + route_bias[None, :]) / math.sqrt(
            max(1.0, float(hidden_dim))
        )
        sparse_weight = _topk_soft_assign(
            route_logits * scale, top_k, route_temperature
        )
        mode_weight = (
            decoy_floor / float(carriers) + (1.0 - decoy_floor) * sparse_weight
        )
        mode_total = np.sum(mode_weight, axis=1, keepdims=True)
        mode_total = np.where(mode_total == 0.0, 1.0, mode_total)
        mode_weight = mode_weight / mode_total
        mode_scale = np.sqrt(np.maximum(mode_weight, 1e-6)).astype(np.float32)[
            :, :, None
        ]
        spread_main = base_hidden[:, None, :] * chip_codes[None, :, :]
        spread_residual = residual_hidden[:, None, :] * residual_codes[None, :, :]
        spread_modes = mode_scale * (spread_main + chip_residual * spread_residual)
        spread_norm = np.sqrt(np.sum(spread_modes**2, axis=(1, 2), keepdims=True))
        spread_norm = np.where(spread_norm == 0.0, 1.0, spread_norm)
        spread_modes = spread_modes / spread_norm
        broadcast_sum = np.sum(spread_modes, axis=1)
        broadcast_sum = _safe_normalize(broadcast_sum.astype(np.float32))
        chip_energy = np.maximum(route_logits, 0.0) * mode_weight
        public = _safe_normalize(np.tanh(mode_weight + 0.30 * chip_energy) @ public_mix)
        public = _mask_public_observation(
            "spread_spectrum_carrier_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public.astype(np.float32),
            "base_hidden": base_hidden.astype(np.float32),
            "residual_hidden": residual_hidden.astype(np.float32),
            "spread_modes": spread_modes.astype(np.float32),
            "broadcast_sum": broadcast_sum.astype(np.float32),
            "mode_weight": mode_weight.astype(np.float32),
            "chip_energy": chip_energy.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _encode(x, doc_top_k, route_scale)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _encode(x, query_top_k, route_scale * collapse_gain)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_overlap = query_state["base_hidden"] @ doc_state["base_hidden"].T
        spread_overlap = np.einsum(
            "qmh,dmh->qdm",
            query_state["spread_modes"],
            doc_state["spread_modes"],
            dtype=np.float32,
        )
        mode_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qm,dm->qdm",
                    query_state["mode_weight"],
                    doc_state["mode_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        matched_score = np.mean(
            (spread_overlap**2) * (0.30 + 0.70 * mode_gate), axis=2, dtype=np.float32
        )
        broadcast_overlap = query_state["broadcast_sum"] @ doc_state["broadcast_sum"].T
        mode_support = query_state["mode_weight"] @ doc_state["mode_weight"].T
        chip_support = query_state["chip_energy"] @ doc_state["chip_energy"].T
        residual_weight = max(0.0, 1.0 - match_weight - support_weight)
        return (
            match_weight * matched_score
            + support_weight
            * (
                0.45 * mode_support
                + 0.30 * chip_support
                + 0.25 * (broadcast_overlap**2)
            )
            + residual_weight * (base_overlap**2)
        )

    return EmbeddingStateMethod(
        method_name="spread_spectrum_carrier_v0",
        family="state_spread_spectrum",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "carriers": float(carriers),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "chip_residual": chip_residual,
            "decoy_floor": decoy_floor,
            "match_weight": match_weight,
            "support_weight": support_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _broadcast_noise_embedding_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    modes = max(2, int(params.get("modes", 10)))
    doc_top_k = max(1, min(modes, int(params.get("doc_top_k", 4))))
    query_top_k = max(1, min(modes, int(params.get("query_top_k", 1))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.20)))
    route_scale = float(params.get("route_scale", 1.28))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 2.5)))
    phase_scale = float(params.get("phase_scale", 0.78))
    envelope_gain = float(params.get("envelope_gain", 0.44))
    residual_gain = float(params.get("residual_gain", 0.38))
    residual_gain = min(max(residual_gain, 0.0), 1.5)
    broadcast_noise = float(params.get("broadcast_noise", 0.42))
    broadcast_noise = min(max(broadcast_noise, 0.0), 1.5)
    decoy_floor = float(params.get("decoy_floor", 0.28))
    decoy_floor = min(max(decoy_floor, 0.0), 0.85)
    coherence_weight = float(params.get("coherence_weight", 0.54))
    coherence_weight = min(max(coherence_weight, 0.0), 1.0)
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.84))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("broadcast_noise_embedding_v0", secret_key, dim)
    )
    route_bank = _safe_normalize(
        local_rng.normal(size=(modes, hidden_dim)).astype(np.float32)
    )
    route_bias = local_rng.uniform(-0.35, 0.35, size=(modes,)).astype(np.float32)
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    residual_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    mode_phase_shift = local_rng.uniform(
        -math.pi, math.pi, size=(modes, hidden_dim)
    ).astype(np.float32)
    noise_phase_shift = local_rng.uniform(
        -math.pi, math.pi, size=(modes, hidden_dim)
    ).astype(np.float32)
    noise_mix = _safe_normalize(
        local_rng.normal(size=(modes, hidden_dim)).astype(np.float32)
    )
    public_mix = local_rng.normal(size=(modes, public_dim)).astype(np.float32)
    mode_cos = np.cos(mode_phase_shift).astype(np.float32)
    mode_sin = np.sin(mode_phase_shift).astype(np.float32)
    noise_cos = np.cos(noise_phase_shift).astype(np.float32)
    noise_sin = np.sin(noise_phase_shift).astype(np.float32)

    def _encode(x: np.ndarray, top_k: int, scale: float) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        carrier_real = y @ real_proj
        carrier_imag = y @ imag_proj
        residual_hidden = _safe_normalize(np.tanh(y @ residual_proj))
        carrier_phase = phase_scale * carrier_real + phase_bias[None, :]
        carrier_amp = np.sqrt(
            np.maximum(1e-6, 1.0 + envelope_gain * np.tanh(carrier_imag))
        ).astype(np.float32)
        base_wave_real = carrier_amp * np.cos(carrier_phase)
        base_wave_imag = carrier_amp * np.sin(carrier_phase)
        base_norm = np.sqrt(
            np.sum(base_wave_real**2 + base_wave_imag**2, axis=1, keepdims=True)
        )
        base_norm = np.where(base_norm == 0.0, 1.0, base_norm)
        base_wave_real = base_wave_real / base_norm
        base_wave_imag = base_wave_imag / base_norm
        carrier_energy = (base_wave_real**2 + base_wave_imag**2).astype(np.float32)
        route_logits = (
            carrier_energy @ route_bank.T + route_bias[None, :]
        ) / math.sqrt(max(1.0, float(hidden_dim)))
        sparse_weight = _topk_soft_assign(
            route_logits * scale, top_k, route_temperature
        )
        mode_weight = decoy_floor / float(modes) + (1.0 - decoy_floor) * sparse_weight
        mode_total = np.sum(mode_weight, axis=1, keepdims=True)
        mode_total = np.where(mode_total == 0.0, 1.0, mode_total)
        mode_weight = mode_weight / mode_total
        mode_scale = np.sqrt(np.maximum(mode_weight, 1e-6)).astype(np.float32)[
            :, :, None
        ]
        rotated_real = (
            base_wave_real[:, None, :] * mode_cos[None, :, :]
            - base_wave_imag[:, None, :] * mode_sin[None, :, :]
        )
        rotated_imag = (
            base_wave_real[:, None, :] * mode_sin[None, :, :]
            + base_wave_imag[:, None, :] * mode_cos[None, :, :]
        )
        noise_logits = (
            carrier_energy @ noise_mix.T
            + residual_gain * (residual_hidden @ route_bank.T)
        ) / math.sqrt(max(1.0, float(hidden_dim)))
        noise_energy = mode_weight * np.tanh(np.abs(noise_logits)).astype(np.float32)
        noise_scale = (
            mode_scale
            * broadcast_noise
            * (0.35 + 0.65 * noise_energy.astype(np.float32)[:, :, None])
        )
        decoy_real = (
            residual_hidden[:, None, :] * noise_cos[None, :, :]
            - base_wave_real[:, None, :] * noise_sin[None, :, :]
        )
        decoy_imag = (
            residual_hidden[:, None, :] * noise_sin[None, :, :]
            + base_wave_real[:, None, :] * noise_cos[None, :, :]
        )
        wave_real = mode_scale * rotated_real + noise_scale * decoy_real
        wave_imag = mode_scale * rotated_imag + noise_scale * decoy_imag
        wave_norm = np.sqrt(
            np.sum(wave_real**2 + wave_imag**2, axis=(1, 2), keepdims=True)
        )
        wave_norm = np.where(wave_norm == 0.0, 1.0, wave_norm)
        wave_real = wave_real / wave_norm
        wave_imag = wave_imag / wave_norm
        mode_energy = mode_weight * np.maximum(route_logits, 0.0)
        public_profile = mode_weight + 0.28 * mode_energy + 0.34 * noise_energy
        public = _safe_normalize(np.tanh(public_profile) @ public_mix)
        public = _mask_public_observation(
            "broadcast_noise_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public.astype(np.float32),
            "base_wave_real": base_wave_real.astype(np.float32),
            "base_wave_imag": base_wave_imag.astype(np.float32),
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "mode_weight": mode_weight.astype(np.float32),
            "mode_energy": mode_energy.astype(np.float32),
            "noise_energy": noise_energy.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _encode(x, doc_top_k, route_scale)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _encode(x, query_top_k, route_scale * collapse_gain)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_overlap = query_state["base_wave_real"] @ doc_state["base_wave_real"].T
        base_overlap = base_overlap + (
            query_state["base_wave_imag"] @ doc_state["base_wave_imag"].T
        )
        component_overlap = np.einsum(
            "qmh,dmh->qdm",
            query_state["wave_real"],
            doc_state["wave_real"],
            dtype=np.float32,
        )
        component_overlap = component_overlap + np.einsum(
            "qmh,dmh->qdm",
            query_state["wave_imag"],
            doc_state["wave_imag"],
            dtype=np.float32,
        )
        mode_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qm,dm->qdm",
                    query_state["mode_weight"],
                    doc_state["mode_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        energy_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qm,dm->qdm",
                    query_state["mode_energy"],
                    doc_state["mode_energy"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        noise_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qm,dm->qdm",
                    query_state["noise_energy"],
                    doc_state["noise_energy"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        coherence = (component_overlap**2) * (
            0.25 + 0.55 * mode_gate + 0.10 * energy_gate + 0.10 * noise_gate
        )
        coherence_score = np.mean(coherence, axis=2, dtype=np.float32)
        mode_support = query_state["mode_weight"] @ doc_state["mode_weight"].T
        energy_support = query_state["mode_energy"] @ doc_state["mode_energy"].T
        noise_support = query_state["noise_energy"] @ doc_state["noise_energy"].T
        broadcast_support = (
            0.45 * mode_support + 0.30 * noise_support + 0.25 * energy_support
        )
        carrier_score = 0.80 * (base_overlap**2) + 0.20 * broadcast_support
        return (
            coherence_weight * coherence_score
            + (1.0 - coherence_weight) * carrier_score
        )

    return EmbeddingStateMethod(
        method_name="broadcast_noise_embedding_v0",
        family="state_broadcast_noise",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "modes": float(modes),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "residual_gain": residual_gain,
            "broadcast_noise": broadcast_noise,
            "decoy_floor": decoy_floor,
            "coherence_weight": coherence_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _matched_filter_embedding_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    filters = max(2, int(params.get("filters", 10)))
    doc_top_k = max(1, min(filters, int(params.get("doc_top_k", 4))))
    query_top_k = max(1, min(filters, int(params.get("query_top_k", 1))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.18)))
    route_scale = float(params.get("route_scale", 1.30))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 2.6)))
    phase_scale = float(params.get("phase_scale", 0.76))
    envelope_gain = float(params.get("envelope_gain", 0.44))
    residual_mix = float(params.get("residual_mix", 0.18))
    residual_mix = min(max(residual_mix, 0.0), 1.0)
    decoy_floor = float(params.get("decoy_floor", 0.14))
    decoy_floor = min(max(decoy_floor, 0.0), 0.85)
    resonance_weight = float(params.get("resonance_weight", 0.64))
    resonance_weight = min(max(resonance_weight, 0.05), 0.90)
    support_weight = float(params.get("support_weight", 0.14))
    support_weight = min(max(support_weight, 0.0), 0.40)
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.84))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("matched_filter_embedding_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    residual_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    coord = np.linspace(-1.0, 1.0, hidden_dim, dtype=np.float32)
    filter_centers = np.linspace(-1.0, 1.0, filters, dtype=np.float32)
    filter_width = max(0.18, 1.8 / float(max(2, filters)))
    filter_window = np.exp(
        -0.5 * ((coord[None, :] - filter_centers[:, None]) / filter_width) ** 2
    ).astype(np.float32)
    filter_window = _safe_normalize(filter_window)
    filter_phase_shift = local_rng.uniform(-math.pi, math.pi, size=(filters,)).astype(
        np.float32
    )
    filter_bias = local_rng.uniform(-0.20, 0.20, size=(filters,)).astype(np.float32)
    public_mix = local_rng.normal(size=(filters, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray, top_k: int, scale: float) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        residual_hidden = _safe_normalize(np.tanh(y @ residual_proj))
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + envelope_gain * np.tanh(imag))).astype(
            np.float32
        )
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        local_real = wave_real @ filter_window.T
        local_imag = wave_imag @ filter_window.T
        response_power = local_real**2 + local_imag**2
        residual_response = np.abs(residual_hidden @ filter_window.T).astype(np.float32)
        route_logits = (
            response_power + residual_mix * residual_response + filter_bias[None, :]
        )
        sparse_weight = _topk_soft_assign(
            route_logits * scale, top_k, route_temperature
        )
        mode_weight = decoy_floor / float(filters) + (1.0 - decoy_floor) * sparse_weight
        mode_total = np.sum(mode_weight, axis=1, keepdims=True)
        mode_total = np.where(mode_total == 0.0, 1.0, mode_total)
        mode_weight = mode_weight / mode_total
        phase_cos = np.cos(filter_phase_shift)[None, :]
        phase_sin = np.sin(filter_phase_shift)[None, :]
        matched_real = local_real * phase_cos - local_imag * phase_sin
        matched_imag = local_real * phase_sin + local_imag * phase_cos
        collapsed_scale = np.sqrt(np.maximum(mode_weight, 1e-6)).astype(np.float32)
        collapsed_real = collapsed_scale * matched_real
        collapsed_imag = collapsed_scale * matched_imag
        filter_energy = mode_weight * np.maximum(route_logits, 0.0)
        public_profile = (
            mode_weight
            + 0.22 * np.tanh(response_power)
            + 0.12 * np.tanh(residual_response)
        )
        public = _safe_normalize(np.tanh(public_profile) @ public_mix)
        public = _mask_public_observation(
            "matched_filter_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public.astype(np.float32),
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "collapsed_real": collapsed_real.astype(np.float32),
            "collapsed_imag": collapsed_imag.astype(np.float32),
            "mode_weight": mode_weight.astype(np.float32),
            "filter_energy": filter_energy.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _encode(x, doc_top_k, route_scale)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _encode(x, query_top_k, route_scale * collapse_gain)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier_overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        carrier_overlap = carrier_overlap + (
            query_state["wave_imag"] @ doc_state["wave_imag"].T
        )
        filter_overlap = np.einsum(
            "qf,df->qdf",
            query_state["collapsed_real"],
            doc_state["collapsed_real"],
            dtype=np.float32,
        )
        filter_overlap = filter_overlap + np.einsum(
            "qf,df->qdf",
            query_state["collapsed_imag"],
            doc_state["collapsed_imag"],
            dtype=np.float32,
        )
        mode_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qf,df->qdf",
                    query_state["mode_weight"],
                    doc_state["mode_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        energy_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qf,df->qdf",
                    query_state["filter_energy"],
                    doc_state["filter_energy"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        resonance = np.mean(
            (filter_overlap**2) * (0.30 + 0.55 * mode_gate + 0.15 * energy_gate),
            axis=2,
            dtype=np.float32,
        )
        mode_support = query_state["mode_weight"] @ doc_state["mode_weight"].T
        energy_support = query_state["filter_energy"] @ doc_state["filter_energy"].T
        residual_weight = max(0.0, 1.0 - resonance_weight - support_weight)
        return (
            resonance_weight * resonance
            + support_weight * (0.65 * mode_support + 0.35 * energy_support)
            + residual_weight * (carrier_overlap**2)
        )

    return EmbeddingStateMethod(
        method_name="matched_filter_embedding_v0",
        family="state_matched_filter_collapse",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "filters": float(filters),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "residual_mix": residual_mix,
            "decoy_floor": decoy_floor,
            "resonance_weight": resonance_weight,
            "support_weight": support_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _keyed_collapse_carrier_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    eigenspaces = max(2, int(params.get("eigenspaces", 8)))
    default_subspace_dim = max(2, hidden_dim // eigenspaces)
    subspace_dim = max(2, int(params.get("subspace_dim", default_subspace_dim)))
    eigenspaces = max(2, min(eigenspaces, hidden_dim // subspace_dim))
    active_dim = eigenspaces * subspace_dim
    doc_top_k = max(1, min(eigenspaces, int(params.get("doc_top_k", 3))))
    query_top_k = max(1, min(eigenspaces, int(params.get("query_top_k", 1))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.18)))
    route_scale = float(params.get("route_scale", 1.22))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 2.6)))
    phase_scale = float(params.get("phase_scale", 0.78))
    envelope_gain = float(params.get("envelope_gain", 0.44))
    residual_mix = float(params.get("residual_mix", 0.16))
    residual_mix = min(max(residual_mix, 0.0), 1.0)
    decoy_floor = float(params.get("decoy_floor", 0.18))
    decoy_floor = min(max(decoy_floor, 0.0), 0.85)
    coherence_weight = float(params.get("coherence_weight", 0.58))
    coherence_weight = min(max(coherence_weight, 0.0), 1.0)
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.84))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("keyed_collapse_carrier_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    residual_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    eigenspace_basis = _qr_orthogonal(local_rng, hidden_dim, hidden_dim)[:, :active_dim]
    eigenspace_basis = eigenspace_basis.reshape(hidden_dim, eigenspaces, subspace_dim)
    eigenspace_basis = np.transpose(eigenspace_basis, (1, 2, 0)).astype(np.float32)
    eigenspace_phase = local_rng.uniform(-math.pi, math.pi, size=(eigenspaces,)).astype(
        np.float32
    )
    eigenspace_bias = local_rng.uniform(-0.20, 0.20, size=(eigenspaces,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(eigenspaces, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray, top_k: int, scale: float) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        carrier_real = y @ real_proj
        carrier_imag = y @ imag_proj
        residual_hidden = _safe_normalize(np.tanh(y @ residual_proj))
        carrier_phase = phase_scale * carrier_real + phase_bias[None, :]
        carrier_amp = np.sqrt(
            np.maximum(1e-6, 1.0 + envelope_gain * np.tanh(carrier_imag))
        ).astype(np.float32)
        base_wave_real = carrier_amp * np.cos(carrier_phase)
        base_wave_imag = carrier_amp * np.sin(carrier_phase)
        base_norm = np.sqrt(
            np.sum(base_wave_real**2 + base_wave_imag**2, axis=1, keepdims=True)
        )
        base_norm = np.where(base_norm == 0.0, 1.0, base_norm)
        base_wave_real = base_wave_real / base_norm
        base_wave_imag = base_wave_imag / base_norm
        eig_real = np.einsum(
            "nh,skh->nsk", base_wave_real, eigenspace_basis, dtype=np.float32
        )
        eig_imag = np.einsum(
            "nh,skh->nsk", base_wave_imag, eigenspace_basis, dtype=np.float32
        )
        residual_eig = np.einsum(
            "nh,skh->nsk", residual_hidden, eigenspace_basis, dtype=np.float32
        )
        eig_energy = np.sum(eig_real**2 + eig_imag**2, axis=2, dtype=np.float32)
        residual_support = np.mean(np.abs(residual_eig), axis=2, dtype=np.float32)
        route_logits = (
            eig_energy + residual_mix * residual_support + eigenspace_bias[None, :]
        )
        sparse_weight = _topk_soft_assign(
            route_logits * scale, top_k, route_temperature
        )
        mode_weight = (
            decoy_floor / float(eigenspaces) + (1.0 - decoy_floor) * sparse_weight
        )
        mode_total = np.sum(mode_weight, axis=1, keepdims=True)
        mode_total = np.where(mode_total == 0.0, 1.0, mode_total)
        mode_weight = mode_weight / mode_total
        phase_cos = np.cos(eigenspace_phase)[None, :, None]
        phase_sin = np.sin(eigenspace_phase)[None, :, None]
        collapse_scale = np.sqrt(np.maximum(mode_weight, 1e-6)).astype(np.float32)[
            :, :, None
        ]
        collapsed_real = collapse_scale * (eig_real * phase_cos - eig_imag * phase_sin)
        collapsed_imag = collapse_scale * (eig_real * phase_sin + eig_imag * phase_cos)
        collapse_energy = mode_weight * np.maximum(route_logits, 0.0)
        public_profile = (
            mode_weight + 0.30 * np.tanh(eig_energy) + 0.14 * np.tanh(residual_support)
        )
        public = _safe_normalize(np.tanh(public_profile) @ public_mix)
        public = _mask_public_observation(
            "keyed_collapse_carrier_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public.astype(np.float32),
            "base_wave_real": base_wave_real.astype(np.float32),
            "base_wave_imag": base_wave_imag.astype(np.float32),
            "collapsed_real": collapsed_real.astype(np.float32),
            "collapsed_imag": collapsed_imag.astype(np.float32),
            "mode_weight": mode_weight.astype(np.float32),
            "collapse_energy": collapse_energy.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _encode(x, doc_top_k, route_scale)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _encode(x, query_top_k, route_scale * collapse_gain)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier_overlap = query_state["base_wave_real"] @ doc_state["base_wave_real"].T
        carrier_overlap = carrier_overlap + (
            query_state["base_wave_imag"] @ doc_state["base_wave_imag"].T
        )
        collapse_overlap = np.einsum(
            "qsk,dsk->qds",
            query_state["collapsed_real"],
            doc_state["collapsed_real"],
            dtype=np.float32,
        )
        collapse_overlap = collapse_overlap + np.einsum(
            "qsk,dsk->qds",
            query_state["collapsed_imag"],
            doc_state["collapsed_imag"],
            dtype=np.float32,
        )
        mode_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qs,ds->qds",
                    query_state["mode_weight"],
                    doc_state["mode_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        energy_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qs,ds->qds",
                    query_state["collapse_energy"],
                    doc_state["collapse_energy"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        collapse_score = np.mean(
            (collapse_overlap**2) * (0.25 + 0.60 * mode_gate + 0.15 * energy_gate),
            axis=2,
            dtype=np.float32,
        )
        mode_support = query_state["mode_weight"] @ doc_state["mode_weight"].T
        energy_support = query_state["collapse_energy"] @ doc_state["collapse_energy"].T
        carrier_score = 0.80 * (carrier_overlap**2) + 0.20 * (
            0.60 * mode_support + 0.40 * energy_support
        )
        return (
            coherence_weight * collapse_score + (1.0 - coherence_weight) * carrier_score
        )

    return EmbeddingStateMethod(
        method_name="keyed_collapse_carrier_v0",
        family="state_keyed_collapse",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "eigenspaces": float(eigenspaces),
            "subspace_dim": float(subspace_dim),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "residual_mix": residual_mix,
            "decoy_floor": decoy_floor,
            "coherence_weight": coherence_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _fountain_code_carrier_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    shards = max(4, int(params.get("shards", 18)))
    density = float(params.get("density", 0.28))
    density = min(max(density, 0.08), 0.90)
    doc_top_k = max(1, min(shards, int(params.get("doc_top_k", 6))))
    query_top_k = max(1, min(shards, int(params.get("query_top_k", 3))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.20)))
    route_scale = float(params.get("route_scale", 1.10))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 1.8)))
    phase_scale = float(params.get("phase_scale", 0.78))
    envelope_gain = float(params.get("envelope_gain", 0.44))
    residual_mix = float(params.get("residual_mix", 0.16))
    residual_mix = min(max(residual_mix, 0.0), 1.0)
    decoy_floor = float(params.get("decoy_floor", 0.24))
    decoy_floor = min(max(decoy_floor, 0.0), 0.85)
    decode_weight = float(params.get("decode_weight", 0.48))
    decode_weight = min(max(decode_weight, 0.0), 1.0)
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.84))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("fountain_code_carrier_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    residual_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    shard_mix = local_rng.binomial(1, density, size=(shards, hidden_dim)).astype(
        np.float32
    )
    shard_sign = local_rng.choice([-1.0, 1.0], size=(shards, hidden_dim)).astype(
        np.float32
    )
    shard_mix = shard_mix * shard_sign
    zero_rows = np.sum(np.abs(shard_mix), axis=1, keepdims=True) == 0.0
    shard_mix = np.where(zero_rows, 1.0, shard_mix)
    shard_mix = _safe_normalize(shard_mix)
    shard_bias = local_rng.uniform(-0.18, 0.18, size=(shards,)).astype(np.float32)
    public_mix = local_rng.normal(size=(shards, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray, top_k: int, scale: float) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        carrier_real = y @ real_proj
        carrier_imag = y @ imag_proj
        residual_hidden = _safe_normalize(np.tanh(y @ residual_proj))
        carrier_phase = phase_scale * carrier_real + phase_bias[None, :]
        carrier_amp = np.sqrt(
            np.maximum(1e-6, 1.0 + envelope_gain * np.tanh(carrier_imag))
        ).astype(np.float32)
        base_wave_real = carrier_amp * np.cos(carrier_phase)
        base_wave_imag = carrier_amp * np.sin(carrier_phase)
        base_norm = np.sqrt(
            np.sum(base_wave_real**2 + base_wave_imag**2, axis=1, keepdims=True)
        )
        base_norm = np.where(base_norm == 0.0, 1.0, base_norm)
        base_wave_real = base_wave_real / base_norm
        base_wave_imag = base_wave_imag / base_norm
        base_hidden = _safe_normalize(base_wave_real + 0.35 * base_wave_imag)
        fragment_value = base_hidden @ shard_mix.T
        residual_fragment = residual_hidden @ shard_mix.T
        route_logits = (
            np.abs(fragment_value)
            + residual_mix * np.abs(residual_fragment)
            + shard_bias[None, :]
        )
        sparse_weight = _topk_soft_assign(
            route_logits * scale, top_k, route_temperature
        )
        shard_weight = decoy_floor / float(shards) + (1.0 - decoy_floor) * sparse_weight
        shard_total = np.sum(shard_weight, axis=1, keepdims=True)
        shard_total = np.where(shard_total == 0.0, 1.0, shard_total)
        shard_weight = shard_weight / shard_total
        shard_scale = np.sqrt(np.maximum(shard_weight, 1e-6)).astype(np.float32)
        packet = shard_scale * fragment_value
        reassembled = packet @ shard_mix
        reassembled = _safe_normalize(reassembled + 0.22 * base_hidden)
        shard_energy = shard_weight * np.maximum(route_logits, 0.0)
        public_profile = (
            shard_weight
            + 0.18 * np.tanh(np.abs(fragment_value))
            + 0.10 * np.tanh(np.abs(residual_fragment))
        )
        public = _safe_normalize(np.tanh(public_profile) @ public_mix)
        public = _mask_public_observation(
            "fountain_code_carrier_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public.astype(np.float32),
            "base_hidden": base_hidden.astype(np.float32),
            "packet": packet.astype(np.float32),
            "shard_weight": shard_weight.astype(np.float32),
            "shard_energy": shard_energy.astype(np.float32),
            "reassembled": reassembled.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _encode(x, doc_top_k, route_scale)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _encode(x, query_top_k, route_scale * collapse_gain)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_overlap = query_state["base_hidden"] @ doc_state["base_hidden"].T
        reassembly_overlap = query_state["reassembled"] @ doc_state["reassembled"].T
        packet_overlap = np.einsum(
            "qs,ds->qds",
            query_state["packet"],
            doc_state["packet"],
            dtype=np.float32,
        )
        weight_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qs,ds->qds",
                    query_state["shard_weight"],
                    doc_state["shard_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        energy_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qs,ds->qds",
                    query_state["shard_energy"],
                    doc_state["shard_energy"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        decode_score = np.mean(
            np.abs(packet_overlap) * (0.25 + 0.55 * weight_gate + 0.20 * energy_gate),
            axis=2,
            dtype=np.float32,
        )
        shard_support = query_state["shard_weight"] @ doc_state["shard_weight"].T
        energy_support = query_state["shard_energy"] @ doc_state["shard_energy"].T
        carrier_score = 0.70 * (base_overlap**2) + 0.30 * (reassembly_overlap**2)
        decode_support = (
            0.55 * decode_score + 0.25 * shard_support + 0.20 * energy_support
        )
        return decode_weight * decode_support + (1.0 - decode_weight) * carrier_score

    return EmbeddingStateMethod(
        method_name="fountain_code_carrier_v0",
        family="state_fountain_code",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "shards": float(shards),
            "density": density,
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "residual_mix": residual_mix,
            "decoy_floor": decoy_floor,
            "decode_weight": decode_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _ecc_syndrome_embedding_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    parity_checks = max(4, int(params.get("parity_checks", 16)))
    doc_top_k = max(1, min(parity_checks, int(params.get("doc_top_k", 6))))
    query_top_k = max(1, min(parity_checks, int(params.get("query_top_k", 4))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.22)))
    route_scale = float(params.get("route_scale", 1.05))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 1.5)))
    phase_scale = float(params.get("phase_scale", 0.78))
    envelope_gain = float(params.get("envelope_gain", 0.44))
    residual_mix = float(params.get("residual_mix", 0.12))
    residual_mix = min(max(residual_mix, 0.0), 1.0)
    decoy_floor = float(params.get("decoy_floor", 0.30))
    decoy_floor = min(max(decoy_floor, 0.0), 0.85)
    syndrome_gain = float(params.get("syndrome_gain", 0.20))
    syndrome_gain = min(max(syndrome_gain, 0.0), 1.5)
    decode_weight = float(params.get("decode_weight", 0.24))
    decode_weight = min(max(decode_weight, 0.0), 1.0)
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.84))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("ecc_syndrome_embedding_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    residual_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    parity_bank = local_rng.choice(
        [-1.0, 1.0], size=(parity_checks, hidden_dim)
    ).astype(np.float32)
    parity_bank = _safe_normalize(parity_bank)
    parity_bias = local_rng.uniform(-0.15, 0.15, size=(parity_checks,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(parity_checks, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray, top_k: int, scale: float) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        carrier_real = y @ real_proj
        carrier_imag = y @ imag_proj
        residual_hidden = _safe_normalize(np.tanh(y @ residual_proj))
        carrier_phase = phase_scale * carrier_real + phase_bias[None, :]
        carrier_amp = np.sqrt(
            np.maximum(1e-6, 1.0 + envelope_gain * np.tanh(carrier_imag))
        ).astype(np.float32)
        base_wave_real = carrier_amp * np.cos(carrier_phase)
        base_wave_imag = carrier_amp * np.sin(carrier_phase)
        base_norm = np.sqrt(
            np.sum(base_wave_real**2 + base_wave_imag**2, axis=1, keepdims=True)
        )
        base_norm = np.where(base_norm == 0.0, 1.0, base_norm)
        base_wave_real = base_wave_real / base_norm
        base_wave_imag = base_wave_imag / base_norm
        base_hidden = _safe_normalize(base_wave_real + 0.35 * base_wave_imag)
        syndrome_value = base_hidden @ parity_bank.T
        residual_syndrome = residual_hidden @ parity_bank.T
        route_logits = (
            np.abs(syndrome_value)
            + residual_mix * np.abs(residual_syndrome)
            + parity_bias[None, :]
        )
        sparse_weight = _topk_soft_assign(
            route_logits * scale, top_k, route_temperature
        )
        syndrome_weight = (
            decoy_floor / float(parity_checks) + (1.0 - decoy_floor) * sparse_weight
        )
        weight_total = np.sum(syndrome_weight, axis=1, keepdims=True)
        weight_total = np.where(weight_total == 0.0, 1.0, weight_total)
        syndrome_weight = syndrome_weight / weight_total
        syndrome_sign = np.sign(syndrome_value + 1e-6).astype(np.float32)
        weighted_syndrome = syndrome_weight * syndrome_sign
        repair = weighted_syndrome @ parity_bank
        repaired_hidden = _safe_normalize(base_hidden + syndrome_gain * repair)
        syndrome_energy = syndrome_weight * np.maximum(route_logits, 0.0)
        public_profile = (
            syndrome_weight
            + 0.16 * np.tanh(np.abs(syndrome_value))
            + 0.08 * np.tanh(np.abs(residual_syndrome))
        )
        public = _safe_normalize(np.tanh(public_profile) @ public_mix)
        public = _mask_public_observation(
            "ecc_syndrome_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public.astype(np.float32),
            "base_hidden": base_hidden.astype(np.float32),
            "weighted_syndrome": weighted_syndrome.astype(np.float32),
            "syndrome_weight": syndrome_weight.astype(np.float32),
            "syndrome_energy": syndrome_energy.astype(np.float32),
            "repaired_hidden": repaired_hidden.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _encode(x, doc_top_k, route_scale)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _encode(x, query_top_k, route_scale * collapse_gain)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_overlap = query_state["base_hidden"] @ doc_state["base_hidden"].T
        repaired_overlap = (
            query_state["repaired_hidden"] @ doc_state["repaired_hidden"].T
        )
        syndrome_overlap = np.einsum(
            "qs,ds->qds",
            query_state["weighted_syndrome"],
            doc_state["weighted_syndrome"],
            dtype=np.float32,
        )
        weight_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qs,ds->qds",
                    query_state["syndrome_weight"],
                    doc_state["syndrome_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        energy_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qs,ds->qds",
                    query_state["syndrome_energy"],
                    doc_state["syndrome_energy"],
                    dtype=np.float32,
                ),
                0.0,
            )
        )
        syndrome_score = np.mean(
            np.maximum(syndrome_overlap, 0.0)
            * (0.25 + 0.55 * weight_gate + 0.20 * energy_gate),
            axis=2,
            dtype=np.float32,
        )
        weight_support = query_state["syndrome_weight"] @ doc_state["syndrome_weight"].T
        energy_support = query_state["syndrome_energy"] @ doc_state["syndrome_energy"].T
        carrier_score = 0.78 * (base_overlap**2) + 0.22 * (repaired_overlap**2)
        syndrome_support = (
            0.55 * syndrome_score + 0.25 * weight_support + 0.20 * energy_support
        )
        return decode_weight * syndrome_support + (1.0 - decode_weight) * carrier_score

    return EmbeddingStateMethod(
        method_name="ecc_syndrome_embedding_v0",
        family="state_ecc_syndrome",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "parity_checks": float(parity_checks),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "phase_scale": phase_scale,
            "envelope_gain": envelope_gain,
            "residual_mix": residual_mix,
            "decoy_floor": decoy_floor,
            "syndrome_gain": syndrome_gain,
            "decode_weight": decode_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _dual_observer_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    branches = int(params.get("branches", 4))
    cross_weight = float(params.get("cross_weight", 0.58))
    public_ratio = float(params.get("public_ratio", 0.22))
    public_mask = float(params.get("public_mask", 0.70))
    public_chunk = int(params.get("public_chunk", 4))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("dual_observer_embedding_v0", secret_key, dim)
    )
    public_dim = max(8, int(dim * public_ratio))
    doc_proj = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(branches)])
    qry_proj = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(branches)])
    doc_obs = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(branches)])
    qry_obs = np.stack([_qr_orthogonal(local_rng, dim, dim) for _ in range(branches)])
    public_mix = np.stack(
        [
            local_rng.normal(size=(dim, max(2, public_dim // branches + 1))).astype(
                np.float32
            )
            for _ in range(branches)
        ]
    )

    def _encode_docs(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        content = []
        observer = []
        public_chunks = []
        for branch in range(branches):
            content_branch = _safe_normalize(y @ doc_proj[branch])
            observer_branch = _safe_normalize(np.tanh(y @ doc_obs[branch]))
            content.append(content_branch.astype(np.float32))
            observer.append(observer_branch.astype(np.float32))
            public_chunks.append(
                (np.abs(content_branch + observer_branch) @ public_mix[branch]).astype(
                    np.float32
                )
            )
        public = _safe_normalize(np.hstack(public_chunks))
        public = _mask_public_observation(
            "dual_observer_embedding_v0", secret_key, public, public_mask, public_chunk
        )
        return {
            "public": public,
            "content": np.stack(content, axis=1),
            "observer": np.stack(observer, axis=1),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        content = []
        observer = []
        public_chunks = []
        for branch in range(branches):
            content_branch = _safe_normalize(y @ qry_proj[branch])
            observer_branch = _safe_normalize(np.tanh(y @ qry_obs[branch]))
            content.append(content_branch.astype(np.float32))
            observer.append(observer_branch.astype(np.float32))
            public_chunks.append(
                (np.abs(content_branch + observer_branch) @ public_mix[branch]).astype(
                    np.float32
                )
            )
        public = _safe_normalize(np.hstack(public_chunks))
        public = _mask_public_observation(
            "dual_observer_embedding_v0", secret_key, public, public_mask, public_chunk
        )
        return {
            "public": public,
            "content": np.stack(content, axis=1),
            "observer": np.stack(observer, axis=1),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        sim = np.zeros(
            (query_state["public"].shape[0], doc_state["public"].shape[0]),
            dtype=np.float32,
        )
        for branch in range(branches):
            direct = (
                query_state["content"][:, branch] @ doc_state["content"][:, branch].T
            )
            cross = (
                query_state["content"][:, branch] @ doc_state["observer"][:, branch].T
            )
            cross = (
                cross
                + query_state["observer"][:, branch] @ doc_state["content"][:, branch].T
            )
            cross = cross / 2.0
            sim = sim + (1.0 - cross_weight) * direct + cross_weight * cross
        return sim / float(max(1, branches))

    return EmbeddingStateMethod(
        method_name="dual_observer_embedding_v0",
        family="state_dual_observer",
        params={
            "dim": dim,
            "branches": float(branches),
            "cross_weight": cross_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _hilbert_space_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    observable_gain = float(params.get("observable_gain", 0.38))
    public_ratio = float(params.get("public_ratio", 0.20))
    public_mask = float(params.get("public_mask", 0.74))
    public_chunk = int(params.get("public_chunk", 4))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("hilbert_space_embedding_v0", secret_key, dim)
    )
    basis = _qr_orthogonal(local_rng, dim, hidden_dim)
    observable_diag = np.abs(local_rng.normal(size=(hidden_dim,)).astype(np.float32))
    observable_diag = observable_diag / np.sum(observable_diag)
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        coeff = y @ basis
        coeff = _safe_normalize(np.tanh((1.0 + observable_gain) * coeff))
        public = _safe_normalize(np.abs(coeff) @ public_mix)
        public = _mask_public_observation(
            "hilbert_space_embedding_v0", secret_key, public, public_mask, public_chunk
        )
        return {"public": public, "coeff": coeff.astype(np.float32)}

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        weighted_query = query_state["coeff"] * observable_diag[None, :]
        return weighted_query @ doc_state["coeff"].T

    return EmbeddingStateMethod(
        method_name="hilbert_space_embedding_v0",
        family="state_hilbert",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "observable_gain": observable_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _projective_hilbert_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_hilbert_embedding_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            "projective_hilbert_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public,
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        overlap = overlap + query_state["wave_imag"] @ doc_state["wave_imag"].T
        return overlap**2

    return EmbeddingStateMethod(
        method_name="projective_hilbert_embedding_v0",
        family="state_projective_hilbert",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _projective_duality_bipartite_state_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_hilbert_embedding_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    role_count = max(8, int(params.get("role_count", 24)))
    role_top_k = min(role_count, max(1, int(params.get("role_top_k", 3))))
    role_temperature = max(1e-4, float(params.get("role_temperature", 0.10)))
    phase_mix = float(params.get("phase_mix", 0.24))
    bridge_mix = float(params.get("bridge_mix", 0.28))
    duality_gain = float(params.get("duality_gain", 0.020))
    uncertainty_width = float(params.get("uncertainty_width", 0.030))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    dual_rng = np.random.default_rng(
        _method_seed("projective_duality_bipartite_state_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    left_basis = _safe_normalize(
        dual_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_basis = _safe_normalize(
        dual_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_phase_basis = _safe_normalize(
        dual_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_phase_basis = _safe_normalize(
        dual_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    role_weight = np.abs(dual_rng.normal(size=(role_count,)).astype(np.float32))
    role_weight = role_weight / np.maximum(1e-6, np.sum(role_weight))
    role_permutation = dual_rng.permutation(role_count)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        energy = (wave_real**2 + wave_imag**2).astype(np.float32)
        public = _safe_normalize(energy @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        phase_left = _safe_normalize(np.maximum(wave_real, 0.0))
        phase_right = _safe_normalize(np.maximum(wave_imag, 0.0))
        left_logits = energy @ left_basis.T + phase_mix * (
            phase_left @ left_phase_basis.T
        )
        right_logits = energy @ right_basis.T + phase_mix * (
            phase_right @ right_phase_basis.T
        )
        left_profile = _topk_soft_assign(left_logits, role_top_k, role_temperature)
        right_profile = _topk_soft_assign(right_logits, role_top_k, role_temperature)
        bridge_logits = left_logits - right_logits[:, role_permutation]
        bridge_profile = _topk_soft_assign(
            bridge_logits,
            role_top_k,
            role_temperature,
        )
        return {
            "public": public,
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "left_profile": left_profile.astype(np.float32),
            "right_profile": right_profile.astype(np.float32),
            "bridge_profile": bridge_profile.astype(np.float32),
            "aux_operator_profile": bridge_profile.astype(np.float32),
        }

    def _carrier_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        overlap = overlap + query_state["wave_imag"] @ doc_state["wave_imag"].T
        return overlap**2

    def _duality_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        left_cross = np.maximum(
            (query_state["left_profile"] * role_weight[None, :])
            @ doc_state["right_profile"].T,
            0.0,
        )
        right_cross = np.maximum(
            (query_state["right_profile"] * role_weight[None, :])
            @ doc_state["left_profile"].T,
            0.0,
        )
        bridge_cross = np.maximum(
            (query_state["bridge_profile"] * role_weight[None, :])
            @ doc_state["bridge_profile"].T,
            0.0,
        )
        dual_core = 0.5 * (left_cross + right_cross)
        # Die Query-Dokument-Kante ist hier selbst Teil des Zustandsobjekts,
        # nicht nur ein spaeter lokaler Support-Reranker.
        return (1.0 - bridge_mix) * dual_core + bridge_mix * np.sqrt(
            np.maximum(dual_core * bridge_cross, 0.0)
        )

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(doc_state, query_state)
        duality = _duality_score(doc_state, query_state)
        duality = duality - np.mean(duality, axis=1, keepdims=True)
        signal_scale = np.max(np.abs(duality), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        duality = duality / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        rerank_gain = np.maximum(0.0, 1.0 + duality_gain * uncertainty_gate * duality)
        return carrier * rerank_gain

    return EmbeddingStateMethod(
        method_name="projective_duality_bipartite_state_v0",
        family="state_projective_duality_bipartite",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "role_count": float(role_count),
            "role_top_k": float(role_top_k),
            "role_temperature": role_temperature,
            "phase_mix": phase_mix,
            "bridge_mix": bridge_mix,
            "duality_gain": duality_gain,
            "uncertainty_width": uncertainty_width,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
        aux_score=_duality_score,
    )


def _projective_phase_codebook_superposition_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = base.encode_docs(np.zeros((1, dim), dtype=np.float32))[
        "wave_real"
    ].shape[1]
    modes = max(4, min(24, int(params.get("modes", 10))))
    doc_top_k = min(modes, max(1, int(params.get("doc_top_k", 4))))
    query_top_k = min(modes, max(1, int(params.get("query_top_k", 2))))
    decode_top_k = min(modes, max(1, int(params.get("decode_top_k", 2))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.12)))
    route_scale = float(params.get("route_scale", 1.10))
    collapse_gain = max(0.0, float(params.get("collapse_gain", 0.18)))
    recover_gain = min(1.0, max(0.0, float(params.get("recover_gain", 0.18))))
    phase_mix = min(max(float(params.get("phase_mix", 0.35)), 0.0), 1.5)
    code_mix = min(max(float(params.get("code_mix", 0.45)), 0.0), 1.5)
    collapse_floor = min(max(float(params.get("collapse_floor", 0.20)), 0.0), 0.80)
    uncertainty_width = float(params.get("uncertainty_width", 0.020))
    public_ratio = float(params.get("public_ratio", 0.14))
    public_mask = float(params.get("public_mask", 0.92))
    public_chunk = int(params.get("public_chunk", 8))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_phase_codebook_superposition_v0", secret_key, dim)
    )

    codebook = (
        np.abs(local_rng.normal(size=(modes, hidden_dim)).astype(np.float32)) + 0.05
    )
    codebook = codebook / np.maximum(1e-6, np.sum(codebook, axis=1, keepdims=True))
    code_phase = local_rng.uniform(-math.pi, math.pi, size=(modes, hidden_dim)).astype(
        np.float32
    )
    code_cos = np.cos(code_phase).astype(np.float32)
    code_sin = np.sin(code_phase).astype(np.float32)
    code_bias = local_rng.uniform(-0.12, 0.12, size=(modes,)).astype(np.float32)
    observer_gate = np.mean(codebook, axis=1).astype(np.float32)
    observer_gate = observer_gate / np.maximum(1e-6, np.sum(observer_gate))
    semantic_centers = np.eye(modes, dtype=np.float32)
    semantic_code_config = np.array(
        [float(query_top_k), route_temperature], dtype=np.float32
    )
    public_mix = local_rng.normal(size=(3 * hidden_dim + modes, public_dim)).astype(
        np.float32
    )

    def _normalize_channel_tensor(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=2, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return x / norm

    def _top_mode_mean(mode_scores: np.ndarray) -> np.ndarray:
        keep = min(decode_top_k, mode_scores.shape[2])
        split = mode_scores.shape[2] - keep
        if split <= 0:
            return np.mean(mode_scores, axis=2, dtype=np.float32)
        top_scores = np.partition(mode_scores, split, axis=2)[:, :, split:]
        return np.mean(top_scores, axis=2, dtype=np.float32)

    def _augment(state: StateMap, top_k: int, scale: float, recover: float) -> StateMap:
        wave_real = state["wave_real"]
        wave_imag = state["wave_imag"]
        energy = np.maximum(1e-6, wave_real**2 + wave_imag**2).astype(np.float32)
        energy = energy / np.maximum(1e-6, np.sum(energy, axis=1, keepdims=True))
        phase = np.arctan2(wave_imag, wave_real).astype(np.float32)

        phase_align = np.cos(phase[:, None, :] - code_phase[None, :, :]).astype(
            np.float32
        )
        phase_support = np.mean(
            energy[:, None, :] * (1.0 + phase_align), axis=2, dtype=np.float32
        )
        code_support = (energy @ codebook.T).astype(np.float32)
        route_logits = code_mix * code_support + phase_mix * phase_support
        route_logits = route_logits + code_bias[None, :]
        recover_mix = recover_gain * recover
        if recover_mix > 0.0:
            route_logits = (
                1.0 - recover_mix
            ) * route_logits + recover_mix * code_support
        local_temperature = max(1e-4, route_temperature * (1.0 - 0.30 * recover))
        mode_weight = _topk_soft_assign(route_logits * scale, top_k, local_temperature)
        mode_weight = (
            collapse_floor / float(modes) + (1.0 - collapse_floor) * mode_weight
        )
        mode_weight_sum = np.sum(mode_weight, axis=1, keepdims=True)
        mode_weight_sum = np.where(mode_weight_sum == 0.0, 1.0, mode_weight_sum)
        mode_weight = (mode_weight / mode_weight_sum).astype(np.float32)

        mode_scale = np.sqrt(np.maximum(mode_weight, 1e-6)).astype(np.float32)[
            :, :, None
        ]
        code_envelope = np.sqrt(np.maximum(codebook[None, :, :], 1e-6)).astype(
            np.float32
        )
        energy_envelope = np.sqrt(np.maximum(energy[:, None, :], 1e-6)).astype(
            np.float32
        )
        rotated_real = wave_real[:, None, :] * code_cos[None, :, :]
        rotated_real = rotated_real - wave_imag[:, None, :] * code_sin[None, :, :]
        rotated_imag = wave_real[:, None, :] * code_sin[None, :, :]
        rotated_imag = rotated_imag + wave_imag[:, None, :] * code_cos[None, :, :]
        super_real = 0.78 * rotated_real * code_envelope
        super_real = super_real + 0.22 * energy_envelope * code_cos[None, :, :]
        super_imag = 0.78 * rotated_imag * code_envelope
        super_imag = super_imag + 0.22 * energy_envelope * code_sin[None, :, :]
        super_real = mode_scale * super_real
        super_imag = mode_scale * super_imag
        super_norm = np.sqrt(
            np.sum(super_real**2 + super_imag**2, axis=2, keepdims=True)
        ).astype(np.float32)
        super_norm = np.where(super_norm == 0.0, 1.0, super_norm)
        super_real = super_real / super_norm
        super_imag = super_imag / super_norm

        code_signature = np.sum(
            mode_weight[:, :, None] * codebook[None, :, :], axis=1, dtype=np.float32
        )
        phase_signature = np.sum(
            mode_weight[:, :, None] * phase_align, axis=1, dtype=np.float32
        )
        phase_signature = np.tanh(phase_signature).astype(np.float32)
        sample_order = np.argsort(code_signature + 0.25 * phase_signature, axis=1)
        sorted_energy = np.take_along_axis(energy, sample_order, axis=1)
        sorted_code = np.take_along_axis(code_signature, sample_order, axis=1)
        sorted_phase = np.take_along_axis(phase_signature, sample_order, axis=1)
        cumulative_energy = np.cumsum(sorted_energy, axis=1).astype(np.float32)
        public_context = np.hstack(
            [cumulative_energy, sorted_code, sorted_phase, mode_weight]
        ).astype(np.float32)
        public = _safe_normalize(np.tanh(public_context @ public_mix))
        public = _mask_public_observation(
            "projective_phase_codebook_superposition_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )

        mode_energy = mode_weight * np.maximum(
            route_logits - np.min(route_logits, axis=1, keepdims=True), 0.0
        )
        aux_profile = _safe_normalize(
            np.hstack([mode_weight, mode_energy, phase_support]).astype(np.float32)
        )
        observer_hidden = _normalize_channel_tensor(
            np.concatenate([super_real, super_imag], axis=2).astype(np.float32)
        )
        return {
            **state,
            "public": public.astype(np.float32),
            "energy": mode_weight.astype(np.float32),
            "mode_weight": mode_weight.astype(np.float32),
            "mode_energy": mode_energy.astype(np.float32),
            "mode_phase_support": phase_support.astype(np.float32),
            "mode_code_support": code_support.astype(np.float32),
            "superpose_real": super_real.astype(np.float32),
            "superpose_imag": super_imag.astype(np.float32),
            "semantic_codes": mode_weight.astype(np.float32),
            "semantic_centers": semantic_centers,
            "semantic_code_config": semantic_code_config,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _augment(base.encode_docs(x), doc_top_k, route_scale, 0.0)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _augment(base.encode_queries(x), query_top_k, route_scale, 1.0)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base.score(doc_state, query_state)
        component_overlap = np.einsum(
            "qmh,dmh->qdm",
            query_state["superpose_real"],
            doc_state["superpose_real"],
            dtype=np.float32,
        )
        component_overlap = component_overlap + np.einsum(
            "qmh,dmh->qdm",
            query_state["superpose_imag"],
            doc_state["superpose_imag"],
            dtype=np.float32,
        )
        mode_gate = np.sqrt(
            np.maximum(
                np.einsum(
                    "qm,dm->qdm",
                    query_state["mode_weight"],
                    doc_state["mode_weight"],
                    dtype=np.float32,
                ),
                0.0,
            )
        ).astype(np.float32)
        mode_scores = (component_overlap**2) * (0.25 + 0.75 * mode_gate)
        list_score = _top_mode_mean(mode_scores)
        semantic_match = query_state["mode_weight"] @ doc_state["mode_weight"].T
        phase_match = (
            query_state["mode_phase_support"] @ doc_state["mode_phase_support"].T
        ) / float(max(1, modes))
        code_match = (
            query_state["mode_code_support"] @ doc_state["mode_code_support"].T
        ) / float(max(1, modes))
        decode_score = 0.55 * list_score + 0.20 * semantic_match + 0.15 * phase_match
        decode_score = decode_score + 0.10 * code_match
        decode_score = decode_score - np.mean(decode_score, axis=1, keepdims=True)
        decode_scale = np.max(np.abs(decode_score), axis=1, keepdims=True)
        decode_scale = np.where(decode_scale == 0.0, 1.0, decode_scale)
        decode_score = decode_score / decode_scale
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        return carrier + collapse_gain * uncertainty_gate * decode_score

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        mode_gate = np.sqrt(
            np.maximum(
                query_state["mode_weight"] @ doc_state["mode_weight"].T,
                0.0,
            )
        ).astype(np.float32)
        phase_match = (
            query_state["mode_phase_support"] @ doc_state["mode_phase_support"].T
        ) / float(max(1, modes))
        code_match = (
            query_state["mode_code_support"] @ doc_state["mode_code_support"].T
        ) / float(max(1, modes))
        energy_match = (
            query_state["mode_energy"] @ doc_state["mode_energy"].T
        ) / float(max(1, modes))
        aux_scores = 0.40 * phase_match + 0.35 * code_match + 0.25 * energy_match
        aux_scores = aux_scores * (0.25 + 0.75 * mode_gate)
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        aux_scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        aux_scale = np.where(aux_scale == 0.0, 1.0, aux_scale)
        return (aux_scores / aux_scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_phase_codebook_superposition_v0",
        family="state_projective_phase_codebook_superposition",
        params={
            **base.params,
            "modes": float(modes),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "decode_top_k": float(decode_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "recover_gain": recover_gain,
            "phase_mix": phase_mix,
            "code_mix": code_mix,
            "collapse_floor": collapse_floor,
            "uncertainty_width": uncertainty_width,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _projective_resonance_window_decode_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = base.encode_docs(np.zeros((1, dim), dtype=np.float32))[
        "wave_real"
    ].shape[1]
    windows = max(4, min(24, int(params.get("windows", 8))))
    doc_top_k = min(windows, max(1, int(params.get("doc_top_k", 4))))
    query_top_k = min(windows, max(1, int(params.get("query_top_k", 2))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.14)))
    route_scale = float(params.get("route_scale", 1.0))
    resonance_gain = max(0.0, float(params.get("resonance_gain", 0.08)))
    recover_gain = min(1.0, max(0.0, float(params.get("recover_gain", 0.40))))
    energy_mix = min(max(float(params.get("energy_mix", 0.75)), 0.0), 2.0)
    phase_mix = min(max(float(params.get("phase_mix", 0.35)), 0.0), 2.0)
    decoy_penalty = min(max(float(params.get("decoy_penalty", 0.18)), 0.0), 1.0)
    window_floor = min(max(float(params.get("window_floor", 0.12)), 0.0), 0.80)
    window_width = max(1e-3, float(params.get("window_width", 0.025)))
    uncertainty_width = float(params.get("uncertainty_width", 0.015))
    public_ratio = float(params.get("public_ratio", 0.14))
    public_mask = float(params.get("public_mask", 0.92))
    public_chunk = int(params.get("public_chunk", 8))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_resonance_window_decode_v0", secret_key, dim)
    )

    coord = np.linspace(0.0, 1.0, hidden_dim, dtype=np.float32)
    centers = np.linspace(0.08, 0.92, windows, dtype=np.float32)
    centers = centers + local_rng.normal(0.0, 0.03, size=(windows,)).astype(np.float32)
    centers = np.clip(centers, 0.02, 0.98)
    window_bank = np.exp(
        -((coord[None, :] - centers[:, None]) ** 2) / max(1e-4, window_width)
    ).astype(np.float32)
    window_bank = window_bank / np.maximum(
        1e-6, np.sum(window_bank, axis=1, keepdims=True)
    )
    window_phase = local_rng.uniform(
        -math.pi, math.pi, size=(windows, hidden_dim)
    ).astype(np.float32)
    window_bias = local_rng.uniform(-0.10, 0.10, size=(windows,)).astype(np.float32)
    observer_gate = np.mean(window_bank, axis=1).astype(np.float32)
    observer_gate = observer_gate / np.maximum(1e-6, np.sum(observer_gate))
    semantic_centers = np.eye(windows, dtype=np.float32)
    semantic_code_config = np.array(
        [float(query_top_k), route_temperature], dtype=np.float32
    )
    public_mix = local_rng.normal(size=(3 * windows, public_dim)).astype(np.float32)

    def _normalize_channel_tensor(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=2, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return x / norm

    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        total = np.sum(x, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        return (x / total).astype(np.float32)

    def _augment(state: StateMap, top_k: int, recover: float) -> StateMap:
        wave_real = state["wave_real"]
        wave_imag = state["wave_imag"]
        energy = np.maximum(1e-6, wave_real**2 + wave_imag**2).astype(np.float32)
        energy = energy / np.maximum(1e-6, np.sum(energy, axis=1, keepdims=True))
        phase = np.arctan2(wave_imag, wave_real).astype(np.float32)
        phase_align = np.cos(phase[:, None, :] - window_phase[None, :, :]).astype(
            np.float32
        )
        window_energy = (energy @ window_bank.T).astype(np.float32)
        window_phase_support = np.mean(
            energy[:, None, :] * (1.0 + phase_align), axis=2, dtype=np.float32
        )
        route_logits = energy_mix * window_energy + phase_mix * window_phase_support
        route_logits = route_logits + window_bias[None, :]

        broad_weight = _topk_soft_assign(
            route_logits * route_scale,
            top_k,
            route_temperature,
        )
        broad_weight = (
            window_floor / float(windows) + (1.0 - window_floor) * broad_weight
        )
        broad_weight = _normalize_rows(broad_weight)

        focus_weight = broad_weight
        recover_mix = recover_gain * recover
        if recover_mix > 0.0:
            focus_logits = route_logits + recover_mix * (
                window_phase_support
                - np.mean(window_phase_support, axis=1, keepdims=True)
            )
            focus_weight = _topk_soft_assign(
                focus_logits * route_scale * (1.0 + 1.25 * recover_mix),
                top_k,
                max(1e-4, route_temperature * (1.0 - 0.55 * recover_mix)),
            )
            focus_weight = (
                window_floor / float(windows) + (1.0 - window_floor) * focus_weight
            )
            focus_weight = _normalize_rows(focus_weight)

        decoy_weight = np.maximum(broad_weight - focus_weight, 0.0).astype(np.float32)
        decoy_weight = decoy_weight + 1e-6 / float(windows)
        decoy_weight = _normalize_rows(decoy_weight)

        sample_order = np.argsort(window_energy + 0.20 * window_phase_support, axis=1)
        sorted_energy = np.take_along_axis(window_energy, sample_order, axis=1)
        sorted_phase = np.take_along_axis(window_phase_support, sample_order, axis=1)
        sorted_broad = np.take_along_axis(broad_weight, sample_order, axis=1)
        cumulative_energy = np.cumsum(sorted_energy, axis=1).astype(np.float32)
        public_context = np.hstack(
            [cumulative_energy, sorted_phase, sorted_broad]
        ).astype(np.float32)
        public = _safe_normalize(np.tanh(public_context @ public_mix))
        public = _mask_public_observation(
            "projective_resonance_window_decode_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )

        window_hidden = np.sqrt(
            np.maximum(energy[:, None, :] * window_bank[None, :, :], 1e-6)
        ).astype(np.float32)
        window_hidden = window_hidden * (0.35 + 0.65 * np.maximum(phase_align, 0.0))
        window_hidden = _normalize_channel_tensor(window_hidden)
        aux_profile = _safe_normalize(
            np.hstack([broad_weight, focus_weight, window_phase_support]).astype(
                np.float32
            )
        )
        return {
            **state,
            "public": public.astype(np.float32),
            "energy": focus_weight.astype(np.float32),
            "window_broad": broad_weight.astype(np.float32),
            "window_focus": focus_weight.astype(np.float32),
            "window_decoy": decoy_weight.astype(np.float32),
            "window_energy_support": window_energy.astype(np.float32),
            "window_phase_support": window_phase_support.astype(np.float32),
            "semantic_codes": focus_weight.astype(np.float32),
            "semantic_centers": semantic_centers,
            "semantic_code_config": semantic_code_config,
            "observer_channels": window_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _augment(base.encode_docs(x), doc_top_k, 0.0)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _augment(base.encode_queries(x), query_top_k, 1.0)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base.score(doc_state, query_state)
        focus_match = query_state["window_focus"] @ doc_state["window_broad"].T
        phase_match = (
            query_state["window_phase_support"] @ doc_state["window_phase_support"].T
        ) / float(max(1, windows))
        energy_match = (
            query_state["window_energy_support"] @ doc_state["window_energy_support"].T
        ) / float(max(1, windows))
        decoy_match = query_state["window_decoy"] @ doc_state["window_broad"].T
        resonance = 0.45 * focus_match + 0.30 * phase_match + 0.25 * energy_match
        resonance = resonance - decoy_penalty * decoy_match
        resonance = resonance - np.mean(resonance, axis=1, keepdims=True)
        resonance_scale = np.max(np.abs(resonance), axis=1, keepdims=True)
        resonance_scale = np.where(resonance_scale == 0.0, 1.0, resonance_scale)
        resonance = resonance / resonance_scale
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        return carrier + resonance_gain * uncertainty_gate * resonance

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        aux_scores = (
            0.55
            * (
                query_state["window_phase_support"]
                @ doc_state["window_phase_support"].T
            )
            / float(max(1, windows))
        )
        aux_scores = aux_scores + 0.45 * (
            query_state["window_focus"] @ doc_state["window_broad"].T
        )
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        aux_scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        aux_scale = np.where(aux_scale == 0.0, 1.0, aux_scale)
        return (aux_scores / aux_scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_resonance_window_decode_v0",
        family="state_projective_resonance_window_decode",
        params={
            **base.params,
            "windows": float(windows),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "resonance_gain": resonance_gain,
            "recover_gain": recover_gain,
            "energy_mix": energy_mix,
            "phase_mix": phase_mix,
            "decoy_penalty": decoy_penalty,
            "window_floor": window_floor,
            "window_width": window_width,
            "uncertainty_width": uncertainty_width,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _projective_pairwise_tournament_collapse_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = base.encode_docs(np.zeros((1, dim), dtype=np.float32))[
        "wave_real"
    ].shape[1]
    slots = max(6, min(32, int(params.get("slots", 12))))
    slot_top_k = min(slots, max(1, int(params.get("slot_top_k", 3))))
    slot_temperature = max(1e-4, float(params.get("slot_temperature", 0.16)))
    support_width = max(4, int(params.get("support_width", 8)))
    rerank_width = max(support_width, int(params.get("rerank_width", 16)))
    support_temperature = max(1e-4, float(params.get("support_temperature", 0.14)))
    support_mix = min(max(float(params.get("support_mix", 0.35)), 0.0), 1.0)
    tournament_temperature = max(
        1e-4, float(params.get("tournament_temperature", 0.10))
    )
    ambiguity_temperature = max(1e-4, float(params.get("ambiguity_temperature", 0.015)))
    collapse_gain = max(0.0, float(params.get("collapse_gain", 0.06)))
    uncertainty_width = float(params.get("uncertainty_width", 0.015))
    observer_channels = max(2, min(slots, int(params.get("observer_channels", 4))))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_pairwise_tournament_collapse_v0", secret_key, dim)
    )

    slot_basis = (
        np.abs(local_rng.normal(size=(slots, hidden_dim)).astype(np.float32)) + 0.05
    )
    slot_basis = slot_basis / np.maximum(
        1e-6, np.sum(slot_basis, axis=1, keepdims=True)
    )
    slot_phase = local_rng.uniform(-math.pi, math.pi, size=(slots, hidden_dim)).astype(
        np.float32
    )
    slot_cos = np.cos(slot_phase).astype(np.float32)
    slot_sin = np.sin(slot_phase).astype(np.float32)
    observer_masks = (
        np.abs(
            local_rng.normal(size=(observer_channels, hidden_dim)).astype(np.float32)
        )
        + 0.05
    )
    observer_masks = observer_masks / np.maximum(
        1e-6, np.sum(observer_masks, axis=1, keepdims=True)
    )
    observer_phase = local_rng.uniform(
        -math.pi, math.pi, size=(observer_channels, hidden_dim)
    ).astype(np.float32)
    observer_cos = np.cos(observer_phase).astype(np.float32)
    observer_sin = np.sin(observer_phase).astype(np.float32)
    observer_gate = np.mean(observer_masks, axis=1).astype(np.float32)
    observer_gate = observer_gate / np.maximum(1e-6, np.sum(observer_gate))
    semantic_centers = np.eye(slots, dtype=np.float32)
    semantic_code_config = np.array(
        [float(slot_top_k), slot_temperature], dtype=np.float32
    )

    def _normalize_channel_tensor(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=2, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return x / norm

    def _augment(state: StateMap) -> StateMap:
        wave_real = state["wave_real"]
        wave_imag = state["wave_imag"]
        energy = np.maximum(1e-6, wave_real**2 + wave_imag**2).astype(np.float32)
        energy = energy / np.maximum(1e-6, np.sum(energy, axis=1, keepdims=True))
        phase_support = np.mean(
            energy[:, None, :]
            * np.maximum(
                0.0,
                wave_real[:, None, :] * slot_cos[None, :, :]
                + wave_imag[:, None, :] * slot_sin[None, :, :],
            ),
            axis=2,
            dtype=np.float32,
        )
        slot_logits = energy @ slot_basis.T + 0.35 * phase_support
        semantic_codes = _topk_soft_assign(slot_logits, slot_top_k, slot_temperature)
        observer_hidden = np.sqrt(
            np.maximum(energy[:, None, :] * observer_masks[None, :, :], 1e-6)
        ).astype(np.float32)
        observer_signal = np.maximum(
            0.0,
            wave_real[:, None, :] * observer_cos[None, :, :]
            + wave_imag[:, None, :] * observer_sin[None, :, :],
        ).astype(np.float32)
        observer_hidden = _normalize_channel_tensor(
            observer_hidden * (0.35 + 0.65 * observer_signal)
        )
        aux_profile = _safe_normalize(
            np.hstack([semantic_codes, np.tanh(slot_logits)]).astype(np.float32)
        )
        return {
            **state,
            "energy": semantic_codes.astype(np.float32),
            "semantic_codes": semantic_codes.astype(np.float32),
            "semantic_centers": semantic_centers,
            "semantic_code_config": semantic_code_config,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _augment(base.encode_docs(x))

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _augment(base.encode_queries(x))

    def _relation_scores(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return np.maximum(
            query_state["semantic_codes"] @ doc_state["semantic_codes"].T,
            0.0,
        ).astype(np.float32)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base.score(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        relation_scores = _relation_scores(doc_state, query_state)
        relation_scale = relation_scores / np.maximum(
            1e-6, np.max(relation_scores, axis=1, keepdims=True)
        )
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_seed = carrier * (1.0 + support_mix * relation_scale)
        support_idx = np.argpartition(-support_seed, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_seed_scores = np.take_along_axis(support_seed, support_idx, axis=1)
        shifted = support_seed_scores - np.max(
            support_seed_scores, axis=1, keepdims=True
        )
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_profiles = doc_state["semantic_codes"][support_idx]
        support_wave_real = doc_state["wave_real"][support_idx]
        support_wave_imag = doc_state["wave_imag"][support_idx]
        support_context = np.sum(
            support_profiles * support_weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        support_context = _safe_normalize(support_context)

        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_profiles = doc_state["semantic_codes"][candidate_idx]
        query_match = np.sum(
            candidate_profiles * query_state["semantic_codes"][:, None, :],
            axis=2,
            dtype=np.float32,
        )
        support_match = np.sum(
            candidate_profiles * support_context[:, None, :],
            axis=2,
            dtype=np.float32,
        )
        candidate_signal = (
            1.0 - support_mix
        ) * query_match + support_mix * support_match
        pairwise_delta = candidate_signal[:, :, None] - candidate_signal[:, None, :]
        carrier_delta = candidate_carrier[:, :, None] - candidate_carrier[:, None, :]
        ambiguity_mask = np.exp(
            -np.abs(carrier_delta) / max(1e-4, ambiguity_temperature)
        ).astype(np.float32)
        pairwise_vote = np.tanh(pairwise_delta / tournament_temperature).astype(
            np.float32
        )
        pairwise_vote = pairwise_vote * ambiguity_mask
        if keep_rerank > 1:
            off_diag = 1.0 - np.eye(keep_rerank, dtype=np.float32)[None, :, :]
            pairwise_vote = pairwise_vote * off_diag
            tournament_signal = np.sum(pairwise_vote, axis=2, dtype=np.float32) / float(
                keep_rerank - 1
            )
        else:
            tournament_signal = np.zeros_like(candidate_carrier, dtype=np.float32)
        tournament_signal = tournament_signal - np.mean(
            tournament_signal, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(tournament_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        tournament_signal = tournament_signal / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        reranked = candidate_carrier + (
            collapse_gain * uncertainty_gate * candidate_carrier * tournament_signal
        )
        reranked = np.maximum(reranked, 0.0).astype(np.float32)
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        aux_scores = _relation_scores(doc_state, query_state)
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        aux_scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        aux_scale = np.where(aux_scale == 0.0, 1.0, aux_scale)
        return (aux_scores / aux_scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_pairwise_tournament_collapse_v0",
        family="state_projective_pairwise_tournament_collapse",
        params={
            **base.params,
            "slots": float(slots),
            "slot_top_k": float(slot_top_k),
            "slot_temperature": slot_temperature,
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "support_mix": support_mix,
            "tournament_temperature": tournament_temperature,
            "ambiguity_temperature": ambiguity_temperature,
            "collapse_gain": collapse_gain,
            "uncertainty_width": uncertainty_width,
            "observer_channels": float(observer_channels),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _projective_support_hypothesis_anchor_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = base.encode_docs(np.zeros((1, dim), dtype=np.float32))[
        "wave_real"
    ].shape[1]
    slots = max(6, min(32, int(params.get("slots", 10))))
    slot_top_k = min(slots, max(1, int(params.get("slot_top_k", 3))))
    slot_temperature = max(1e-4, float(params.get("slot_temperature", 0.18)))
    support_width = max(4, int(params.get("support_width", 6)))
    rerank_width = max(support_width, int(params.get("rerank_width", 12)))
    support_temperature = max(1e-4, float(params.get("support_temperature", 0.16)))
    support_mix = min(max(float(params.get("support_mix", 0.18)), 0.0), 1.0)
    hypotheses = max(2, min(slots, int(params.get("hypotheses", 3))))
    hypothesis_top_k = min(hypotheses, max(1, int(params.get("hypothesis_top_k", 2))))
    hypothesis_temperature = max(
        1e-4, float(params.get("hypothesis_temperature", 0.14))
    )
    query_mix = min(max(float(params.get("query_mix", 0.45)), 0.0), 1.0)
    collapse_gain = max(0.0, float(params.get("collapse_gain", 0.012)))
    uncertainty_width = float(params.get("uncertainty_width", 0.004))
    observer_channels = max(2, min(slots, int(params.get("observer_channels", 4))))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_support_hypothesis_anchor_v0", secret_key, dim)
    )

    slot_basis = (
        np.abs(local_rng.normal(size=(slots, hidden_dim)).astype(np.float32)) + 0.05
    )
    slot_basis = slot_basis / np.maximum(
        1e-6, np.sum(slot_basis, axis=1, keepdims=True)
    )
    slot_phase = local_rng.uniform(-math.pi, math.pi, size=(slots, hidden_dim)).astype(
        np.float32
    )
    slot_cos = np.cos(slot_phase).astype(np.float32)
    slot_sin = np.sin(slot_phase).astype(np.float32)
    hypothesis_basis = (
        np.abs(local_rng.normal(size=(hypotheses, slots)).astype(np.float32)) + 0.05
    )
    hypothesis_basis = hypothesis_basis / np.maximum(
        1e-6, np.sum(hypothesis_basis, axis=1, keepdims=True)
    )
    hypothesis_bias = local_rng.uniform(-0.08, 0.08, size=(hypotheses,)).astype(
        np.float32
    )
    observer_masks = (
        np.abs(
            local_rng.normal(size=(observer_channels, hidden_dim)).astype(np.float32)
        )
        + 0.05
    )
    observer_masks = observer_masks / np.maximum(
        1e-6, np.sum(observer_masks, axis=1, keepdims=True)
    )
    observer_phase = local_rng.uniform(
        -math.pi, math.pi, size=(observer_channels, hidden_dim)
    ).astype(np.float32)
    observer_cos = np.cos(observer_phase).astype(np.float32)
    observer_sin = np.sin(observer_phase).astype(np.float32)
    observer_gate = np.mean(observer_masks, axis=1).astype(np.float32)
    observer_gate = observer_gate / np.maximum(1e-6, np.sum(observer_gate))
    semantic_centers = np.eye(slots, dtype=np.float32)
    semantic_code_config = np.array(
        [float(slot_top_k), slot_temperature], dtype=np.float32
    )

    def _normalize_channel_tensor(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=2, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return x / norm

    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        total = np.sum(x, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        return (x / total).astype(np.float32)

    def _assign_hypotheses(logits: np.ndarray) -> np.ndarray:
        flat = logits.reshape(-1, logits.shape[-1])
        assigned = _topk_soft_assign(flat, hypothesis_top_k, hypothesis_temperature)
        return assigned.reshape(logits.shape).astype(np.float32)

    def _augment(state: StateMap) -> StateMap:
        wave_real = state["wave_real"]
        wave_imag = state["wave_imag"]
        energy = np.maximum(1e-6, wave_real**2 + wave_imag**2).astype(np.float32)
        energy = energy / np.maximum(1e-6, np.sum(energy, axis=1, keepdims=True))
        phase_support = np.mean(
            energy[:, None, :]
            * np.maximum(
                0.0,
                wave_real[:, None, :] * slot_cos[None, :, :]
                + wave_imag[:, None, :] * slot_sin[None, :, :],
            ),
            axis=2,
            dtype=np.float32,
        )
        slot_logits = energy @ slot_basis.T + 0.35 * phase_support
        semantic_codes = _topk_soft_assign(slot_logits, slot_top_k, slot_temperature)
        hypothesis_logits = (
            semantic_codes @ hypothesis_basis.T + hypothesis_bias[None, :]
        )
        hypothesis_profile = _topk_soft_assign(
            hypothesis_logits,
            hypothesis_top_k,
            hypothesis_temperature,
        )
        observer_hidden = np.sqrt(
            np.maximum(energy[:, None, :] * observer_masks[None, :, :], 1e-6)
        ).astype(np.float32)
        observer_signal = np.maximum(
            0.0,
            wave_real[:, None, :] * observer_cos[None, :, :]
            + wave_imag[:, None, :] * observer_sin[None, :, :],
        ).astype(np.float32)
        observer_hidden = _normalize_channel_tensor(
            observer_hidden * (0.35 + 0.65 * observer_signal)
        )
        aux_profile = _safe_normalize(
            np.hstack(
                [semantic_codes, hypothesis_profile, np.tanh(slot_logits)]
            ).astype(np.float32)
        )
        return {
            **state,
            "energy": semantic_codes.astype(np.float32),
            "semantic_codes": semantic_codes.astype(np.float32),
            "hypothesis_profile": hypothesis_profile.astype(np.float32),
            "semantic_centers": semantic_centers,
            "semantic_code_config": semantic_code_config,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _augment(base.encode_docs(x))

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _augment(base.encode_queries(x))

    def _relation_scores(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return np.maximum(
            query_state["semantic_codes"] @ doc_state["semantic_codes"].T,
            0.0,
        ).astype(np.float32)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base.score(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        relation_scores = _relation_scores(doc_state, query_state)
        relation_scale = relation_scores / np.maximum(
            1e-6, np.max(relation_scores, axis=1, keepdims=True)
        )
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_seed = carrier * (1.0 + support_mix * relation_scale)
        support_idx = np.argpartition(-support_seed, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_seed_scores = np.take_along_axis(support_seed, support_idx, axis=1)
        shifted = support_seed_scores - np.max(
            support_seed_scores, axis=1, keepdims=True
        )
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_profiles = doc_state["semantic_codes"][support_idx]
        support_wave_real = doc_state["wave_real"][support_idx]
        support_wave_imag = doc_state["wave_imag"][support_idx]
        support_context = np.sum(
            support_profiles * support_weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        support_context = _safe_normalize(support_context)
        query_anchor_logits = (
            query_state["semantic_codes"] @ hypothesis_basis.T
            + hypothesis_bias[None, :]
        )
        query_anchor_weight = _topk_soft_assign(
            query_anchor_logits,
            hypothesis_top_k,
            hypothesis_temperature,
        )
        support_anchor_logits = np.einsum(
            "qsk,hk->qsh",
            support_profiles,
            hypothesis_basis,
            dtype=np.float32,
        )
        support_anchor_logits = support_anchor_logits + hypothesis_bias[None, None, :]
        support_anchor_logits = (
            support_anchor_logits + 0.35 * query_anchor_logits[:, None, :]
        )
        support_anchor_weight = _assign_hypotheses(support_anchor_logits)
        weighted_anchor = support_anchor_weight * support_weights[:, :, None]
        local_anchor_mass = np.sum(weighted_anchor, axis=1, dtype=np.float32)
        local_anchor_mass = _normalize_rows(local_anchor_mass)
        hypothesis_state = np.einsum(
            "qsk,qsh->qhk",
            support_profiles,
            weighted_anchor,
            dtype=np.float32,
        )
        hypothesis_norm = np.sum(hypothesis_state, axis=2, keepdims=True)
        hypothesis_norm = np.where(hypothesis_norm == 0.0, 1.0, hypothesis_norm)
        hypothesis_state = (hypothesis_state / hypothesis_norm).astype(np.float32)
        hypothesis_wave_real = np.einsum(
            "qsd,qsh->qhd",
            support_wave_real,
            weighted_anchor,
            dtype=np.float32,
        )
        hypothesis_wave_imag = np.einsum(
            "qsd,qsh->qhd",
            support_wave_imag,
            weighted_anchor,
            dtype=np.float32,
        )
        hypothesis_wave_norm = np.sqrt(
            np.sum(
                hypothesis_wave_real**2 + hypothesis_wave_imag**2,
                axis=2,
                keepdims=True,
            )
        ).astype(np.float32)
        hypothesis_wave_norm = np.where(
            hypothesis_wave_norm == 0.0, 1.0, hypothesis_wave_norm
        )
        hypothesis_wave_real = (hypothesis_wave_real / hypothesis_wave_norm).astype(
            np.float32
        )
        hypothesis_wave_imag = (hypothesis_wave_imag / hypothesis_wave_norm).astype(
            np.float32
        )
        blended_anchor = (
            1.0 - query_mix
        ) * query_anchor_weight + query_mix * local_anchor_mass
        blended_anchor = _normalize_rows(blended_anchor)

        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_profiles = doc_state["semantic_codes"][candidate_idx]
        candidate_hypothesis = doc_state["hypothesis_profile"][candidate_idx]
        candidate_wave_real = doc_state["wave_real"][candidate_idx]
        candidate_wave_imag = doc_state["wave_imag"][candidate_idx]
        query_match = np.sum(
            candidate_profiles * query_state["semantic_codes"][:, None, :],
            axis=2,
            dtype=np.float32,
        )
        support_match = np.sum(
            candidate_profiles * support_context[:, None, :],
            axis=2,
            dtype=np.float32,
        )
        candidate_anchor_match = np.einsum(
            "qck,qhk->qch",
            candidate_profiles,
            hypothesis_state,
            dtype=np.float32,
        )
        profile_hypothesis_signal = np.sum(
            candidate_anchor_match * blended_anchor[:, None, :],
            axis=2,
            dtype=np.float32,
        )
        wave_anchor_match = np.einsum(
            "qcd,qhd->qch",
            candidate_wave_real,
            hypothesis_wave_real,
            dtype=np.float32,
        )
        wave_anchor_match = wave_anchor_match + np.einsum(
            "qcd,qhd->qch",
            candidate_wave_imag,
            hypothesis_wave_imag,
            dtype=np.float32,
        )
        wave_hypothesis_signal = np.sum(
            (wave_anchor_match**2) * blended_anchor[:, None, :],
            axis=2,
            dtype=np.float32,
        )
        profile_signal = np.sum(
            candidate_hypothesis * blended_anchor[:, None, :],
            axis=2,
            dtype=np.float32,
        )
        local_signal = 0.60 * wave_hypothesis_signal + 0.15 * profile_hypothesis_signal
        local_signal = local_signal + 0.15 * query_match + 0.10 * support_match
        local_signal = local_signal + 0.05 * profile_signal
        local_signal = local_signal - np.mean(local_signal, axis=1, keepdims=True)
        signal_scale = np.max(np.abs(local_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        local_signal = local_signal / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        reranked = candidate_carrier + (
            collapse_gain * uncertainty_gate * candidate_carrier * local_signal
        )
        reranked = np.maximum(reranked, 0.0).astype(np.float32)
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        aux_scores = (
            query_state["hypothesis_profile"] @ doc_state["hypothesis_profile"].T
        )
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        aux_scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        aux_scale = np.where(aux_scale == 0.0, 1.0, aux_scale)
        return (aux_scores / aux_scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_support_hypothesis_anchor_v0",
        family="state_projective_support_hypothesis_anchor",
        params={
            **base.params,
            "slots": float(slots),
            "slot_top_k": float(slot_top_k),
            "slot_temperature": slot_temperature,
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "support_mix": support_mix,
            "hypotheses": float(hypotheses),
            "hypothesis_top_k": float(hypothesis_top_k),
            "hypothesis_temperature": hypothesis_temperature,
            "query_mix": query_mix,
            "collapse_gain": collapse_gain,
            "uncertainty_width": uncertainty_width,
            "observer_channels": float(observer_channels),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _protein_folding_minima_embedding_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    residues = max(6, min(32, int(params.get("residues", max(8, hidden_dim // 8)))))
    minima = max(3, min(16, int(params.get("minima", 6))))
    steps = max(1, int(params.get("steps", 3)))
    local_strength = float(params.get("local_strength", 0.28))
    global_strength = float(params.get("global_strength", 0.16))
    collapse_gain = float(params.get("collapse_gain", 0.14))
    key_gain = float(params.get("key_gain", 0.30))
    top_k = max(1, min(minima, int(params.get("top_k", 2))))
    temperature = max(1e-4, float(params.get("temperature", 0.12)))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    phase_scale = float(params.get("phase_scale", 0.75))
    public_ratio = float(params.get("public_ratio", 0.14))
    public_mask = float(params.get("public_mask", 0.90))
    public_chunk = int(params.get("public_chunk", 8))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("protein_folding_minima_embedding_v0", secret_key, dim)
    )

    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    residue_proj = np.abs(
        local_rng.normal(size=(hidden_dim, residues)).astype(np.float32)
    )
    residue_proj = residue_proj / np.maximum(
        1e-6, np.sum(residue_proj, axis=1, keepdims=True)
    )
    local_kernel = np.eye(residues, dtype=np.float32)
    local_kernel = local_kernel + np.roll(local_kernel, 1, axis=1)
    local_kernel = local_kernel + np.roll(local_kernel, -1, axis=1)
    local_kernel = local_kernel / np.maximum(
        1e-6, np.sum(local_kernel, axis=1, keepdims=True)
    )
    contact_map = local_rng.normal(size=(residues, residues)).astype(np.float32)
    contact_map = (contact_map + contact_map.T) / 2.0
    contact_map = np.abs(contact_map) + np.eye(residues, dtype=np.float32) * 0.25
    contact_map = contact_map / np.maximum(
        1e-6, np.sum(contact_map, axis=1, keepdims=True)
    )
    minima_centers = np.abs(
        local_rng.normal(size=(minima, residues)).astype(np.float32)
    )
    minima_centers = _safe_normalize(minima_centers)
    key_real = _safe_normalize(
        local_rng.normal(size=(minima, hidden_dim)).astype(np.float32)
    )
    key_imag = _safe_normalize(
        local_rng.normal(size=(minima, hidden_dim)).astype(np.float32)
    )
    boundary_mix = local_rng.normal(size=(residues, public_dim)).astype(np.float32)
    decoy_mix = local_rng.normal(size=(minima, public_dim)).astype(np.float32)
    semantic_code_config = np.array([float(top_k), temperature], dtype=np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm

        base_energy = wave_real**2 + wave_imag**2
        residue_state = base_energy @ residue_proj
        residue_state = residue_state / np.maximum(
            1e-6, np.sum(residue_state, axis=1, keepdims=True)
        )

        # Lokale Nachbarschaft plus geheimer Kontaktgraph erzeugen viele plausible
        # Faltungsminima; der Schlüssel kippt nur die Auswahl der stabilen Senke.
        fold_state = residue_state
        for _ in range(steps):
            local_flow = fold_state @ local_kernel.T
            global_flow = fold_state @ contact_map.T
            fold_state = residue_state
            fold_state = fold_state + local_strength * np.tanh(local_flow)
            fold_state = fold_state + global_strength * np.tanh(global_flow)
            fold_state = np.maximum(fold_state, 0.0)
            fold_state = fold_state / np.maximum(
                1e-6, np.sum(fold_state, axis=1, keepdims=True)
            )

        basin_logits = fold_state @ minima_centers.T
        basin_logits = basin_logits + key_gain * np.abs(
            wave_real @ key_real.T + wave_imag @ key_imag.T
        )
        minima_weights = _topk_soft_assign(basin_logits, top_k, temperature)

        public = fold_state @ boundary_mix
        public = public + 0.35 * (minima_weights @ decoy_mix)
        public = _safe_normalize(public.astype(np.float32, copy=False))
        public = _mask_public_observation(
            "protein_folding_minima_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        aux_profile = _safe_normalize(np.hstack([fold_state, minima_weights]))
        return {
            "public": public,
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "energy": fold_state.astype(np.float32),
            "fold_state": fold_state.astype(np.float32),
            "minima_weights": minima_weights.astype(np.float32),
            "semantic_codes": minima_weights.astype(np.float32),
            "semantic_centers": minima_centers.astype(np.float32),
            "semantic_code_config": semantic_code_config,
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        overlap = overlap + query_state["wave_imag"] @ doc_state["wave_imag"].T
        carrier = overlap**2
        fold_match = query_state["fold_state"] @ doc_state["fold_state"].T
        minima_match = query_state["minima_weights"] @ doc_state["minima_weights"].T
        folded = 0.5 * fold_match + 0.5 * minima_match
        folded = folded - np.mean(folded, axis=1, keepdims=True)
        folded_scale = np.max(np.abs(folded), axis=1, keepdims=True)
        folded_scale = np.where(folded_scale == 0.0, 1.0, folded_scale)
        folded = folded / folded_scale
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        return carrier + uncertainty_gate * collapse_gain * folded

    return EmbeddingStateMethod(
        method_name="protein_folding_minima_embedding_v0",
        family="state_protein_folding",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "residues": float(residues),
            "minima": float(minima),
            "steps": float(steps),
            "local_strength": local_strength,
            "global_strength": global_strength,
            "collapse_gain": collapse_gain,
            "key_gain": key_gain,
            "top_k": float(top_k),
            "temperature": temperature,
            "uncertainty_width": uncertainty_width,
            "phase_scale": phase_scale,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _metamaterial_channel_embedding_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    channels = max(4, min(24, int(params.get("channels", 10))))
    steps = max(1, min(4, int(params.get("steps", 2))))
    top_k = max(1, min(channels, int(params.get("top_k", max(2, channels // 4)))))
    transport_mix = float(params.get("transport_mix", 0.32))
    channel_gain = float(params.get("channel_gain", 0.08))
    key_gain = float(params.get("key_gain", 0.28))
    temperature = max(1e-4, float(params.get("temperature", 0.12)))
    uncertainty_width = float(params.get("uncertainty_width", 0.025))
    boundary_gain = float(params.get("boundary_gain", 0.35))
    phase_scale = float(params.get("phase_scale", 0.75))
    public_ratio = float(params.get("public_ratio", 0.16))
    public_mask = float(params.get("public_mask", 0.84))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("metamaterial_channel_embedding_v0", secret_key, dim)
    )

    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    guide_basis = _safe_normalize(
        np.abs(local_rng.normal(size=(channels, hidden_dim)).astype(np.float32))
    )
    guide_real = _safe_normalize(
        local_rng.normal(size=(channels, hidden_dim)).astype(np.float32)
    )
    guide_imag = _safe_normalize(
        local_rng.normal(size=(channels, hidden_dim)).astype(np.float32)
    )
    key_basis = _safe_normalize(
        local_rng.normal(size=(channels, hidden_dim)).astype(np.float32)
    )
    directed_graph = local_rng.uniform(0.0, 1.0, size=(channels, channels)).astype(
        np.float32
    )
    direction_axis = np.linspace(0.0, 1.0, channels, dtype=np.float32)
    directed_graph = directed_graph + 0.45 * np.maximum(
        direction_axis[None, :] - direction_axis[:, None],
        0.0,
    )
    directed_graph = directed_graph + np.eye(channels, dtype=np.float32) * 0.15
    directed_graph = directed_graph / np.maximum(
        1e-6,
        np.sum(directed_graph, axis=1, keepdims=True),
    )
    boundary_proj = local_rng.normal(size=(channels, public_dim)).astype(np.float32)
    edge_proj = local_rng.normal(size=(channels, public_dim)).astype(np.float32)
    semantic_code_config = np.array([float(top_k), temperature], dtype=np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm

        energy = (wave_real**2 + wave_imag**2).astype(np.float32)
        phase_drive = np.abs(
            wave_real @ guide_real.T + wave_imag @ guide_imag.T
        ).astype(np.float32)
        keyed_drive = np.abs(wave_real @ key_basis.T).astype(np.float32)
        channel_logits = energy @ guide_basis.T
        channel_logits = channel_logits + key_gain * keyed_drive
        channel_logits = channel_logits + 0.25 * phase_drive
        active = _topk_soft_assign(channel_logits, top_k, temperature)

        transport = active
        for _ in range(steps):
            transported = transport @ directed_graph
            transport = (1.0 - transport_mix) * active + transport_mix * transported
            transport = np.maximum(transport, 0.0)
            transport = transport / np.maximum(
                1e-6,
                np.sum(transport, axis=1, keepdims=True),
            )

        # Oeffentlich sichtbar ist nur der Boundary-Mix; die gerichtete Leitfaehigkeit
        # selbst lebt im autorisierten Kanalprofil.
        boundary = np.abs(active - transport)
        public = boundary @ boundary_proj
        public = public + boundary_gain * (active @ edge_proj)
        public = _safe_normalize(public.astype(np.float32, copy=False))
        public = _mask_public_observation(
            "metamaterial_channel_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        aux_profile = _safe_normalize(np.hstack([active, transport, boundary]))
        return {
            "public": public,
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "energy": energy,
            "channel_active": active.astype(np.float32),
            "channel_transport": transport.astype(np.float32),
            "boundary_state": boundary.astype(np.float32),
            "semantic_codes": active.astype(np.float32),
            "semantic_centers": guide_basis.astype(np.float32),
            "semantic_code_config": semantic_code_config,
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        overlap = overlap + query_state["wave_imag"] @ doc_state["wave_imag"].T
        carrier = overlap**2
        active_match = query_state["channel_active"] @ doc_state["channel_active"].T
        transport_match = (
            query_state["channel_transport"] @ doc_state["channel_transport"].T
        )
        boundary_match = query_state["boundary_state"] @ doc_state["boundary_state"].T
        directed = 0.45 * active_match + 0.55 * transport_match
        directed = directed - boundary_gain * boundary_match
        directed = directed - np.mean(directed, axis=1, keepdims=True)
        directed_scale = np.max(np.abs(directed), axis=1, keepdims=True)
        directed_scale = np.where(directed_scale == 0.0, 1.0, directed_scale)
        directed = directed / directed_scale
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        return carrier + uncertainty_gate * channel_gain * directed

    return EmbeddingStateMethod(
        method_name="metamaterial_channel_embedding_v0",
        family="state_metamaterial_channel",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "channels": float(channels),
            "steps": float(steps),
            "top_k": float(top_k),
            "transport_mix": transport_mix,
            "channel_gain": channel_gain,
            "key_gain": key_gain,
            "temperature": temperature,
            "uncertainty_width": uncertainty_width,
            "boundary_gain": boundary_gain,
            "phase_scale": phase_scale,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _immune_self_nonself_embedding_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    requested_hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    receptors = max(6, min(24, int(params.get("receptors", 12))))
    self_top_k = max(1, min(receptors, int(params.get("self_top_k", 3))))
    self_gain = float(params.get("self_gain", 0.24))
    foreign_gain = float(params.get("foreign_gain", 0.16))
    foreign_penalty = float(params.get("foreign_penalty", 0.22))
    immune_gain = float(params.get("immune_gain", 0.06))
    tolerance_mix = float(params.get("tolerance_mix", 0.45))
    temperature = max(1e-4, float(params.get("temperature", 0.12)))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    phase_scale = float(params.get("phase_scale", 0.75))
    public_ratio = float(params.get("public_ratio", 0.16))
    public_mask = float(params.get("public_mask", 0.86))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("immune_self_nonself_embedding_v0", secret_key, dim)
    )

    real_proj = _qr_orthogonal(local_rng, dim, requested_hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, requested_hidden_dim)
    hidden_dim = real_proj.shape[1]
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    antigen_basis = np.abs(
        local_rng.normal(size=(hidden_dim, receptors)).astype(np.float32)
    )
    antigen_basis = antigen_basis / np.maximum(
        1e-6, np.sum(antigen_basis, axis=1, keepdims=True)
    )
    self_real = _safe_normalize(
        local_rng.normal(size=(receptors, hidden_dim)).astype(np.float32)
    )
    self_imag = _safe_normalize(
        local_rng.normal(size=(receptors, hidden_dim)).astype(np.float32)
    )
    foreign_real = _safe_normalize(
        local_rng.normal(size=(receptors, hidden_dim)).astype(np.float32)
    )
    foreign_imag = _safe_normalize(
        local_rng.normal(size=(receptors, hidden_dim)).astype(np.float32)
    )
    public_mix = local_rng.normal(size=(receptors, public_dim)).astype(np.float32)
    decoy_mix = local_rng.normal(size=(receptors, public_dim)).astype(np.float32)
    semantic_centers = _safe_normalize(antigen_basis.T.astype(np.float32, copy=True))
    semantic_code_config = np.array([float(self_top_k), temperature], dtype=np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm

        energy = (wave_real**2 + wave_imag**2).astype(np.float32)
        antigen = energy @ antigen_basis
        self_drive = np.abs(wave_real @ self_real.T + wave_imag @ self_imag.T)
        foreign_drive = np.abs(wave_real @ foreign_real.T - wave_imag @ foreign_imag.T)
        self_logits = antigen + self_gain * self_drive
        foreign_logits = antigen + foreign_gain * foreign_drive
        self_codes = _topk_soft_assign(self_logits, self_top_k, temperature)
        foreign_codes = _topk_soft_assign(
            foreign_logits, self_top_k, temperature * (1.0 + 0.25 * foreign_penalty)
        )

        tolerance_gate = 0.5 + 0.5 * np.tanh(
            (self_logits - foreign_logits) / max(1e-4, temperature)
        )
        tolerance_state = self_codes * tolerance_gate.astype(np.float32)
        tolerance_state = tolerance_state + tolerance_mix * np.maximum(
            self_codes - foreign_codes,
            0.0,
        )
        tolerance_sum = np.sum(tolerance_state, axis=1, keepdims=True)
        tolerance_sum = np.where(tolerance_sum == 0.0, 1.0, tolerance_sum)
        tolerance_state = (tolerance_state / tolerance_sum).astype(np.float32)

        decoy_state = (1.0 - tolerance_gate.astype(np.float32)) * foreign_codes
        decoy_sum = np.sum(decoy_state, axis=1, keepdims=True)
        decoy_sum = np.where(decoy_sum == 0.0, 1.0, decoy_sum)
        decoy_state = (decoy_state / decoy_sum).astype(np.float32)
        public = tolerance_state @ public_mix
        public = public + 0.30 * (decoy_state @ decoy_mix)
        public = _safe_normalize(public.astype(np.float32, copy=False))
        public = _mask_public_observation(
            "immune_self_nonself_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )

        aux_profile = _safe_normalize(
            np.hstack(
                [
                    tolerance_state,
                    self_codes.astype(np.float32),
                    foreign_codes.astype(np.float32),
                ]
            ).astype(np.float32)
        )
        return {
            "public": public,
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "energy": energy,
            "immune_self": self_codes.astype(np.float32),
            "immune_nonself": foreign_codes.astype(np.float32),
            "immune_tolerance": tolerance_state.astype(np.float32),
            "semantic_codes": tolerance_state.astype(np.float32),
            "semantic_centers": semantic_centers.astype(np.float32),
            "semantic_code_config": semantic_code_config,
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        overlap = overlap + query_state["wave_imag"] @ doc_state["wave_imag"].T
        carrier = overlap**2
        self_match = query_state["immune_self"] @ doc_state["immune_self"].T
        tolerance_match = (
            query_state["immune_tolerance"] @ doc_state["immune_tolerance"].T
        )
        foreign_match = query_state["immune_nonself"] @ doc_state["immune_nonself"].T
        immune = 0.45 * self_match + 0.55 * tolerance_match
        immune = immune - foreign_penalty * foreign_match
        immune = immune - np.mean(immune, axis=1, keepdims=True)
        immune_scale = np.max(np.abs(immune), axis=1, keepdims=True)
        immune_scale = np.where(immune_scale == 0.0, 1.0, immune_scale)
        immune = immune / immune_scale
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        return carrier + uncertainty_gate * immune_gain * immune

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        self_match = query_state["immune_self"] @ doc_state["immune_self"].T
        tolerance_match = (
            query_state["immune_tolerance"] @ doc_state["immune_tolerance"].T
        )
        foreign_match = query_state["immune_nonself"] @ doc_state["immune_nonself"].T
        return (
            0.35 * self_match
            + 0.55 * tolerance_match
            - 0.10 * foreign_penalty * foreign_match
        ).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="immune_self_nonself_embedding_v0",
        family="state_immune_self_nonself",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "receptors": float(receptors),
            "self_top_k": float(self_top_k),
            "self_gain": self_gain,
            "foreign_gain": foreign_gain,
            "foreign_penalty": foreign_penalty,
            "immune_gain": immune_gain,
            "tolerance_mix": tolerance_mix,
            "temperature": temperature,
            "uncertainty_width": uncertainty_width,
            "phase_scale": phase_scale,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
        aux_score=_aux_score,
    )


def _projective_kahler_symplectic_hybrid_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    symplectic_weight = float(params.get("symplectic_weight", 0.06))
    compatibility_power = float(params.get("compatibility_power", 1.0))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        overlap = overlap + query_state["wave_imag"] @ doc_state["wave_imag"].T
        metric = overlap**2
        form = query_state["wave_real"] @ doc_state["wave_imag"].T
        form = form - query_state["wave_imag"] @ doc_state["wave_real"].T
        compatibility = np.power(
            np.clip(metric, 0.0, 1.0),
            max(0.0, compatibility_power),
        ).astype(np.float32)
        uncertainty_gate = _relative_uncertainty_gate(metric, uncertainty_width)
        return metric - uncertainty_gate * symplectic_weight * compatibility * np.abs(
            form
        )

    return EmbeddingStateMethod(
        method_name="projective_kahler_symplectic_hybrid_v0",
        family="state_projective_kahler_symplectic",
        params={
            **base.params,
            "symplectic_weight": symplectic_weight,
            "compatibility_power": compatibility_power,
            "uncertainty_width": uncertainty_width,
        },
        encode_docs=base.encode_docs,
        encode_queries=base.encode_queries,
        score=_score,
    )


def _projective_observer_resonance_hybrid_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    observer = _observer_resonance_build(rng, params)
    observer_mix = float(params.get("observer_mix", 0.015))
    compatibility_power = float(params.get("compatibility_power", 1.0))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    channels = int(params.get("channels", 6))
    channel_gain = float(params.get("channel_gain", 0.60))

    def _observer_profile(observer_state: StateMap) -> np.ndarray:
        channel_state = observer_state.get("channels")
        if channel_state is None:
            return np.zeros((0, 0), dtype=np.float32)
        if channel_state.shape[0] == 0:
            return np.zeros((0, channels), dtype=np.float32)
        summary = np.mean(np.maximum(channel_state, 0.0), axis=2)
        summary = summary - np.mean(summary, axis=1, keepdims=True)
        return _safe_normalize(summary.astype(np.float32, copy=False))

    def _merge_state(base_state: StateMap, observer_state: StateMap) -> StateMap:
        return {
            **base_state,
            "channels": observer_state["channels"],
            "observer_channels": observer_state["channels"],
            "aux_operator_profile": _observer_profile(observer_state),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _merge_state(base.encode_docs(x), observer.encode_docs(x))

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _merge_state(base.encode_queries(x), observer.encode_queries(x))

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        observer_scores = observer.score(doc_state, query_state)
        observer_scores = observer_scores - np.mean(
            observer_scores, axis=1, keepdims=True
        )
        observer_scale = np.max(np.abs(observer_scores), axis=1, keepdims=True)
        observer_scale = np.where(observer_scale == 0.0, 1.0, observer_scale)
        return observer_scores / observer_scale

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base.score(doc_state, query_state)
        observer_scores = _aux_score(doc_state, query_state)
        compatibility = np.power(
            np.clip(carrier, 0.0, 1.0),
            max(0.0, compatibility_power),
        ).astype(np.float32)
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        return (
            carrier + observer_mix * uncertainty_gate * compatibility * observer_scores
        )

    return EmbeddingStateMethod(
        method_name="projective_observer_resonance_hybrid_v0",
        family="state_projective_observer_resonance_hybrid",
        params={
            **base.params,
            "channels": float(channels),
            "channel_gain": channel_gain,
            "observer_mix": observer_mix,
            "compatibility_power": compatibility_power,
            "uncertainty_width": uncertainty_width,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _projective_chart_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    chart = _calabi_yau_chart_build(rng, params)
    charts = int(params.get("charts", 6))
    chart_dim = int(params.get("chart_dim", max(12, int(params.get("dim", 0)) // 2)))
    fiber_gain = float(params.get("fiber_gain", 0.45))

    def _chart_profile(chart_state: StateMap) -> np.ndarray:
        chart_coords = chart_state.get("chart_coords")
        chart_weights = chart_state.get("chart_weights")
        if chart_coords is None:
            return np.zeros((0, 0), dtype=np.float32)
        if chart_weights is None:
            return np.zeros((0, 0), dtype=np.float32)
        if chart_coords.shape[0] == 0:
            return np.zeros((0, 2 * charts), dtype=np.float32)
        coord_energy = np.linalg.norm(chart_coords, axis=2).astype(np.float32)
        merged = np.hstack([chart_weights, coord_energy]).astype(np.float32)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _merge_state(base_state: StateMap, chart_state: StateMap) -> StateMap:
        return {
            **base_state,
            "chart_coords": chart_state["chart_coords"],
            "chart_weights": chart_state["chart_weights"],
            "aux_operator_profile": _chart_profile(chart_state),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _merge_state(base.encode_docs(x), chart.encode_docs(x))

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _merge_state(base.encode_queries(x), chart.encode_queries(x))

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        chart_scores = chart.score(doc_state, query_state)
        chart_scores = chart_scores - np.mean(chart_scores, axis=1, keepdims=True)
        chart_scale = np.max(np.abs(chart_scores), axis=1, keepdims=True)
        chart_scale = np.where(chart_scale == 0.0, 1.0, chart_scale)
        return chart_scores / chart_scale

    return EmbeddingStateMethod(
        method_name="projective_chart_observable_v0",
        family="state_projective_chart_observable",
        params={
            **base.params,
            "charts": float(charts),
            "chart_dim": float(chart_dim),
            "fiber_gain": fiber_gain,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base.score,
        aux_score=_aux_score,
    )


def _projective_topological_code_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = base.encode_docs(np.zeros((1, dim), dtype=np.float32))[
        "wave_real"
    ].shape[1]
    plaquettes = max(4, int(params.get("plaquettes", 10)))
    patch_width = max(4, min(hidden_dim, int(params.get("patch_width", 12))))
    syndrome_gain = max(0.05, float(params.get("syndrome_gain", 1.8)))
    phase_gain = max(0.05, float(params.get("phase_gain", 0.7)))
    recover_gain = min(1.0, max(0.0, float(params.get("recover_gain", 0.20))))
    profile_mix = min(1.0, max(0.0, float(params.get("profile_mix", 0.35))))
    defect_top_k = max(1, min(plaquettes, int(params.get("defect_top_k", 3))))
    defect_temperature = max(1e-4, float(params.get("defect_temperature", 0.08)))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_topological_code_observable_v0", secret_key, dim)
    )
    coord_to_plaquette = local_rng.integers(0, plaquettes, size=(hidden_dim,))
    coord_signs = local_rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32), size=(hidden_dim,)
    ).astype(np.float32)

    def _energy_simplex(state: StateMap) -> np.ndarray:
        energy = np.maximum(
            1e-6, state["wave_real"] ** 2 + state["wave_imag"] ** 2
        ).astype(np.float32)
        energy_sum = np.sum(energy, axis=1, keepdims=True)
        energy_sum = np.where(energy_sum == 0.0, 1.0, energy_sum)
        return energy / energy_sum

    def _topological_profile(state: StateMap, recover: float) -> StateMap:
        simplex = _energy_simplex(state)
        phase = np.arctan2(state["wave_imag"], state["wave_real"]).astype(np.float32)
        top_components = min(hidden_dim, patch_width)
        top_idx = np.argpartition(-simplex, top_components - 1, axis=1)[
            :, :top_components
        ]
        patch_mass_bank = np.zeros((simplex.shape[0], plaquettes), dtype=np.float32)
        flux_bank = np.zeros((simplex.shape[0], plaquettes), dtype=np.float32)

        for row in range(simplex.shape[0]):
            row_idx = top_idx[row]
            row_mass = simplex[row, row_idx]
            row_bins = coord_to_plaquette[row_idx]
            row_flux = (
                row_mass
                * coord_signs[row_idx]
                * np.cos(phase_gain * phase[row, row_idx])
            )
            np.add.at(patch_mass_bank[row], row_bins, row_mass)
            np.add.at(flux_bank[row], row_bins, row_flux.astype(np.float32))

        if recover > 0.0:
            sharpen = np.power(np.maximum(patch_mass_bank, 1e-6), 1.0 + recover_gain)
            sharpen_sum = np.sum(sharpen, axis=1, keepdims=True)
            sharpen_sum = np.where(sharpen_sum == 0.0, 1.0, sharpen_sum)
            patch_mass_bank = (sharpen / sharpen_sum).astype(np.float32)

        flux_bank = np.tanh(phase_gain * flux_bank).astype(np.float32)
        stabilizer_bank = patch_mass_bank - np.roll(patch_mass_bank, -1, axis=1)
        stabilizer_bank = np.tanh(
            (1.0 + recover_gain * recover) * syndrome_gain * stabilizer_bank
        ).astype(np.float32)
        defect_logits = patch_mass_bank + profile_mix * np.abs(stabilizer_bank)
        defect_logits = defect_logits + 0.25 * np.abs(flux_bank)
        defect_temp = max(
            1e-4, defect_temperature * (1.0 - 0.5 * recover_gain * recover)
        )
        defect_code = _topk_soft_assign(defect_logits, defect_top_k, defect_temp)
        profile = np.hstack(
            [
                defect_code,
                stabilizer_bank,
                flux_bank,
                patch_mass_bank,
            ]
        ).astype(np.float32)
        return {
            **state,
            "topological_patch_mass": patch_mass_bank,
            "topological_stabilizer": stabilizer_bank,
            "topological_flux": flux_bank,
            "topological_defect_code": defect_code.astype(np.float32),
            "aux_operator_profile": _safe_normalize(profile),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _topological_profile(base.encode_docs(x), 0.0)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _topological_profile(base.encode_queries(x), 1.0)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        patch_mass = (
            query_state["topological_patch_mass"]
            @ doc_state["topological_patch_mass"].T
        )
        stabilizer = (
            query_state["topological_stabilizer"]
            @ doc_state["topological_stabilizer"].T
        )
        flux = query_state["topological_flux"] @ doc_state["topological_flux"].T
        defect = (
            query_state["topological_defect_code"]
            @ doc_state["topological_defect_code"].T
        )
        return (
            0.30 * patch_mass
            + (0.25 + 0.20 * (1.0 - profile_mix)) * stabilizer
            + 0.20 * flux
            + (0.25 + 0.40 * profile_mix) * defect
        ).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_topological_code_observable_v0",
        family="state_projective_topological_code_observable",
        params={
            **base.params,
            "plaquettes": float(plaquettes),
            "patch_width": float(patch_width),
            "syndrome_gain": syndrome_gain,
            "phase_gain": phase_gain,
            "recover_gain": recover_gain,
            "profile_mix": profile_mix,
            "defect_top_k": float(defect_top_k),
            "defect_temperature": defect_temperature,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base.score,
        aux_score=_aux_score,
    )


def _projective_keyed_collapse_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = base.encode_docs(np.zeros((1, dim), dtype=np.float32))[
        "wave_real"
    ].shape[1]
    eigenspaces = max(4, int(params.get("eigenspaces", 8)))
    max_subspace_dim = max(1, hidden_dim // eigenspaces)
    subspace_dim = max(
        1,
        min(max_subspace_dim, int(params.get("subspace_dim", max(4, hidden_dim // 8)))),
    )
    eigenspaces = max(2, min(eigenspaces, hidden_dim // subspace_dim))
    active_dim = eigenspaces * subspace_dim
    doc_top_k = min(eigenspaces, max(1, int(params.get("doc_top_k", 4))))
    query_top_k = min(eigenspaces, max(1, int(params.get("query_top_k", 2))))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.14)))
    route_scale = float(params.get("route_scale", 1.05))
    collapse_gain = max(1.0, float(params.get("collapse_gain", 1.7)))
    phase_mix = max(0.0, float(params.get("phase_mix", 0.30)))
    query_recover_gain = min(
        1.0, max(0.0, float(params.get("query_recover_gain", 0.18)))
    )
    decoy_floor = min(max(float(params.get("decoy_floor", 0.12)), 0.0), 0.75)
    profile_mix = min(max(float(params.get("profile_mix", 0.32)), 0.0), 1.0)
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_keyed_collapse_observable_v0", secret_key, dim)
    )
    eigenspace_basis = _qr_orthogonal(local_rng, hidden_dim, hidden_dim)[:, :active_dim]
    eigenspace_basis = eigenspace_basis.reshape(hidden_dim, eigenspaces, subspace_dim)
    eigenspace_basis = np.transpose(eigenspace_basis, (1, 2, 0)).astype(np.float32)
    eigenspace_phase = local_rng.uniform(-math.pi, math.pi, size=(eigenspaces,)).astype(
        np.float32
    )
    eigenspace_bias = local_rng.uniform(-0.18, 0.18, size=(eigenspaces,)).astype(
        np.float32
    )
    phase_probe_real = _safe_normalize(
        local_rng.normal(size=(eigenspaces, hidden_dim)).astype(np.float32)
    )
    phase_probe_imag = _safe_normalize(
        local_rng.normal(size=(eigenspaces, hidden_dim)).astype(np.float32)
    )

    def _collapse_features(
        state: StateMap, top_k: int, scale: float, recover: float
    ) -> StateMap:
        wave_real = state["wave_real"]
        wave_imag = state["wave_imag"]
        phase_real = np.maximum(wave_real, 0.0).astype(np.float32)
        phase_imag = np.maximum(wave_imag, 0.0).astype(np.float32)
        eig_real = np.einsum(
            "nh,skh->nsk", wave_real, eigenspace_basis, dtype=np.float32
        )
        eig_imag = np.einsum(
            "nh,skh->nsk", wave_imag, eigenspace_basis, dtype=np.float32
        )
        eig_energy = np.sum(eig_real**2 + eig_imag**2, axis=2, dtype=np.float32)
        phase_support = (
            np.maximum(phase_real @ phase_probe_real.T, 0.0)
            + np.maximum(phase_imag @ phase_probe_imag.T, 0.0)
        ) / float(max(1, hidden_dim))
        route_logits = eig_energy + phase_mix * phase_support + eigenspace_bias[None, :]
        if recover > 0.0:
            route_logits = (1.0 - recover) * route_logits + recover * (
                eig_energy + 0.5 * phase_support
            )
        local_temperature = max(1e-4, route_temperature * (1.0 - 0.35 * recover))
        mode_weight = _topk_soft_assign(route_logits * scale, top_k, local_temperature)
        mode_weight = (
            decoy_floor / float(eigenspaces) + (1.0 - decoy_floor) * mode_weight
        ).astype(np.float32)
        mode_sum = np.sum(mode_weight, axis=1, keepdims=True)
        mode_sum = np.where(mode_sum == 0.0, 1.0, mode_sum)
        mode_weight = mode_weight / mode_sum
        phase_cos = np.cos(eigenspace_phase)[None, :, None]
        phase_sin = np.sin(eigenspace_phase)[None, :, None]
        collapse_scale = np.sqrt(np.maximum(mode_weight, 1e-6)).astype(np.float32)[
            :, :, None
        ]
        collapsed_real = collapse_scale * (eig_real * phase_cos - eig_imag * phase_sin)
        collapsed_imag = collapse_scale * (eig_real * phase_sin + eig_imag * phase_cos)
        collapse_energy = mode_weight * np.maximum(route_logits, 0.0)
        collapse_flux = np.mean(collapsed_real, axis=2, dtype=np.float32)
        collapse_spin = np.mean(collapsed_imag, axis=2, dtype=np.float32)
        aux_profile = _safe_normalize(
            np.hstack(
                [
                    mode_weight,
                    collapse_energy,
                    profile_mix * phase_support
                    + (1.0 - profile_mix) * np.abs(collapse_flux),
                    collapse_spin,
                ]
            ).astype(np.float32)
        )
        return {
            **state,
            "collapse_mode_weight": mode_weight.astype(np.float32),
            "collapse_energy": collapse_energy.astype(np.float32),
            "collapse_phase_support": phase_support.astype(np.float32),
            "collapse_real": collapsed_real.astype(np.float32),
            "collapse_imag": collapsed_imag.astype(np.float32),
            "collapse_flux": collapse_flux.astype(np.float32),
            "collapse_spin": collapse_spin.astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _collapse_features(base.encode_docs(x), doc_top_k, route_scale, 0.0)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _collapse_features(
            base.encode_queries(x),
            query_top_k,
            route_scale * collapse_gain,
            query_recover_gain,
        )

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        collapse_overlap = np.einsum(
            "qsk,dsk->qds",
            query_state["collapse_real"],
            doc_state["collapse_real"],
            dtype=np.float32,
        )
        collapse_overlap = collapse_overlap + np.einsum(
            "qsk,dsk->qds",
            query_state["collapse_imag"],
            doc_state["collapse_imag"],
            dtype=np.float32,
        )
        collapse_score = np.mean(collapse_overlap**2, axis=2, dtype=np.float32)
        mode_gate = np.sqrt(
            np.maximum(
                query_state["collapse_mode_weight"]
                @ doc_state["collapse_mode_weight"].T,
                0.0,
            )
        ).astype(np.float32)
        energy_support = (
            query_state["collapse_energy"] @ doc_state["collapse_energy"].T
        ) / float(max(1, eigenspaces))
        phase_align = (
            query_state["collapse_phase_support"]
            @ doc_state["collapse_phase_support"].T
        ) / float(max(1, eigenspaces))
        flux_align = (
            query_state["collapse_flux"] @ doc_state["collapse_flux"].T
        ) / float(max(1, eigenspaces))
        spin_align = (
            query_state["collapse_spin"] @ doc_state["collapse_spin"].T
        ) / float(max(1, eigenspaces))
        aux_scores = collapse_score * (0.25 + 0.75 * mode_gate)
        aux_scores = aux_scores + 0.22 * energy_support + 0.18 * phase_align
        aux_scores = aux_scores + 0.18 * flux_align + 0.17 * spin_align
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return (aux_scores / scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_keyed_collapse_observable_v0",
        family="state_projective_keyed_collapse_observable",
        params={
            **base.params,
            "eigenspaces": float(eigenspaces),
            "subspace_dim": float(subspace_dim),
            "doc_top_k": float(doc_top_k),
            "query_top_k": float(query_top_k),
            "route_temperature": route_temperature,
            "route_scale": route_scale,
            "collapse_gain": collapse_gain,
            "phase_mix": phase_mix,
            "query_recover_gain": query_recover_gain,
            "decoy_floor": decoy_floor,
            "profile_mix": profile_mix,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base.score,
        aux_score=_aux_score,
    )


def _causal_set_percolation_embedding_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = base.encode_docs(np.zeros((1, dim), dtype=np.float32))[
        "wave_real"
    ].shape[1]
    anchors = max(6, int(params.get("anchors", 16)))
    order_top_k = min(anchors, max(2, int(params.get("order_top_k", 4))))
    order_temperature = max(1e-4, float(params.get("order_temperature", 0.10)))
    channels = max(6, int(params.get("channels", 12)))
    active_top_k = min(channels, max(1, int(params.get("active_top_k", 3))))
    threshold = max(0.0, float(params.get("threshold", 0.08)))
    order_gain = max(0.05, float(params.get("order_gain", 1.10)))
    phase_gain = max(0.0, float(params.get("phase_gain", 0.30)))
    percolation_gain = max(0.05, float(params.get("percolation_gain", 0.65)))
    relation_mix = max(0.0, float(params.get("relation_mix", 0.004)))
    support_mix = max(0.0, float(params.get("support_mix", 0.002)))
    uncertainty_width = float(params.get("uncertainty_width", 0.03))
    support_width = max(4, int(params.get("support_width", 8)))
    support_temperature = max(1e-4, float(params.get("support_temperature", 0.10)))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("causal_set_percolation_embedding_v0", secret_key, dim)
    )
    anchor_real = _safe_normalize(
        local_rng.normal(size=(anchors, hidden_dim)).astype(np.float32)
    )
    anchor_imag = _safe_normalize(
        local_rng.normal(size=(anchors, hidden_dim)).astype(np.float32)
    )
    channel_proj = _safe_normalize(
        local_rng.normal(size=(channels, anchors)).astype(np.float32)
    )
    channel_graph = local_rng.uniform(0.0, 1.0, size=(channels, channels)).astype(
        np.float32
    )
    channel_graph = 0.5 * (channel_graph + channel_graph.T)
    channel_graph = channel_graph / max(1.0, float(channels))

    def _state_profiles(state: StateMap, recover: float) -> StateMap:
        phase = np.arctan2(state["wave_imag"], state["wave_real"]).astype(np.float32)
        order_logits = state["wave_real"] @ anchor_real.T
        order_logits = order_logits + state["wave_imag"] @ anchor_imag.T
        phase_probe = np.cos(phase_gain * phase).astype(np.float32)
        order_logits = order_gain * order_logits + 0.20 * (
            phase_probe @ np.abs(anchor_real).T
        )
        order_logits = order_logits - np.mean(order_logits, axis=1, keepdims=True)
        order_scale = np.max(np.abs(order_logits), axis=1, keepdims=True)
        order_scale = np.where(order_scale == 0.0, 1.0, order_scale)
        order_logits = order_logits / order_scale

        order_profile = np.sign(
            order_logits[:, :, None] - order_logits[:, None, :]
        ).mean(axis=2)
        order_profile = order_profile.astype(np.float32, copy=False)
        order_profile = order_profile - np.mean(order_profile, axis=1, keepdims=True)
        order_profile = _safe_normalize(order_profile)

        focus_temperature = max(1e-4, order_temperature * (1.0 - 0.20 * recover))
        order_focus = _topk_soft_assign(order_logits**2, order_top_k, focus_temperature)
        active_floor = threshold * (1.0 - 0.35 * recover)
        channel_drive = np.maximum(np.abs(order_logits) - active_floor, 0.0)
        channel_drive = channel_drive @ channel_proj.T
        active = _topk_soft_assign(channel_drive**2, active_top_k, order_temperature)
        percolation = active @ channel_graph
        percolation = percolation + 0.35 * (order_focus @ channel_proj.T)
        percolation = np.tanh(percolation_gain * percolation).astype(np.float32)
        percolation = percolation - np.mean(percolation, axis=1, keepdims=True)
        percolation = _safe_normalize(percolation)

        profile = np.hstack([order_profile, order_focus, active, percolation]).astype(
            np.float32
        )
        return {
            **state,
            "causal_order_profile": order_profile,
            "causal_focus": order_focus.astype(np.float32),
            "percolation_active": active.astype(np.float32),
            "percolation_profile": percolation,
            "aux_operator_profile": _safe_normalize(profile),
        }

    def _build_support_graph(doc_state: StateMap) -> np.ndarray:
        order_profile = doc_state["causal_order_profile"]
        percolation = doc_state["percolation_profile"]
        doc_count = order_profile.shape[0]
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        if doc_count <= 1:
            return support_graph
        keep_neighbors = min(support_width, doc_count - 1)
        if keep_neighbors <= 0:
            return support_graph
        similarity = order_profile @ order_profile.T
        similarity = similarity + percolation @ percolation.T
        similarity = 0.5 * similarity
        masked_similarity = np.array(similarity, dtype=np.float32, copy=True)
        np.fill_diagonal(masked_similarity, -1e9)
        neighbor_idx = np.argpartition(-masked_similarity, keep_neighbors - 1, axis=1)[
            :, :keep_neighbors
        ]
        neighbor_scores = np.take_along_axis(masked_similarity, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
        return support_graph

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = _state_profiles(base.encode_docs(x), 0.0)
        return {
            **state,
            "support_graph": _build_support_graph(state),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _state_profiles(base.encode_queries(x), 1.0)

    def _relation_signal(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        order_score = (
            query_state["causal_order_profile"] @ doc_state["causal_order_profile"].T
        )
        focus_score = query_state["causal_focus"] @ doc_state["causal_focus"].T
        active_score = (
            query_state["percolation_active"] @ doc_state["percolation_active"].T
        )
        percolation_score = (
            query_state["percolation_profile"] @ doc_state["percolation_profile"].T
        )
        order_score = np.clip(0.5 * (order_score + 1.0), 0.0, 1.0)
        focus_score = np.clip(focus_score, 0.0, 1.0)
        active_score = np.clip(active_score, 0.0, 1.0)
        percolation_score = np.clip(0.5 * (percolation_score + 1.0), 0.0, 1.0)
        return (
            0.35 * order_score
            + 0.20 * focus_score
            + 0.20 * active_score
            + 0.25 * percolation_score
        ).astype(np.float32)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        aux = _relation_signal(doc_state, query_state)
        aux = aux - np.mean(aux, axis=1, keepdims=True)
        aux_scale = np.max(np.abs(aux), axis=1, keepdims=True)
        aux_scale = np.where(aux_scale == 0.0, 1.0, aux_scale)
        return aux / aux_scale

    def _support_signal(doc_state: StateMap, carrier: np.ndarray) -> np.ndarray:
        doc_count = carrier.shape[1]
        support_signal = np.zeros_like(carrier)
        if doc_count == 0:
            return support_signal
        support_graph = doc_state["support_graph"]
        if support_graph.shape != (doc_count, doc_count):
            return support_signal
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return support_signal
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_vector = np.zeros_like(carrier)
        np.put_along_axis(support_vector, support_idx, support_weights, axis=1)
        support_signal = 0.5 * (
            support_vector @ support_graph + support_vector @ support_graph.T
        )
        return np.clip(support_signal, 0.0, 1.0).astype(np.float32)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base.score(doc_state, query_state)
        relation_signal = _relation_signal(doc_state, query_state)
        support_signal = _support_signal(doc_state, carrier)
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        rerank_signal = np.clip(
            (1.0 - support_mix) * relation_signal + support_mix * support_signal,
            0.0,
            1.0,
        ).astype(np.float32)
        rerank_gate = 1.0 - relation_mix * uncertainty_gate * (1.0 - rerank_signal)
        rerank_gate = np.clip(rerank_gate, 1.0 - relation_mix, 1.0)
        return (
            carrier * rerank_gate
            + support_mix * uncertainty_gate * carrier * support_signal
        )

    return EmbeddingStateMethod(
        method_name="causal_set_percolation_embedding_v0",
        family="state_causal_set_percolation",
        params={
            **base.params,
            "anchors": float(anchors),
            "order_top_k": float(order_top_k),
            "order_temperature": order_temperature,
            "channels": float(channels),
            "active_top_k": float(active_top_k),
            "threshold": threshold,
            "order_gain": order_gain,
            "phase_gain": phase_gain,
            "percolation_gain": percolation_gain,
            "relation_mix": relation_mix,
            "support_mix": support_mix,
            "uncertainty_width": uncertainty_width,
            "support_width": float(support_width),
            "support_temperature": support_temperature,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _complex_wave_kahler_symplectic_hybrid_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _complex_wavepacket_build(rng, params)
    symplectic_weight = float(params.get("symplectic_weight", 0.04))
    compatibility_power = float(params.get("compatibility_power", 1.0))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    amplitude_mix = float(params.get("amplitude_mix", 0.35))

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        metric = base.score(doc_state, query_state)
        bands = query_state["wave_real"].shape[1]
        form_total = np.zeros_like(metric)
        compatibility_total = np.zeros_like(metric)
        for band in range(bands):
            q_real = query_state["wave_real"][:, band]
            d_real = doc_state["wave_real"][:, band]
            q_imag = query_state["wave_imag"][:, band]
            d_imag = doc_state["wave_imag"][:, band]
            coherence = q_real @ d_real.T
            coherence = coherence + q_imag @ d_imag.T
            amplitude = (
                query_state["amp_hidden"][:, band] @ doc_state["amp_hidden"][:, band].T
            )
            form = q_real @ d_imag.T
            form = form - q_imag @ d_real.T
            band_compatibility = (1.0 - amplitude_mix) * np.clip(coherence, 0.0, 1.0)
            band_compatibility = band_compatibility + amplitude_mix * np.clip(
                amplitude,
                0.0,
                1.0,
            )
            compatibility_total = compatibility_total + band_compatibility
            form_total = form_total + np.abs(form)
        compatibility = compatibility_total / float(max(1, bands))
        instability = np.power(
            np.clip(1.0 - compatibility, 0.0, 1.0),
            max(0.0, compatibility_power),
        ).astype(np.float32)
        form_penalty = form_total / float(max(1, bands))
        uncertainty_gate = _relative_uncertainty_gate(metric, uncertainty_width)
        # Keep the wave carrier intact and only use the symplectic form as a keyed tie-breaker.
        return (
            metric - uncertainty_gate * symplectic_weight * instability * form_penalty
        )

    return EmbeddingStateMethod(
        method_name="complex_wave_kahler_symplectic_hybrid_v0",
        family="state_wave_kahler_symplectic",
        params={
            **base.params,
            "symplectic_weight": symplectic_weight,
            "compatibility_power": compatibility_power,
            "uncertainty_width": uncertainty_width,
            "amplitude_mix": amplitude_mix,
        },
        encode_docs=base.encode_docs,
        encode_queries=base.encode_queries,
        score=_score,
    )


def _projective_phase_holonomy_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_hilbert_embedding_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    loops = max(2, int(params.get("loops", 4)))
    curvature = float(params.get("curvature", 0.85))
    circulation_weight = float(params.get("circulation_weight", 0.40))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode_projective(
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _phase_holonomy_profile(
        wave_real: np.ndarray, wave_imag: np.ndarray
    ) -> np.ndarray:
        loop_features = []
        width = wave_real.shape[1]
        for loop_idx in range(loops):
            shift = 1 + (loop_idx % max(1, width - 1))
            rolled_real = np.roll(wave_real, shift, axis=1)
            rolled_imag = np.roll(wave_imag, shift, axis=1)
            coherence = wave_real * rolled_real + wave_imag * rolled_imag
            circulation = wave_imag * rolled_real - wave_real * rolled_imag
            loop_state = np.cos(curvature * coherence)
            loop_state = loop_state + circulation_weight * np.sin(
                curvature * circulation
            )
            loop_features.append(loop_state.astype(np.float32))
        return _safe_normalize(np.hstack(loop_features))

    def _encode(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        aux_profile = _phase_holonomy_profile(wave_real, wave_imag)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )

    return EmbeddingStateMethod(
        method_name="projective_phase_holonomy_observable_v0",
        family="state_projective_phase_holonomy_observable",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "loops": float(loops),
            "curvature": curvature,
            "circulation_weight": circulation_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _build_projective_holonomy_observable(
    method_name: str, rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_hilbert_embedding_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    holonomy_loops = max(2, int(params.get("holonomy_loops", 8)))
    holonomy_curvature = float(params.get("holonomy_curvature", 0.82))
    residual_mix = float(params.get("residual_mix", 0.0))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))

    carrier_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    real_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    phase_bias = carrier_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = carrier_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    holonomy_rng = np.random.default_rng(
        _method_seed("holonomy_loop_embedding_v0", secret_key, dim)
    )
    holonomy_proj = np.stack(
        [_qr_orthogonal(holonomy_rng, dim, dim) for _ in range(holonomy_loops)]
    )
    holonomy_phase_bias = holonomy_rng.uniform(
        -math.pi, math.pi, size=(holonomy_loops, dim)
    ).astype(np.float32)
    aux_dim = holonomy_loops * dim
    shared_bridge = np.zeros((hidden_dim, aux_dim), dtype=np.float32)
    if residual_mix > 0.0:
        residual_rng = np.random.default_rng(_method_seed(method_name, secret_key, dim))
        shared_bridge = residual_rng.normal(size=(hidden_dim, aux_dim)).astype(
            np.float32
        )

    def _encode_projective(
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        energy = (wave_real**2 + wave_imag**2).astype(np.float32)
        return (
            public,
            wave_real.astype(np.float32),
            wave_imag.astype(np.float32),
            energy,
        )

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _holonomy_aux_profile(x: np.ndarray) -> np.ndarray:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        loop_states = []
        for loop_idx in range(holonomy_loops):
            base = y @ holonomy_proj[loop_idx]
            shifted = np.roll(base, loop_idx + 1, axis=1)
            holonomy = np.cos(holonomy_curvature * base + holonomy_phase_bias[loop_idx])
            holonomy = holonomy + np.sin(holonomy_curvature * shifted)
            holonomy = _safe_normalize(holonomy)
            loop_states.append(holonomy.astype(np.float32))
        flat_loops = np.reshape(np.stack(loop_states, axis=1), (y.shape[0], -1))
        return _safe_normalize(flat_loops.astype(np.float32, copy=False))

    def _encode(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, energy = _encode_projective(x)
        aux_profile = _holonomy_aux_profile(x)
        if residual_mix > 0.0:
            shared_projection = _safe_normalize(energy @ shared_bridge)
            aux_profile = aux_profile - residual_mix * shared_projection
            aux_profile = aux_profile - np.mean(aux_profile, axis=1, keepdims=True)
            aux_profile = _safe_normalize(aux_profile.astype(np.float32, copy=False))
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "aux_operator_profile": aux_profile,
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )

    return EmbeddingStateMethod(
        method_name=method_name,
        family="state_projective_parallel_holonomy_observable",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "holonomy_loops": float(holonomy_loops),
            "holonomy_curvature": holonomy_curvature,
            "residual_mix": residual_mix,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _projective_parallel_holonomy_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    return _build_projective_holonomy_observable(
        "projective_parallel_holonomy_observable_v0",
        rng,
        params,
    )


def _projective_residual_holonomy_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    local_params = dict(params)
    local_params.setdefault("residual_mix", 0.35)
    return _build_projective_holonomy_observable(
        "projective_residual_holonomy_observable_v0",
        rng,
        local_params,
    )


def _projective_graph_transport_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_hilbert_embedding_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    relation_slots = max(8, int(params.get("relation_slots", 16)))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 2))))
    relation_temperature = float(params.get("relation_temperature", 0.08))
    transport_mix = float(params.get("transport_mix", 0.72))
    symmetry_mix = float(params.get("symmetry_mix", 0.35))
    support_width = max(graph_k, int(params.get("support_width", 8)))
    support_temperature = float(params.get("support_temperature", 0.10))
    score_gain = float(params.get("score_gain", 0.012))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))

    carrier_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    real_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    phase_bias = carrier_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = carrier_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    local_rng = np.random.default_rng(
        _method_seed("projective_graph_transport_observable_v0", secret_key, dim)
    )
    anchor_real = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )
    anchor_imag = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _relation_profile(wave_real: np.ndarray, wave_imag: np.ndarray) -> np.ndarray:
        relation_logits = wave_real @ anchor_real.T
        relation_logits = relation_logits + wave_imag @ anchor_imag.T
        relation_logits = relation_logits**2
        return _topk_soft_assign(relation_logits, relation_top_k, relation_temperature)

    def _query_aux_profile(relation_profile: np.ndarray) -> np.ndarray:
        query_profile = relation_profile - np.mean(
            relation_profile, axis=1, keepdims=True
        )
        return _safe_normalize(query_profile.astype(np.float32, copy=False))

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_scores = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = base_scores.shape[1]
        if doc_count == 0:
            return base_scores
        support_graph = doc_state["support_graph"]
        if support_graph.shape[0] != doc_count:
            return base_scores
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return base_scores
        support_idx = np.argpartition(-base_scores, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(base_scores, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_vector = np.zeros_like(base_scores, dtype=np.float32)
        np.put_along_axis(support_vector, support_idx, support_weights, axis=1)
        transport_graph = (
            1.0 - symmetry_mix
        ) * support_graph + symmetry_mix * support_graph.T
        transported_scores = support_vector @ transport_graph
        transported_scores = transported_scores - np.mean(
            transported_scores, axis=1, keepdims=True
        )
        rerank_gain = np.maximum(0.0, 1.0 + score_gain * transported_scores)
        return base_scores * rerank_gain

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag)
        doc_count = wave_real.shape[0]
        empty_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        if doc_count <= 1:
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "support_graph": empty_graph,
                "relation_profile": relation_profile.astype(np.float32),
                "aux_operator_profile": _query_aux_profile(relation_profile),
            }
        graph_scores = _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
        np.fill_diagonal(graph_scores, -np.inf)
        keep = min(graph_k, doc_count - 1)
        if keep <= 0:
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "support_graph": empty_graph,
                "relation_profile": relation_profile.astype(np.float32),
                "aux_operator_profile": _query_aux_profile(relation_profile),
            }
        neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)

        # Der Zusatzoperator misst transportierte Community-Kontexte statt nur die nackte Carrier-Geometrie.
        transport_graph = (
            1.0 - symmetry_mix
        ) * support_graph + symmetry_mix * support_graph.T
        transported_profile = transport_graph @ relation_profile
        mixed_profile = (1.0 - transport_mix) * relation_profile
        mixed_profile = mixed_profile + transport_mix * transported_profile
        mixed_profile = mixed_profile - np.mean(mixed_profile, axis=1, keepdims=True)
        aux_operator_profile = _safe_normalize(
            mixed_profile.astype(np.float32, copy=False)
        )
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "support_graph": support_graph,
            "relation_profile": relation_profile.astype(np.float32),
            "aux_operator_profile": aux_operator_profile,
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "relation_profile": relation_profile.astype(np.float32),
            "aux_operator_profile": _query_aux_profile(relation_profile),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )

    return EmbeddingStateMethod(
        method_name="projective_graph_transport_observable_v0",
        family="state_projective_graph_transport_observable",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "graph_temperature": graph_temperature,
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "transport_mix": transport_mix,
            "symmetry_mix": symmetry_mix,
            "support_width": float(support_width),
            "support_temperature": support_temperature,
            "score_gain": score_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _projective_observer_subgraph_transport_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_hilbert_embedding_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    relation_slots = max(8, int(params.get("relation_slots", 16)))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 2))))
    relation_temperature = float(params.get("relation_temperature", 0.08))
    transport_mix = float(params.get("transport_mix", 0.68))
    support_width = max(graph_k, int(params.get("support_width", 8)))
    support_temperature = float(params.get("support_temperature", 0.10))
    observer_channels = max(2, int(params.get("observer_channels", 4)))
    observer_dim = max(12, int(params.get("observer_dim", 24)))
    observer_gain = float(params.get("observer_gain", 0.45))
    observer_temperature = float(params.get("observer_temperature", 0.16))
    observer_edge_floor = float(params.get("observer_edge_floor", 0.18))
    profile_mix = float(params.get("profile_mix", 0.35))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))

    carrier_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    local_rng = np.random.default_rng(
        _method_seed(
            "projective_observer_subgraph_transport_observable_v0", secret_key, dim
        )
    )
    real_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    phase_bias = carrier_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = carrier_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    anchor_real = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )
    anchor_imag = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )
    observer_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, observer_dim) for _ in range(observer_channels)]
    )
    observer_mod_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, observer_dim) for _ in range(observer_channels)]
    )
    observer_bias = local_rng.uniform(
        -math.pi, math.pi, size=(observer_channels, observer_dim)
    ).astype(np.float32)
    observer_gate = np.abs(
        local_rng.normal(size=(observer_channels,)).astype(np.float32)
    )
    observer_gate = observer_gate / np.maximum(1e-6, np.sum(observer_gate))

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _relation_profile(wave_real: np.ndarray, wave_imag: np.ndarray) -> np.ndarray:
        relation_logits = wave_real @ anchor_real.T
        relation_logits = relation_logits + wave_imag @ anchor_imag.T
        relation_logits = relation_logits**2
        return _topk_soft_assign(relation_logits, relation_top_k, relation_temperature)

    def _observer_hidden(x: np.ndarray) -> np.ndarray:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=False))
        channels = []
        for channel in range(observer_channels):
            base = y @ observer_proj[channel]
            mod = np.sin(y @ observer_mod_proj[channel] + observer_bias[channel])
            hidden = _safe_normalize(base + observer_gain * mod)
            channels.append(hidden.astype(np.float32))
        return np.stack(channels, axis=1)

    def _observer_summary(observer_hidden: np.ndarray) -> np.ndarray:
        summary = np.mean(np.maximum(observer_hidden, 0.0), axis=2)
        summary = summary * observer_gate[None, :]
        return summary.astype(np.float32, copy=False)

    def _aux_profile(
        relation_profile: np.ndarray,
        observer_hidden: np.ndarray,
        transport_graph: np.ndarray | None,
    ) -> np.ndarray:
        mixed_profile = np.array(relation_profile, dtype=np.float32, copy=True)
        if (
            transport_graph is not None
            and transport_graph.shape[0] == relation_profile.shape[0]
        ):
            transported = transport_graph @ relation_profile
            mixed_profile = (
                1.0 - transport_mix
            ) * mixed_profile + transport_mix * transported
        observer_profile = _observer_summary(observer_hidden)
        merged = np.concatenate(
            [mixed_profile, profile_mix * observer_profile], axis=1
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _channel_weights(
        query_observer: np.ndarray,
        support_observer: np.ndarray,
        support_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        support_response = np.maximum(
            np.sum(query_observer[:, None, :, :] * support_observer, axis=3), 0.0
        ).astype(np.float32)
        channel_weight = np.sum(
            support_response * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        channel_weight = channel_weight * observer_gate[None, :]
        shifted = channel_weight - np.max(channel_weight, axis=1, keepdims=True)
        channel_weight = np.exp(shifted / max(1e-4, observer_temperature)).astype(
            np.float32
        )
        total = np.sum(channel_weight, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        channel_weight = channel_weight / total
        return support_response, channel_weight

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_scores = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = base_scores.shape[1]
        if doc_count == 0:
            return base_scores
        transport_graph = doc_state["transport_graph"]
        doc_observer = doc_state["observer_channels"]
        if transport_graph.shape != (doc_count, doc_count):
            return base_scores
        if doc_observer.shape[0] != doc_count:
            return base_scores
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return base_scores
        support_idx = np.argpartition(-base_scores, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(base_scores, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_rows = transport_graph[support_idx]
        support_observer = doc_observer[support_idx]
        query_observer = query_state["observer_channels"]
        support_response, channel_weight = _channel_weights(
            query_observer, support_observer, support_weights
        )
        candidate_response = np.maximum(
            np.sum(query_observer[:, None, :, :] * doc_observer[None, :, :, :], axis=3),
            0.0,
        ).astype(np.float32)
        weighted_support = np.sqrt(np.maximum(support_response, 0.0)) * np.sqrt(
            channel_weight[:, None, :]
        )
        weighted_candidate = np.sqrt(np.maximum(candidate_response, 0.0)) * np.sqrt(
            channel_weight[:, None, :]
        )
        observer_filter = np.einsum(
            "qsc,qdc->qsd", weighted_support, weighted_candidate, dtype=np.float32
        )
        observer_filter = np.clip(observer_filter, 0.0, 1.0)
        filtered_rows = support_rows * (
            observer_edge_floor + (1.0 - observer_edge_floor) * observer_filter
        )
        transport_signal = np.sum(
            filtered_rows * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        relation_signal = (
            query_state["relation_profile"] @ doc_state["relation_profile"].T
        ).astype(np.float32)
        aux_scores = (
            1.0 - transport_mix
        ) * relation_signal + transport_mix * transport_signal
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return aux_scores / scale

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag).astype(np.float32)
        observer_hidden = _observer_hidden(x)
        doc_count = wave_real.shape[0]
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        transport_graph = support_graph
        if doc_count > 1:
            graph_scores = _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
            np.fill_diagonal(graph_scores, -np.inf)
            keep = min(graph_k, doc_count - 1)
            if keep > 0:
                neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[
                    :, :keep
                ]
                neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
                shifted = neighbor_scores - np.max(
                    neighbor_scores, axis=1, keepdims=True
                )
                neighbor_weights = np.exp(
                    shifted / max(1e-4, graph_temperature)
                ).astype(np.float32)
                weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
                weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
                neighbor_weights = neighbor_weights / weight_sum
                np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
                transport_graph = 0.5 * (support_graph + support_graph.T)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "support_graph": support_graph,
            "transport_graph": transport_graph,
            "relation_profile": relation_profile,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
            "aux_operator_profile": _aux_profile(
                relation_profile, observer_hidden, transport_graph
            ),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag).astype(np.float32)
        observer_hidden = _observer_hidden(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "relation_profile": relation_profile,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
            "aux_operator_profile": _aux_profile(
                relation_profile, observer_hidden, None
            ),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )

    return EmbeddingStateMethod(
        method_name="projective_observer_subgraph_transport_observable_v0",
        family="state_projective_observer_subgraph_transport_observable",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "graph_temperature": graph_temperature,
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "transport_mix": transport_mix,
            "support_width": float(support_width),
            "support_temperature": support_temperature,
            "observer_channels": float(observer_channels),
            "observer_dim": float(observer_dim),
            "observer_gain": observer_gain,
            "observer_temperature": observer_temperature,
            "observer_edge_floor": observer_edge_floor,
            "profile_mix": profile_mix,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _projective_observer_subgraph_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    hybrid_gain = float(params.get("hybrid_gain", 0.012))
    rerank_width = max(8, int(params.get("rerank_width", 24)))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    operator_method = _projective_observer_subgraph_transport_observable_build(
        rng, params
    )
    aux_score = operator_method.aux_score

    if aux_score is None:
        return EmbeddingStateMethod(
            method_name="projective_observer_subgraph_head_v0",
            family="state_projective_observer_subgraph_head",
            params={
                **operator_method.params,
                "hybrid_gain": hybrid_gain,
                "rerank_width": float(rerank_width),
                "uncertainty_width": uncertainty_width,
            },
            encode_docs=operator_method.encode_docs,
            encode_queries=operator_method.encode_queries,
            score=operator_method.score,
        )

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = operator_method.score(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        operator_scores = aux_score(doc_state, query_state)
        if operator_scores.shape != carrier.shape:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_operator = np.take_along_axis(operator_scores, candidate_idx, axis=1)
        candidate_operator = candidate_operator - np.mean(
            candidate_operator, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_operator), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_operator = candidate_operator / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        rerank_gain = np.maximum(
            0.0, 1.0 + hybrid_gain * uncertainty_gate * candidate_operator
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_observer_subgraph_head_v0",
        family="state_projective_observer_subgraph_head",
        params={
            **operator_method.params,
            "hybrid_gain": hybrid_gain,
            "rerank_width": float(rerank_width),
            "uncertainty_width": uncertainty_width,
        },
        encode_docs=operator_method.encode_docs,
        encode_queries=operator_method.encode_queries,
        score=_score,
        aux_score=aux_score,
    )


def _projective_observer_ambiguity_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    hybrid_gain = float(params.get("hybrid_gain", 0.0015))
    rerank_width = max(8, int(params.get("rerank_width", 8)))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    ambiguity_quantile = float(params.get("ambiguity_quantile", 0.5))
    ambiguity_power = float(params.get("ambiguity_power", 0.6))
    operator_method = _projective_observer_subgraph_transport_observable_build(
        rng, params
    )
    aux_score = operator_method.aux_score

    if aux_score is None:
        return EmbeddingStateMethod(
            method_name="projective_observer_ambiguity_head_v0",
            family="state_projective_observer_ambiguity_head",
            params={
                **operator_method.params,
                "hybrid_gain": hybrid_gain,
                "rerank_width": float(rerank_width),
                "uncertainty_width": uncertainty_width,
                "ambiguity_quantile": ambiguity_quantile,
                "ambiguity_power": ambiguity_power,
            },
            encode_docs=operator_method.encode_docs,
            encode_queries=operator_method.encode_queries,
            score=operator_method.score,
        )

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = operator_method.score(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        operator_scores = aux_score(doc_state, query_state)
        if operator_scores.shape != carrier.shape:
            return carrier
        observer_query = _observer_query_ambiguity(
            doc_state,
            query_state,
            carrier,
            operator_method.params,
        )
        ambiguity = observer_query.get("per_query_ambiguity")
        if ambiguity is None:
            return carrier
        if ambiguity.shape[0] != carrier.shape[0]:
            return carrier
        threshold = float(
            np.quantile(np.asarray(ambiguity, dtype=np.float32), ambiguity_quantile)
        )
        ambiguity_scale = float(np.max(ambiguity) - threshold)
        if ambiguity_scale <= 1e-6:
            return carrier
        ambiguity_gate = np.clip((ambiguity - threshold) / ambiguity_scale, 0.0, 1.0)
        ambiguity_gate = np.power(ambiguity_gate, max(1e-4, ambiguity_power)).astype(
            np.float32
        )[:, None]
        if not np.any(ambiguity_gate > 0.0):
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_operator = np.take_along_axis(operator_scores, candidate_idx, axis=1)
        candidate_operator = candidate_operator - np.mean(
            candidate_operator, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_operator), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_operator = candidate_operator / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        rerank_gain = np.maximum(
            0.0,
            1.0 + hybrid_gain * uncertainty_gate * ambiguity_gate * candidate_operator,
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_observer_ambiguity_head_v0",
        family="state_projective_observer_ambiguity_head",
        params={
            **operator_method.params,
            "hybrid_gain": hybrid_gain,
            "rerank_width": float(rerank_width),
            "uncertainty_width": uncertainty_width,
            "ambiguity_quantile": ambiguity_quantile,
            "ambiguity_power": ambiguity_power,
        },
        encode_docs=operator_method.encode_docs,
        encode_queries=operator_method.encode_queries,
        score=_score,
        aux_score=aux_score,
    )


def _projective_observer_margin_coherence_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    hybrid_gain = float(params.get("hybrid_gain", 0.0012))
    rerank_width = max(8, int(params.get("rerank_width", 8)))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    ambiguity_quantile = float(params.get("ambiguity_quantile", 0.5))
    ambiguity_power = float(params.get("ambiguity_power", 0.6))
    coherence_floor = float(params.get("coherence_floor", 0.58))
    coherence_power = float(params.get("coherence_power", 1.2))
    operator_method = _projective_observer_subgraph_transport_observable_build(
        rng, params
    )
    aux_score = operator_method.aux_score

    if aux_score is None:
        return EmbeddingStateMethod(
            method_name="projective_observer_margin_coherence_head_v0",
            family="state_projective_observer_margin_coherence_head",
            params={
                **operator_method.params,
                "hybrid_gain": hybrid_gain,
                "rerank_width": float(rerank_width),
                "uncertainty_width": uncertainty_width,
                "ambiguity_quantile": ambiguity_quantile,
                "ambiguity_power": ambiguity_power,
                "coherence_floor": coherence_floor,
                "coherence_power": coherence_power,
            },
            encode_docs=operator_method.encode_docs,
            encode_queries=operator_method.encode_queries,
            score=operator_method.score,
        )

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = operator_method.score(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        operator_scores = aux_score(doc_state, query_state)
        if operator_scores.shape != carrier.shape:
            return carrier
        observer_query = _observer_query_ambiguity(
            doc_state,
            query_state,
            carrier,
            operator_method.params,
        )
        ambiguity = observer_query.get("per_query_ambiguity")
        if ambiguity is None:
            return carrier
        if ambiguity.shape[0] != carrier.shape[0]:
            return carrier
        threshold = float(
            np.quantile(np.asarray(ambiguity, dtype=np.float32), ambiguity_quantile)
        )
        ambiguity_scale = float(np.max(ambiguity) - threshold)
        if ambiguity_scale <= 1e-6:
            return carrier
        ambiguity_gate = np.clip((ambiguity - threshold) / ambiguity_scale, 0.0, 1.0)
        ambiguity_gate = np.power(ambiguity_gate, max(1e-4, ambiguity_power)).astype(
            np.float32
        )[:, None]
        if not np.any(ambiguity_gate > 0.0):
            return carrier
        relation_profile = doc_state.get("relation_profile")
        query_profile = query_state.get("relation_profile")
        if relation_profile is None or query_profile is None:
            return carrier
        if relation_profile.shape[0] != doc_count:
            return carrier
        if query_profile.shape[0] != carrier.shape[0]:
            return carrier
        support_width = min(
            max(1, int(operator_method.params.get("support_width", 8))),
            doc_count,
        )
        if support_width <= 0:
            return carrier
        support_temperature = max(
            1e-4, float(operator_method.params.get("support_temperature", 0.10))
        )
        support_idx = np.argpartition(-carrier, support_width - 1, axis=1)[
            :, :support_width
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_profiles = relation_profile[support_idx]
        support_context = np.sum(
            support_profiles * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        query_alignment = np.sum(
            query_profile * support_context, axis=1, keepdims=True, dtype=np.float32
        )
        support_focus = np.sum(
            support_context * support_context,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        coherence = np.sqrt(np.clip(query_alignment * support_focus, 0.0, 1.0)).astype(
            np.float32
        )
        coherence_gate = np.clip(
            (coherence - coherence_floor) / max(1e-4, 1.0 - coherence_floor),
            0.0,
            1.0,
        )
        coherence_gate = np.power(coherence_gate, max(1e-4, coherence_power)).astype(
            np.float32
        )
        if not np.any(coherence_gate > 0.0):
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_operator = np.take_along_axis(operator_scores, candidate_idx, axis=1)
        candidate_operator = candidate_operator - np.mean(
            candidate_operator, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_operator), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_operator = candidate_operator / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        combined_gate = uncertainty_gate * ambiguity_gate * coherence_gate
        rerank_gain = np.maximum(
            0.0,
            1.0 + hybrid_gain * combined_gate * candidate_operator,
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_observer_margin_coherence_head_v0",
        family="state_projective_observer_margin_coherence_head",
        params={
            **operator_method.params,
            "hybrid_gain": hybrid_gain,
            "rerank_width": float(rerank_width),
            "uncertainty_width": uncertainty_width,
            "ambiguity_quantile": ambiguity_quantile,
            "ambiguity_power": ambiguity_power,
            "coherence_floor": coherence_floor,
            "coherence_power": coherence_power,
        },
        encode_docs=operator_method.encode_docs,
        encode_queries=operator_method.encode_queries,
        score=_score,
        aux_score=aux_score,
    )


def _projective_observer_prototype_margin_coherence_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    hybrid_gain = float(params.get("hybrid_gain", 0.0012))
    rerank_width = max(8, int(params.get("rerank_width", 8)))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    ambiguity_quantile = float(params.get("ambiguity_quantile", 0.5))
    ambiguity_power = float(params.get("ambiguity_power", 0.6))
    coherence_floor = float(params.get("coherence_floor", 0.58))
    coherence_power = float(params.get("coherence_power", 1.2))
    operator_method = _projective_observer_prototype_residual_routing_observable_build(
        rng, params
    )
    aux_score = operator_method.aux_score

    if aux_score is None:
        return EmbeddingStateMethod(
            method_name="projective_observer_prototype_margin_coherence_head_v0",
            family="state_projective_observer_prototype_margin_coherence_head",
            params={
                **operator_method.params,
                "hybrid_gain": hybrid_gain,
                "rerank_width": float(rerank_width),
                "uncertainty_width": uncertainty_width,
                "ambiguity_quantile": ambiguity_quantile,
                "ambiguity_power": ambiguity_power,
                "coherence_floor": coherence_floor,
                "coherence_power": coherence_power,
            },
            encode_docs=operator_method.encode_docs,
            encode_queries=operator_method.encode_queries,
            score=operator_method.score,
        )

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = operator_method.score(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        operator_scores = aux_score(doc_state, query_state)
        if operator_scores.shape != carrier.shape:
            return carrier
        observer_query = _observer_query_ambiguity(
            doc_state,
            query_state,
            carrier,
            operator_method.params,
        )
        ambiguity = observer_query.get("per_query_ambiguity")
        if ambiguity is None:
            return carrier
        if ambiguity.shape[0] != carrier.shape[0]:
            return carrier
        threshold = float(
            np.quantile(np.asarray(ambiguity, dtype=np.float32), ambiguity_quantile)
        )
        ambiguity_scale = float(np.max(ambiguity) - threshold)
        if ambiguity_scale <= 1e-6:
            return carrier
        ambiguity_gate = np.clip((ambiguity - threshold) / ambiguity_scale, 0.0, 1.0)
        ambiguity_gate = np.power(ambiguity_gate, max(1e-4, ambiguity_power)).astype(
            np.float32
        )[:, None]
        if not np.any(ambiguity_gate > 0.0):
            return carrier
        relation_profile = doc_state.get("relation_profile")
        query_profile = query_state.get("relation_profile")
        if relation_profile is None or query_profile is None:
            return carrier
        if relation_profile.shape[0] != doc_count:
            return carrier
        if query_profile.shape[0] != carrier.shape[0]:
            return carrier
        support_width = min(
            max(1, int(operator_method.params.get("support_width", 8))),
            doc_count,
        )
        if support_width <= 0:
            return carrier
        support_temperature = max(
            1e-4, float(operator_method.params.get("support_temperature", 0.10))
        )
        support_idx = np.argpartition(-carrier, support_width - 1, axis=1)[
            :, :support_width
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_profiles = relation_profile[support_idx]
        support_context = np.sum(
            support_profiles * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        query_alignment = np.sum(
            query_profile * support_context, axis=1, keepdims=True, dtype=np.float32
        )
        support_focus = np.sum(
            support_context * support_context,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        coherence = np.sqrt(np.clip(query_alignment * support_focus, 0.0, 1.0)).astype(
            np.float32
        )
        coherence_gate = np.clip(
            (coherence - coherence_floor) / max(1e-4, 1.0 - coherence_floor),
            0.0,
            1.0,
        )
        coherence_gate = np.power(coherence_gate, max(1e-4, coherence_power)).astype(
            np.float32
        )
        if not np.any(coherence_gate > 0.0):
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_operator = np.take_along_axis(operator_scores, candidate_idx, axis=1)
        candidate_operator = candidate_operator - np.mean(
            candidate_operator, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_operator), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_operator = candidate_operator / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        combined_gate = uncertainty_gate * ambiguity_gate * coherence_gate
        rerank_gain = np.maximum(
            0.0,
            1.0 + hybrid_gain * combined_gate * candidate_operator,
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_observer_prototype_margin_coherence_head_v0",
        family="state_projective_observer_prototype_margin_coherence_head",
        params={
            **operator_method.params,
            "hybrid_gain": hybrid_gain,
            "rerank_width": float(rerank_width),
            "uncertainty_width": uncertainty_width,
            "ambiguity_quantile": ambiguity_quantile,
            "ambiguity_power": ambiguity_power,
            "coherence_floor": coherence_floor,
            "coherence_power": coherence_power,
        },
        encode_docs=operator_method.encode_docs,
        encode_queries=operator_method.encode_queries,
        score=_score,
        aux_score=aux_score,
    )


def _projective_observer_family_selective_residual_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    hybrid_gain = float(params.get("hybrid_gain", 0.006))
    rerank_width = max(8, int(params.get("rerank_width", 8)))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    ambiguity_quantile = float(params.get("ambiguity_quantile", 0.5))
    ambiguity_power = float(params.get("ambiguity_power", 0.6))
    coherence_floor = float(params.get("coherence_floor", 0.35))
    coherence_power = float(params.get("coherence_power", 1.2))
    family_focus_floor = float(params.get("family_focus_floor", 0.18))
    family_focus_power = float(params.get("family_focus_power", 1.0))
    disagreement_floor = float(params.get("disagreement_floor", 0.12))
    disagreement_power = float(params.get("disagreement_power", 1.0))
    base_method = _projective_observer_subgraph_transport_observable_build(rng, params)
    prototype_method = _projective_observer_prototype_residual_routing_observable_build(
        rng, params
    )
    base_aux_score = base_method.aux_score
    prototype_aux_score = prototype_method.aux_score

    if base_aux_score is None or prototype_aux_score is None:
        return EmbeddingStateMethod(
            method_name="projective_observer_family_selective_residual_head_v0",
            family="state_projective_observer_family_selective_residual_head",
            params={
                **prototype_method.params,
                "hybrid_gain": hybrid_gain,
                "rerank_width": float(rerank_width),
                "uncertainty_width": uncertainty_width,
                "ambiguity_quantile": ambiguity_quantile,
                "ambiguity_power": ambiguity_power,
                "coherence_floor": coherence_floor,
                "coherence_power": coherence_power,
                "family_focus_floor": family_focus_floor,
                "family_focus_power": family_focus_power,
                "disagreement_floor": disagreement_floor,
                "disagreement_power": disagreement_power,
            },
            encode_docs=prototype_method.encode_docs,
            encode_queries=prototype_method.encode_queries,
            score=prototype_method.score,
        )

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = prototype_method.score(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        base_scores = base_aux_score(doc_state, query_state)
        prototype_scores = prototype_aux_score(doc_state, query_state)
        if base_scores.shape != carrier.shape:
            return carrier
        if prototype_scores.shape != carrier.shape:
            return carrier

        observer_query = _observer_query_ambiguity(
            doc_state,
            query_state,
            carrier,
            prototype_method.params,
        )
        ambiguity = observer_query.get("per_query_ambiguity")
        if ambiguity is None:
            return carrier
        if ambiguity.shape[0] != carrier.shape[0]:
            return carrier
        threshold = float(
            np.quantile(np.asarray(ambiguity, dtype=np.float32), ambiguity_quantile)
        )
        ambiguity_scale = float(np.max(ambiguity) - threshold)
        if ambiguity_scale <= 1e-6:
            return carrier
        ambiguity_gate = np.clip((ambiguity - threshold) / ambiguity_scale, 0.0, 1.0)
        ambiguity_gate = np.power(ambiguity_gate, max(1e-4, ambiguity_power)).astype(
            np.float32
        )[:, None]
        if not np.any(ambiguity_gate > 0.0):
            return carrier

        relation_profile = doc_state.get("relation_profile")
        query_profile = query_state.get("relation_profile")
        if relation_profile is None or query_profile is None:
            return carrier
        if relation_profile.shape[0] != doc_count:
            return carrier
        if query_profile.shape[0] != carrier.shape[0]:
            return carrier
        support_width = min(
            max(1, int(prototype_method.params.get("support_width", 8))),
            doc_count,
        )
        if support_width <= 0:
            return carrier
        support_temperature = max(
            1e-4, float(prototype_method.params.get("support_temperature", 0.10))
        )
        support_idx = np.argpartition(-carrier, support_width - 1, axis=1)[
            :, :support_width
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_profiles = relation_profile[support_idx]
        support_context = np.sum(
            support_profiles * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        query_alignment = np.sum(
            query_profile * support_context, axis=1, keepdims=True, dtype=np.float32
        )
        support_focus = np.sum(
            support_context * support_context,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        coherence = np.sqrt(np.clip(query_alignment * support_focus, 0.0, 1.0)).astype(
            np.float32
        )
        coherence_gate = np.clip(
            (coherence - coherence_floor) / max(1e-4, 1.0 - coherence_floor),
            0.0,
            1.0,
        )
        coherence_gate = np.power(coherence_gate, max(1e-4, coherence_power)).astype(
            np.float32
        )
        if not np.any(coherence_gate > 0.0):
            return carrier

        semantic_codes = doc_state.get("semantic_codes")
        semantic_centers = doc_state.get("semantic_centers")
        semantic_config = doc_state.get("semantic_code_config")
        query_energy = query_state.get("energy")
        if (
            semantic_codes is None
            or semantic_centers is None
            or semantic_config is None
        ):
            return carrier
        if query_energy is None:
            return carrier
        if semantic_codes.shape[0] != doc_count:
            return carrier
        if semantic_centers.shape[0] == 0:
            return carrier
        local_top_k = min(
            semantic_centers.shape[0],
            max(1, int(round(float(semantic_config[0])))),
        )
        local_temperature = max(1e-4, float(semantic_config[1]))
        query_codes = _topk_soft_assign(
            query_energy @ semantic_centers.T,
            local_top_k,
            local_temperature,
        )
        support_codes = semantic_codes[support_idx]
        support_family = np.sum(
            support_codes * support_weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        support_family = _safe_normalize(support_family)
        query_peak = np.max(query_codes, axis=1, keepdims=True)
        family_alignment = np.sum(
            query_codes * support_family,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        family_focus = np.sqrt(
            np.clip(query_peak * np.maximum(family_alignment, 0.0), 0.0, 1.0)
        ).astype(np.float32)
        family_focus_gate = np.clip(
            (family_focus - family_focus_floor) / max(1e-4, 1.0 - family_focus_floor),
            0.0,
            1.0,
        )
        family_focus_gate = np.power(
            family_focus_gate, max(1e-4, family_focus_power)
        ).astype(np.float32)
        if not np.any(family_focus_gate > 0.0):
            return carrier

        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_base = np.take_along_axis(base_scores, candidate_idx, axis=1)
        candidate_prototype = np.take_along_axis(
            prototype_scores, candidate_idx, axis=1
        )
        candidate_delta = candidate_prototype - candidate_base
        candidate_delta = candidate_delta - np.mean(
            candidate_delta, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_delta), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_delta = candidate_delta / signal_scale
        disagreement_gate = np.clip(
            (np.abs(candidate_delta) - disagreement_floor)
            / max(1e-4, 1.0 - disagreement_floor),
            0.0,
            1.0,
        )
        disagreement_gate = np.power(
            disagreement_gate, max(1e-4, disagreement_power)
        ).astype(np.float32)
        if not np.any(disagreement_gate > 0.0):
            return carrier

        candidate_codes = semantic_codes[candidate_idx]
        candidate_alignment = np.sum(
            candidate_codes * support_family[:, None, :],
            axis=2,
            dtype=np.float32,
        )
        candidate_family_gate = np.sqrt(
            np.clip(np.maximum(candidate_alignment, 0.0), 0.0, 1.0)
        ).astype(np.float32)
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        combined_gate = (
            uncertainty_gate
            * ambiguity_gate
            * coherence_gate
            * family_focus_gate
            * candidate_family_gate
            * disagreement_gate
        )
        rerank_gain = np.maximum(
            0.0,
            1.0 + hybrid_gain * combined_gate * candidate_delta,
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_observer_family_selective_residual_head_v0",
        family="state_projective_observer_family_selective_residual_head",
        params={
            **prototype_method.params,
            "hybrid_gain": hybrid_gain,
            "rerank_width": float(rerank_width),
            "uncertainty_width": uncertainty_width,
            "ambiguity_quantile": ambiguity_quantile,
            "ambiguity_power": ambiguity_power,
            "coherence_floor": coherence_floor,
            "coherence_power": coherence_power,
            "family_focus_floor": family_focus_floor,
            "family_focus_power": family_focus_power,
            "disagreement_floor": disagreement_floor,
            "disagreement_power": disagreement_power,
        },
        encode_docs=prototype_method.encode_docs,
        encode_queries=prototype_method.encode_queries,
        score=_score,
        aux_score=prototype_aux_score,
    )


def _projective_observer_family_order_flip_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    hybrid_gain = float(params.get("hybrid_gain", 0.006))
    rerank_width = max(8, int(params.get("rerank_width", 8)))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    ambiguity_quantile = float(params.get("ambiguity_quantile", 0.5))
    ambiguity_power = float(params.get("ambiguity_power", 0.6))
    coherence_floor = float(params.get("coherence_floor", 0.35))
    coherence_power = float(params.get("coherence_power", 1.2))
    family_focus_floor = float(params.get("family_focus_floor", 0.18))
    family_focus_power = float(params.get("family_focus_power", 1.0))
    disagreement_floor = float(params.get("disagreement_floor", 0.12))
    disagreement_power = float(params.get("disagreement_power", 1.0))
    family_temperature = float(params.get("family_temperature", 0.10))
    flip_shift_floor = float(params.get("flip_shift_floor", 0.08))
    flip_shift_power = float(params.get("flip_shift_power", 1.0))
    competition_margin_floor = float(params.get("competition_margin_floor", 0.04))
    competition_margin_power = float(params.get("competition_margin_power", 1.0))
    base_method = _projective_observer_subgraph_transport_observable_build(rng, params)
    prototype_method = _projective_observer_prototype_residual_routing_observable_build(
        rng, params
    )
    base_aux_score = base_method.aux_score
    prototype_aux_score = prototype_method.aux_score

    if base_aux_score is None or prototype_aux_score is None:
        return EmbeddingStateMethod(
            method_name="projective_observer_family_order_flip_head_v0",
            family="state_projective_observer_family_order_flip_head",
            params={
                **prototype_method.params,
                "hybrid_gain": hybrid_gain,
                "rerank_width": float(rerank_width),
                "uncertainty_width": uncertainty_width,
                "ambiguity_quantile": ambiguity_quantile,
                "ambiguity_power": ambiguity_power,
                "coherence_floor": coherence_floor,
                "coherence_power": coherence_power,
                "family_focus_floor": family_focus_floor,
                "family_focus_power": family_focus_power,
                "disagreement_floor": disagreement_floor,
                "disagreement_power": disagreement_power,
                "family_temperature": family_temperature,
                "flip_shift_floor": flip_shift_floor,
                "flip_shift_power": flip_shift_power,
                "competition_margin_floor": competition_margin_floor,
                "competition_margin_power": competition_margin_power,
            },
            encode_docs=prototype_method.encode_docs,
            encode_queries=prototype_method.encode_queries,
            score=prototype_method.score,
        )

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = prototype_method.score(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier

        base_scores = base_aux_score(doc_state, query_state)
        prototype_scores = prototype_aux_score(doc_state, query_state)
        if base_scores.shape != carrier.shape:
            return carrier
        if prototype_scores.shape != carrier.shape:
            return carrier

        observer_query = _observer_query_ambiguity(
            doc_state,
            query_state,
            carrier,
            prototype_method.params,
        )
        ambiguity = observer_query.get("per_query_ambiguity")
        if ambiguity is None:
            return carrier
        if ambiguity.shape[0] != carrier.shape[0]:
            return carrier
        threshold = float(
            np.quantile(np.asarray(ambiguity, dtype=np.float32), ambiguity_quantile)
        )
        ambiguity_scale = float(np.max(ambiguity) - threshold)
        if ambiguity_scale <= 1e-6:
            return carrier
        ambiguity_gate = np.clip((ambiguity - threshold) / ambiguity_scale, 0.0, 1.0)
        ambiguity_gate = np.power(ambiguity_gate, max(1e-4, ambiguity_power)).astype(
            np.float32
        )[:, None]
        if not np.any(ambiguity_gate > 0.0):
            return carrier

        relation_profile = doc_state.get("relation_profile")
        query_profile = query_state.get("relation_profile")
        if relation_profile is None or query_profile is None:
            return carrier
        if relation_profile.shape[0] != doc_count:
            return carrier
        if query_profile.shape[0] != carrier.shape[0]:
            return carrier
        support_width = min(
            max(1, int(prototype_method.params.get("support_width", 8))),
            doc_count,
        )
        if support_width <= 0:
            return carrier
        support_temperature = max(
            1e-4, float(prototype_method.params.get("support_temperature", 0.10))
        )
        support_idx = np.argpartition(-carrier, support_width - 1, axis=1)[
            :, :support_width
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_profiles = relation_profile[support_idx]
        support_context = np.sum(
            support_profiles * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        query_alignment = np.sum(
            query_profile * support_context, axis=1, keepdims=True, dtype=np.float32
        )
        support_focus = np.sum(
            support_context * support_context,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        coherence = np.sqrt(np.clip(query_alignment * support_focus, 0.0, 1.0)).astype(
            np.float32
        )
        coherence_gate = np.clip(
            (coherence - coherence_floor) / max(1e-4, 1.0 - coherence_floor),
            0.0,
            1.0,
        )
        coherence_gate = np.power(coherence_gate, max(1e-4, coherence_power)).astype(
            np.float32
        )
        if not np.any(coherence_gate > 0.0):
            return carrier

        semantic_codes = doc_state.get("semantic_codes")
        semantic_centers = doc_state.get("semantic_centers")
        semantic_config = doc_state.get("semantic_code_config")
        query_energy = query_state.get("energy")
        if (
            semantic_codes is None
            or semantic_centers is None
            or semantic_config is None
        ):
            return carrier
        if query_energy is None:
            return carrier
        if semantic_codes.shape[0] != doc_count:
            return carrier
        if semantic_centers.shape[0] == 0:
            return carrier
        local_top_k = min(
            semantic_centers.shape[0],
            max(1, int(round(float(semantic_config[0])))),
        )
        local_temperature = max(1e-4, float(semantic_config[1]))
        query_codes = _topk_soft_assign(
            query_energy @ semantic_centers.T,
            local_top_k,
            local_temperature,
        )
        support_codes = semantic_codes[support_idx]
        support_family = np.sum(
            support_codes * support_weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        support_family = _safe_normalize(support_family)
        query_peak = np.max(query_codes, axis=1, keepdims=True)
        family_alignment = np.sum(
            query_codes * support_family,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        family_focus = np.sqrt(
            np.clip(query_peak * np.maximum(family_alignment, 0.0), 0.0, 1.0)
        ).astype(np.float32)
        family_focus_gate = np.clip(
            (family_focus - family_focus_floor) / max(1e-4, 1.0 - family_focus_floor),
            0.0,
            1.0,
        )
        family_focus_gate = np.power(
            family_focus_gate, max(1e-4, family_focus_power)
        ).astype(np.float32)
        if not np.any(family_focus_gate > 0.0):
            return carrier

        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_base = np.take_along_axis(base_scores, candidate_idx, axis=1)
        candidate_prototype = np.take_along_axis(
            prototype_scores, candidate_idx, axis=1
        )
        candidate_delta = candidate_prototype - candidate_base
        candidate_delta = candidate_delta - np.mean(
            candidate_delta, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_delta), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_delta = candidate_delta / signal_scale
        disagreement_gate = np.clip(
            (np.abs(candidate_delta) - disagreement_floor)
            / max(1e-4, 1.0 - disagreement_floor),
            0.0,
            1.0,
        )
        disagreement_gate = np.power(
            disagreement_gate, max(1e-4, disagreement_power)
        ).astype(np.float32)
        if not np.any(disagreement_gate > 0.0):
            return carrier

        candidate_codes = semantic_codes[candidate_idx]
        family_temp = max(1e-4, family_temperature)
        base_shifted = candidate_base - np.max(candidate_base, axis=1, keepdims=True)
        base_weights = np.exp(base_shifted / family_temp).astype(np.float32)
        base_weight_sum = np.sum(base_weights, axis=1, keepdims=True)
        base_weight_sum = np.where(base_weight_sum == 0.0, 1.0, base_weight_sum)
        base_weights = base_weights / base_weight_sum
        prototype_shifted = candidate_prototype - np.max(
            candidate_prototype, axis=1, keepdims=True
        )
        prototype_weights = np.exp(prototype_shifted / family_temp).astype(np.float32)
        prototype_weight_sum = np.sum(prototype_weights, axis=1, keepdims=True)
        prototype_weight_sum = np.where(
            prototype_weight_sum == 0.0, 1.0, prototype_weight_sum
        )
        prototype_weights = prototype_weights / prototype_weight_sum

        base_family = np.sum(
            candidate_codes * base_weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        prototype_family = np.sum(
            candidate_codes * prototype_weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        base_family = _safe_normalize(base_family)
        prototype_family = _safe_normalize(prototype_family)

        base_top_idx = np.argmax(base_family, axis=1, keepdims=True)
        prototype_top_idx = np.argmax(prototype_family, axis=1, keepdims=True)
        top_family_flip = (base_top_idx != prototype_top_idx).astype(np.float32)
        prototype_top_mass = np.take_along_axis(
            prototype_family, prototype_top_idx, axis=1
        )
        base_proto_mass = np.take_along_axis(base_family, prototype_top_idx, axis=1)
        rank_shift = 0.5 * np.sum(
            np.abs(prototype_family - base_family),
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        flip_shift = np.maximum(
            np.maximum(0.0, prototype_top_mass - base_proto_mass),
            rank_shift,
        )
        flip_shift_gate = np.clip(
            (flip_shift - flip_shift_floor) / max(1e-4, 1.0 - flip_shift_floor),
            0.0,
            1.0,
        )
        flip_shift_gate = np.power(flip_shift_gate, max(1e-4, flip_shift_power)).astype(
            np.float32
        )
        family_flip_gate = np.maximum(top_family_flip, flip_shift_gate)
        if not np.any(flip_shift_gate > 0.0):
            return carrier

        if prototype_family.shape[1] <= 1:
            competition_gate = np.ones_like(flip_shift_gate, dtype=np.float32)
        else:
            base_family_sorted = np.sort(base_family, axis=1)
            prototype_family_sorted = np.sort(prototype_family, axis=1)
            base_margin = base_family_sorted[:, -1:] - base_family_sorted[:, -2:-1]
            prototype_margin = (
                prototype_family_sorted[:, -1:] - prototype_family_sorted[:, -2:-1]
            )
            competition_gain = np.maximum(0.0, prototype_margin - base_margin)
            competition_gate = np.clip(
                (competition_gain - competition_margin_floor)
                / max(1e-4, 1.0 - competition_margin_floor),
                0.0,
                1.0,
            )
            competition_gate = np.power(
                competition_gate, max(1e-4, competition_margin_power)
            ).astype(np.float32)
        if not np.any(competition_gate > 0.0):
            return carrier

        candidate_family_gate = np.sum(
            candidate_codes * prototype_family[:, None, :],
            axis=2,
            dtype=np.float32,
        )
        candidate_family_gate = np.sqrt(
            np.clip(np.maximum(candidate_family_gate, 0.0), 0.0, 1.0)
        ).astype(np.float32)
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        combined_gate = (
            uncertainty_gate
            * ambiguity_gate
            * coherence_gate
            * family_focus_gate
            * family_flip_gate
            * flip_shift_gate
            * competition_gate
            * candidate_family_gate
            * disagreement_gate
        )
        rerank_gain = np.maximum(
            0.0,
            1.0 + hybrid_gain * combined_gate * candidate_delta,
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_observer_family_order_flip_head_v0",
        family="state_projective_observer_family_order_flip_head",
        params={
            **prototype_method.params,
            "hybrid_gain": hybrid_gain,
            "rerank_width": float(rerank_width),
            "uncertainty_width": uncertainty_width,
            "ambiguity_quantile": ambiguity_quantile,
            "ambiguity_power": ambiguity_power,
            "coherence_floor": coherence_floor,
            "coherence_power": coherence_power,
            "family_focus_floor": family_focus_floor,
            "family_focus_power": family_focus_power,
            "disagreement_floor": disagreement_floor,
            "disagreement_power": disagreement_power,
            "family_temperature": family_temperature,
            "flip_shift_floor": flip_shift_floor,
            "flip_shift_power": flip_shift_power,
            "competition_margin_floor": competition_margin_floor,
            "competition_margin_power": competition_margin_power,
        },
        encode_docs=prototype_method.encode_docs,
        encode_queries=prototype_method.encode_queries,
        score=_score,
        aux_score=prototype_aux_score,
    )


def _projective_observer_community_support_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    codebook_size = max(8, int(params.get("codebook_size", 16)))
    code_top_k = min(codebook_size, max(1, int(params.get("code_top_k", 2))))
    code_temperature = float(params.get("code_temperature", 0.10))
    code_support_gain = float(params.get("code_support_gain", 0.30))
    code_focus_floor = float(params.get("code_focus_floor", 0.18))
    code_profile_mix = float(params.get("code_profile_mix", 0.20))
    codebook_builder = str(params.get("codebook_builder", "kmeans")).strip() or "kmeans"
    prototype_refine_steps = max(0, int(params.get("prototype_refine_steps", 1)))
    dim = int(params.get("dim", 0))
    secret_key = str(params.get("secret_key", ""))
    head_seed = _method_seed(
        "projective_observer_community_support_observable_v0", secret_key, dim
    ) % (2**32 - 1)
    operator_method = _projective_observer_subgraph_transport_observable_build(
        rng, params
    )
    aux_score = operator_method.aux_score

    if aux_score is None:
        return EmbeddingStateMethod(
            method_name="projective_observer_community_support_observable_v0",
            family="state_projective_observer_community_support_observable",
            params={
                **operator_method.params,
                "codebook_size": float(codebook_size),
                "code_top_k": float(code_top_k),
                "code_temperature": code_temperature,
                "code_support_gain": code_support_gain,
                "code_focus_floor": code_focus_floor,
                "code_profile_mix": code_profile_mix,
            },
            encode_docs=operator_method.encode_docs,
            encode_queries=operator_method.encode_queries,
            score=operator_method.score,
        )

    def _wave_energy(state: StateMap) -> np.ndarray:
        return (state["wave_real"] ** 2 + state["wave_imag"] ** 2).astype(np.float32)

    def _semantic_codes(energy: np.ndarray, centers: np.ndarray) -> np.ndarray:
        if centers.shape[0] == 0:
            return np.zeros((energy.shape[0], 0), dtype=np.float32)
        local_top_k = min(code_top_k, centers.shape[0])
        logits = energy @ centers.T
        return _topk_soft_assign(logits, local_top_k, code_temperature)

    def _merge_aux_profile(
        base_profile: np.ndarray | None, semantic_codes: np.ndarray
    ) -> np.ndarray:
        if base_profile is None:
            return _safe_normalize(semantic_codes.astype(np.float32, copy=False))
        if base_profile.shape[0] != semantic_codes.shape[0]:
            return base_profile.astype(np.float32, copy=False)
        merged = np.concatenate(
            [
                base_profile.astype(np.float32, copy=False),
                code_profile_mix * semantic_codes.astype(np.float32, copy=False),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _prototype_centers(
        energy: np.ndarray, effective_codebook_size: int
    ) -> np.ndarray:
        if effective_codebook_size <= 0:
            return np.zeros((0, energy.shape[1]), dtype=np.float32)
        prototype_rng = np.random.default_rng(
            _method_seed(
                "projective_observer_community_support_observable_v0_prototype_bank",
                secret_key,
                dim,
                offset=energy.shape[0],
            )
        )
        prototype_idx = np.sort(
            prototype_rng.choice(
                energy.shape[0], size=effective_codebook_size, replace=False
            )
        )
        centers = np.asarray(energy[prototype_idx], dtype=np.float32)
        centers = _safe_normalize(centers)
        if prototype_refine_steps <= 0:
            return centers
        for _ in range(prototype_refine_steps):
            logits = energy @ centers.T
            assign_idx = np.argmax(logits, axis=1)
            updated_centers = np.array(centers, copy=True)
            for center_idx in range(effective_codebook_size):
                member_mask = assign_idx == center_idx
                if not np.any(member_mask):
                    continue
                updated_centers[center_idx] = np.mean(
                    energy[member_mask], axis=0, dtype=np.float32
                )
            centers = _safe_normalize(updated_centers.astype(np.float32, copy=False))
        return centers

    def _semantic_centers(
        energy: np.ndarray, effective_codebook_size: int
    ) -> np.ndarray:
        if effective_codebook_size <= 0:
            return np.zeros((0, energy.shape[1]), dtype=np.float32)
        if codebook_builder == "keyed_prototypes":
            return _prototype_centers(energy, effective_codebook_size)
        kmeans = KMeans(
            n_clusters=effective_codebook_size,
            n_init=6,
            random_state=head_seed,
            max_iter=150,
        )
        kmeans.fit(energy)
        return _safe_normalize(np.asarray(kmeans.cluster_centers_, dtype=np.float32))

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(operator_method.encode_docs(x))
        energy = _wave_energy(state)
        state["energy"] = energy
        doc_count = energy.shape[0]
        if doc_count == 0:
            empty_codes = np.zeros((0, 0), dtype=np.float32)
            empty_centers = np.zeros((0, energy.shape[1]), dtype=np.float32)
            state["semantic_codes"] = empty_codes
            state["semantic_centers"] = empty_centers
            state["semantic_code_config"] = np.asarray(
                [float(code_top_k), code_temperature], dtype=np.float32
            )
            state["aux_operator_profile"] = empty_codes
            return state
        effective_codebook_size = min(codebook_size, doc_count)
        semantic_centers = _semantic_centers(energy, effective_codebook_size)
        semantic_codes = _semantic_codes(energy, semantic_centers).astype(np.float32)
        state["semantic_codes"] = semantic_codes
        state["semantic_centers"] = semantic_centers.astype(np.float32)
        state["semantic_code_config"] = np.asarray(
            [float(code_top_k), code_temperature], dtype=np.float32
        )
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), semantic_codes
        )
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        state = dict(operator_method.encode_queries(x))
        state["energy"] = _wave_energy(state)
        return state

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_aux = aux_score(doc_state, query_state)
        semantic_codes = doc_state.get("semantic_codes")
        semantic_centers = doc_state.get("semantic_centers")
        semantic_config = doc_state.get("semantic_code_config")
        if semantic_codes is None:
            return base_aux
        if semantic_centers is None:
            return base_aux
        if semantic_config is None:
            return base_aux
        if "energy" not in query_state:
            return base_aux
        doc_count = base_aux.shape[1]
        if semantic_codes.shape[0] != doc_count:
            return base_aux
        if semantic_centers.shape[0] == 0:
            return base_aux

        local_top_k = min(
            semantic_centers.shape[0],
            max(1, int(round(float(semantic_config[0])))),
        )
        local_temperature = max(1e-4, float(semantic_config[1]))
        query_codes = _topk_soft_assign(
            query_state["energy"] @ semantic_centers.T,
            local_top_k,
            local_temperature,
        )
        carrier = operator_method.score(doc_state, query_state)
        support_width = min(
            max(1, int(round(float(operator_method.params.get("support_width", 8))))),
            doc_count,
        )
        if support_width <= 0:
            return base_aux
        support_idx = np.argpartition(-carrier, support_width - 1, axis=1)[
            :, :support_width
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        support_temperature = max(
            1e-4, float(operator_method.params.get("support_temperature", 0.10))
        )
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_codes = semantic_codes[support_idx]
        support_context = np.sum(
            support_codes * support_weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        support_context = _safe_normalize(support_context)
        support_alignment = np.sum(
            query_codes[:, None, :] * support_codes,
            axis=2,
            dtype=np.float32,
        )
        candidate_alignment = np.sum(
            support_context[:, None, :] * semantic_codes[None, :, :],
            axis=2,
            dtype=np.float32,
        )
        community_support = np.sum(
            support_weights[:, :, None]
            * np.sqrt(np.clip(np.maximum(support_alignment, 0.0), 0.0, 1.0))[:, :, None]
            * np.sqrt(np.clip(np.maximum(candidate_alignment, 0.0), 0.0, 1.0))[
                :, None, :
            ],
            axis=1,
            dtype=np.float32,
        )
        community_support = community_support - np.mean(
            community_support, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(community_support), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        community_support = community_support / signal_scale
        query_peak = np.max(query_codes, axis=1, keepdims=True)
        family_alignment = np.sum(
            query_codes * support_context,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        focus = np.sqrt(
            np.clip(query_peak * np.maximum(family_alignment, 0.0), 0.0, 1.0)
        ).astype(np.float32)
        focus_gate = np.clip(
            (focus - code_focus_floor) / max(1e-4, 1.0 - code_focus_floor),
            0.0,
            1.0,
        )
        combined = base_aux + code_support_gain * focus_gate * community_support
        combined = combined - np.mean(combined, axis=1, keepdims=True)
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return combined / combined_scale

    return EmbeddingStateMethod(
        method_name="projective_observer_community_support_observable_v0",
        family="state_projective_observer_community_support_observable",
        params={
            **operator_method.params,
            "codebook_size": float(codebook_size),
            "code_top_k": float(code_top_k),
            "code_temperature": code_temperature,
            "code_support_gain": code_support_gain,
            "code_focus_floor": code_focus_floor,
            "code_profile_mix": code_profile_mix,
            "codebook_builder": codebook_builder,
            "prototype_refine_steps": float(prototype_refine_steps),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=operator_method.score,
        aux_score=_aux_score,
    )


def _projective_observer_prototype_support_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    local_params = dict(params)
    local_params.setdefault("codebook_builder", "keyed_prototypes")
    local_params.setdefault("prototype_refine_steps", 1)
    method = _projective_observer_community_support_observable_build(rng, local_params)
    return EmbeddingStateMethod(
        method_name="projective_observer_prototype_support_observable_v0",
        family="state_projective_observer_prototype_support_observable",
        params=dict(method.params),
        encode_docs=method.encode_docs,
        encode_queries=method.encode_queries,
        score=method.score,
        aux_score=method.aux_score,
    )


def _projective_observer_prototype_routing_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    route_mix = float(params.get("route_mix", 0.45))
    route_floor = float(params.get("route_floor", 0.35))
    local_params = dict(params)
    local_params.setdefault("codebook_builder", "keyed_prototypes")
    local_params.setdefault("prototype_refine_steps", 1)
    local_params.setdefault("code_profile_mix", 0.08)
    operator_method = _projective_observer_subgraph_transport_observable_build(
        rng, local_params
    )
    prototype_method = _projective_observer_prototype_support_observable_build(
        rng, local_params
    )
    base_aux_score = operator_method.aux_score

    if base_aux_score is None:
        return EmbeddingStateMethod(
            method_name="projective_observer_prototype_routing_observable_v0",
            family="state_projective_observer_prototype_routing_observable",
            params={
                **prototype_method.params,
                "route_mix": route_mix,
                "route_floor": route_floor,
            },
            encode_docs=prototype_method.encode_docs,
            encode_queries=prototype_method.encode_queries,
            score=prototype_method.score,
        )

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_aux = base_aux_score(doc_state, query_state)
        semantic_codes = doc_state.get("semantic_codes")
        semantic_centers = doc_state.get("semantic_centers")
        semantic_config = doc_state.get("semantic_code_config")
        if semantic_codes is None:
            return base_aux
        if semantic_centers is None:
            return base_aux
        if semantic_config is None:
            return base_aux
        if "energy" not in query_state:
            return base_aux
        doc_count = base_aux.shape[1]
        if semantic_codes.shape[0] != doc_count:
            return base_aux
        if semantic_centers.shape[0] == 0:
            return base_aux

        local_top_k = min(
            semantic_centers.shape[0],
            max(1, int(round(float(semantic_config[0])))),
        )
        local_temperature = max(1e-4, float(semantic_config[1]))
        query_codes = _topk_soft_assign(
            query_state["energy"] @ semantic_centers.T,
            local_top_k,
            local_temperature,
        )
        carrier = operator_method.score(doc_state, query_state)
        support_width = min(
            max(1, int(round(float(operator_method.params.get("support_width", 8))))),
            doc_count,
        )
        if support_width <= 0:
            return base_aux
        support_idx = np.argpartition(-carrier, support_width - 1, axis=1)[
            :, :support_width
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        support_temperature = max(
            1e-4, float(operator_method.params.get("support_temperature", 0.10))
        )
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_codes = semantic_codes[support_idx]
        support_context = np.sum(
            support_codes * support_weights[:, :, None],
            axis=1,
            dtype=np.float32,
        )
        support_context = _safe_normalize(support_context)
        support_alignment = np.sum(
            query_codes[:, None, :] * support_codes,
            axis=2,
            dtype=np.float32,
        )
        candidate_alignment = np.sum(
            support_context[:, None, :] * semantic_codes[None, :, :],
            axis=2,
            dtype=np.float32,
        )
        route_signal = np.sum(
            support_weights[:, :, None]
            * np.sqrt(np.clip(np.maximum(support_alignment, 0.0), 0.0, 1.0))[:, :, None]
            * np.sqrt(np.clip(np.maximum(candidate_alignment, 0.0), 0.0, 1.0))[
                :, None, :
            ],
            axis=1,
            dtype=np.float32,
        )
        query_peak = np.max(query_codes, axis=1, keepdims=True)
        family_alignment = np.sum(
            query_codes * support_context,
            axis=1,
            keepdims=True,
            dtype=np.float32,
        )
        code_focus_floor = float(prototype_method.params.get("code_focus_floor", 0.18))
        focus = np.sqrt(
            np.clip(query_peak * np.maximum(family_alignment, 0.0), 0.0, 1.0)
        ).astype(np.float32)
        focus_gate = np.clip(
            (focus - code_focus_floor) / max(1e-4, 1.0 - code_focus_floor),
            0.0,
            1.0,
        )
        # Prototype codes only route the existing operator instead of adding a second score.
        route_gate = 1.0 - route_mix * focus_gate * (
            1.0 - np.clip(route_signal, 0.0, 1.0)
        )
        route_gate = np.clip(route_gate, route_floor, 1.0).astype(np.float32)
        routed = base_aux * route_gate
        routed = routed - np.mean(routed, axis=1, keepdims=True)
        routed_scale = np.max(np.abs(routed), axis=1, keepdims=True)
        routed_scale = np.where(routed_scale == 0.0, 1.0, routed_scale)
        return routed / routed_scale

    return EmbeddingStateMethod(
        method_name="projective_observer_prototype_routing_observable_v0",
        family="state_projective_observer_prototype_routing_observable",
        params={
            **prototype_method.params,
            "route_mix": route_mix,
            "route_floor": route_floor,
        },
        encode_docs=prototype_method.encode_docs,
        encode_queries=prototype_method.encode_queries,
        score=prototype_method.score,
        aux_score=_aux_score,
    )


def _projective_observer_prototype_residual_routing_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    residual_mix = float(params.get("residual_mix", 0.08))
    local_params = dict(params)
    local_params.setdefault("codebook_builder", "keyed_prototypes")
    local_params.setdefault("prototype_refine_steps", 1)
    local_params.setdefault("code_profile_mix", 0.08)
    operator_method = _projective_observer_subgraph_transport_observable_build(
        rng, local_params
    )
    prototype_method = _projective_observer_prototype_support_observable_build(
        rng, local_params
    )
    routing_method = _projective_observer_prototype_routing_observable_build(
        rng, local_params
    )
    base_aux_score = operator_method.aux_score
    support_aux_score = prototype_method.aux_score
    routed_aux_score = routing_method.aux_score

    if base_aux_score is None or support_aux_score is None or routed_aux_score is None:
        return EmbeddingStateMethod(
            method_name="projective_observer_prototype_residual_routing_observable_v0",
            family="state_projective_observer_prototype_residual_routing_observable",
            params={
                **prototype_method.params,
                "route_mix": float(routing_method.params.get("route_mix", 0.45)),
                "route_floor": float(routing_method.params.get("route_floor", 0.35)),
                "residual_mix": residual_mix,
            },
            encode_docs=prototype_method.encode_docs,
            encode_queries=prototype_method.encode_queries,
            score=prototype_method.score,
        )

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        routed_aux = routed_aux_score(doc_state, query_state)
        base_aux = base_aux_score(doc_state, query_state)
        support_aux = support_aux_score(doc_state, query_state)
        if base_aux.shape != routed_aux.shape:
            return routed_aux
        if support_aux.shape != routed_aux.shape:
            return routed_aux
        residual = support_aux - base_aux
        residual = residual - np.mean(residual, axis=1, keepdims=True)
        routed_norm = np.sum(routed_aux * routed_aux, axis=1, keepdims=True)
        routed_norm = np.where(routed_norm <= 1e-6, 1.0, routed_norm)
        # Remove the routed component so the residual adds readability without simply cloning the routed operator.
        projection = np.sum(residual * routed_aux, axis=1, keepdims=True) / routed_norm
        residual = residual - projection * routed_aux
        residual_scale = np.max(np.abs(residual), axis=1, keepdims=True)
        residual_scale = np.where(residual_scale == 0.0, 1.0, residual_scale)
        residual = residual / residual_scale
        combined = routed_aux + residual_mix * residual
        combined = combined - np.mean(combined, axis=1, keepdims=True)
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return combined / combined_scale

    return EmbeddingStateMethod(
        method_name="projective_observer_prototype_residual_routing_observable_v0",
        family="state_projective_observer_prototype_residual_routing_observable",
        params={
            **prototype_method.params,
            "route_mix": float(routing_method.params.get("route_mix", 0.45)),
            "route_floor": float(routing_method.params.get("route_floor", 0.35)),
            "residual_mix": residual_mix,
        },
        encode_docs=prototype_method.encode_docs,
        encode_queries=prototype_method.encode_queries,
        score=prototype_method.score,
        aux_score=_aux_score,
    )


def _projective_semantic_codebook_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_hilbert_embedding_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    support_width = max(4, int(params.get("support_width", 10)))
    rerank_width = max(support_width, int(params.get("rerank_width", 24)))
    support_temperature = float(params.get("support_temperature", 0.14))
    codebook_size = max(8, int(params.get("codebook_size", 32)))
    code_top_k = min(codebook_size, max(1, int(params.get("code_top_k", 3))))
    code_temperature = float(params.get("code_temperature", 0.10))
    code_gain = float(params.get("code_gain", 0.012))
    uncertainty_width = float(params.get("uncertainty_width", 0.03))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    head_seed = _method_seed(
        "projective_semantic_codebook_head_v0", secret_key, dim
    ) % (2**32 - 1)
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode_projective(
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        energy = (wave_real**2 + wave_imag**2).astype(np.float32)
        public = _safe_normalize(energy @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return (
            public,
            wave_real.astype(np.float32),
            wave_imag.astype(np.float32),
            energy,
        )

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _semantic_codes(energy: np.ndarray, centers: np.ndarray) -> np.ndarray:
        if centers.shape[0] == 0:
            return np.zeros((energy.shape[0], 0), dtype=np.float32)
        local_top_k = min(code_top_k, centers.shape[0])
        logits = energy @ centers.T
        return _topk_soft_assign(logits, local_top_k, code_temperature)

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, energy = _encode_projective(x)
        doc_count = energy.shape[0]
        if doc_count == 0:
            empty_codes = np.zeros((0, 0), dtype=np.float32)
            empty_centers = np.zeros((0, hidden_dim), dtype=np.float32)
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "energy": energy,
                "semantic_codes": empty_codes,
                "semantic_centers": empty_centers,
            }
        effective_codebook_size = min(codebook_size, doc_count)
        # Die diskreten Codes kommen aus echten Carrier-Zuständen statt aus zufälligen Basen.
        kmeans = KMeans(
            n_clusters=effective_codebook_size,
            n_init=6,
            random_state=head_seed,
            max_iter=150,
        )
        kmeans.fit(energy)
        semantic_centers = _safe_normalize(
            np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        )
        semantic_codes = _semantic_codes(energy, semantic_centers)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "energy": energy,
            "semantic_codes": semantic_codes.astype(np.float32),
            "semantic_centers": semantic_centers.astype(np.float32),
            "semantic_code_config": np.asarray(
                [float(code_top_k), code_temperature], dtype=np.float32
            ),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, energy = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "energy": energy,
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        semantic_codes = doc_state["semantic_codes"]
        if semantic_codes.shape[0] != doc_count:
            return carrier
        semantic_centers = doc_state["semantic_centers"]
        if semantic_centers.shape[0] == 0:
            return carrier
        query_codes = _semantic_codes(query_state["energy"], semantic_centers)
        code_score = query_codes @ semantic_codes.T
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_codes = semantic_codes[support_idx]
        support_context = np.sum(
            support_codes * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        support_context = _safe_normalize(support_context)
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_codes = semantic_codes[candidate_idx]
        candidate_code = np.take_along_axis(code_score, candidate_idx, axis=1)
        context_code = np.sum(
            candidate_codes * support_context[:, None, :], axis=2, dtype=np.float32
        )
        candidate_signal = np.sqrt(np.maximum(candidate_code * context_code, 0.0))
        candidate_signal = candidate_signal - np.mean(
            candidate_signal, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_signal = candidate_signal / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        rerank_gain = np.maximum(
            0.0, 1.0 + code_gain * uncertainty_gate * candidate_signal
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_semantic_codebook_head_v0",
        family="state_projective_semantic_codebook",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "codebook_size": float(codebook_size),
            "code_top_k": float(code_top_k),
            "code_temperature": code_temperature,
            "code_gain": code_gain,
            "uncertainty_width": uncertainty_width,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_relational_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    relation_slots = max(4, int(params.get("relation_slots", 12)))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 3))))
    relation_temperature = float(params.get("relation_temperature", 0.24))
    relation_gain = float(params.get("relation_gain", 0.08))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_relational_head_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    relation_basis = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        energy = wave_real**2 + wave_imag**2
        relation_logits = energy @ relation_basis.T
        relation_head = _topk_soft_assign(
            relation_logits, relation_top_k, relation_temperature
        )
        public = _safe_normalize(energy @ public_mix)
        public = _mask_public_observation(
            "projective_relational_head_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public,
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "relation_head": relation_head.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        overlap = overlap + query_state["wave_imag"] @ doc_state["wave_imag"].T
        carrier = overlap**2
        relation_score = query_state["relation_head"] @ doc_state["relation_head"].T
        return carrier * (1.0 + relation_gain * relation_score)

    return EmbeddingStateMethod(
        method_name="projective_relational_head_v0",
        family="state_projective_relational",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "relation_gain": relation_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _projective_anchor_profile_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    anchor_count = max(8, int(params.get("anchor_count", 24)))
    anchor_top_k = max(1, int(params.get("anchor_top_k", 4)))
    anchor_temperature = float(params.get("anchor_temperature", 0.18))
    anchor_gain = float(params.get("anchor_gain", 0.06))
    rerank_width = max(8, int(params.get("rerank_width", 64)))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_anchor_profile_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            "projective_anchor_profile_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        if wave_real.shape[0] == 0:
            empty = np.zeros((0, hidden_dim), dtype=np.float32)
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "anchor_real": empty,
                "anchor_imag": empty,
                "anchor_profile": np.zeros((0, 0), dtype=np.float32),
            }
        effective_anchor_count = min(anchor_count, wave_real.shape[0])
        effective_anchor_top_k = min(anchor_top_k, effective_anchor_count)
        anchor_rng = np.random.default_rng(
            _method_seed(
                "projective_anchor_profile_v0_anchor_bank",
                secret_key,
                dim,
                offset=wave_real.shape[0],
            )
        )
        anchor_idx = np.sort(
            anchor_rng.choice(
                wave_real.shape[0], size=effective_anchor_count, replace=False
            )
        )
        anchor_real = wave_real[anchor_idx]
        anchor_imag = wave_imag[anchor_idx]
        anchor_overlap = wave_real @ anchor_real.T
        anchor_overlap = anchor_overlap + wave_imag @ anchor_imag.T
        anchor_profile = _topk_soft_assign(
            anchor_overlap**2,
            effective_anchor_top_k,
            anchor_temperature,
        )
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "anchor_real": anchor_real.astype(np.float32),
            "anchor_imag": anchor_imag.astype(np.float32),
            "anchor_profile": anchor_profile.astype(np.float32),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        overlap = overlap + query_state["wave_imag"] @ doc_state["wave_imag"].T
        carrier = overlap**2
        if doc_state["anchor_real"].shape[0] == 0:
            return carrier
        effective_anchor_top_k = min(anchor_top_k, doc_state["anchor_real"].shape[0])
        query_anchor_overlap = query_state["wave_real"] @ doc_state["anchor_real"].T
        query_anchor_overlap = query_anchor_overlap + (
            query_state["wave_imag"] @ doc_state["anchor_imag"].T
        )
        query_anchor_profile = _topk_soft_assign(
            query_anchor_overlap**2,
            effective_anchor_top_k,
            anchor_temperature,
        )
        anchor_score = query_anchor_profile @ doc_state["anchor_profile"].T
        keep = min(rerank_width, carrier.shape[1])
        if keep <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep - 1, axis=1)[:, :keep]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_anchor = np.take_along_axis(anchor_score, candidate_idx, axis=1)
        candidate_anchor = candidate_anchor - np.mean(
            candidate_anchor, axis=1, keepdims=True
        )
        reranked = candidate_carrier * (1.0 + anchor_gain * candidate_anchor)
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_anchor_profile_v0",
        family="state_projective_anchor_profile",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "anchor_count": float(anchor_count),
            "anchor_top_k": float(anchor_top_k),
            "anchor_temperature": anchor_temperature,
            "anchor_gain": anchor_gain,
            "rerank_width": float(rerank_width),
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_bipartite_profile_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    relation_slots = max(8, int(params.get("relation_slots", 24)))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 4))))
    relation_temperature = float(params.get("relation_temperature", 0.18))
    rerank_width = max(8, int(params.get("rerank_width", 64)))
    context_temperature = float(params.get("context_temperature", 0.08))
    context_mix = float(params.get("context_mix", 0.5))
    bipartite_gain = float(params.get("bipartite_gain", 0.10))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_bipartite_profile_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    relation_basis = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        energy = wave_real**2 + wave_imag**2
        relation_logits = energy @ relation_basis.T
        relation_profile = _topk_soft_assign(
            relation_logits, relation_top_k, relation_temperature
        )
        public = _safe_normalize(energy @ public_mix)
        public = _mask_public_observation(
            "projective_bipartite_profile_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public,
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "relation_profile": relation_profile.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = query_state["wave_real"] @ doc_state["wave_real"].T
        overlap = overlap + query_state["wave_imag"] @ doc_state["wave_imag"].T
        carrier = overlap**2
        if doc_state["relation_profile"].shape[1] == 0:
            return carrier
        keep = min(rerank_width, carrier.shape[1])
        if keep <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep - 1, axis=1)[:, :keep]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_profiles = doc_state["relation_profile"][candidate_idx]
        shifted = candidate_carrier - np.max(candidate_carrier, axis=1, keepdims=True)
        carrier_weights = np.exp(shifted / max(1e-4, context_temperature)).astype(
            np.float32
        )
        carrier_weights_sum = np.sum(carrier_weights, axis=1, keepdims=True)
        carrier_weights_sum = np.where(
            carrier_weights_sum == 0.0, 1.0, carrier_weights_sum
        )
        carrier_weights = carrier_weights / carrier_weights_sum
        context_profile = np.sum(
            candidate_profiles * carrier_weights[:, :, None], axis=1, dtype=np.float32
        )
        context_profile = _safe_normalize(context_profile)
        direct_signal = np.sum(
            candidate_profiles * query_state["relation_profile"][:, None, :], axis=2
        )
        context_signal = np.sum(
            candidate_profiles * context_profile[:, None, :], axis=2
        )
        local_signal = (
            context_mix * direct_signal + (1.0 - context_mix) * context_signal
        )
        local_signal = local_signal - np.mean(local_signal, axis=1, keepdims=True)
        rerank_gain = np.maximum(0.0, 1.0 + bipartite_gain * local_signal)
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_bipartite_profile_v0",
        family="state_projective_bipartite_profile",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "rerank_width": float(rerank_width),
            "context_temperature": context_temperature,
            "context_mix": context_mix,
            "bipartite_gain": bipartite_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _projective_graph_support_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    support_width = max(graph_k, int(params.get("support_width", 8)))
    rerank_width = max(support_width, int(params.get("rerank_width", 12)))
    support_temperature = float(params.get("support_temperature", 0.14))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    graph_gain = float(params.get("graph_gain", 0.012))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_graph_support_head_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            "projective_graph_support_head_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        doc_count = wave_real.shape[0]
        if doc_count <= 1:
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "support_graph": np.zeros((doc_count, doc_count), dtype=np.float32),
            }
        graph_scores = _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
        np.fill_diagonal(graph_scores, -np.inf)
        keep = min(graph_k, doc_count - 1)
        neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "support_graph": support_graph,
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        support_graph = doc_state["support_graph"]
        if support_graph.shape[0] != doc_count:
            return carrier
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_vector = np.zeros_like(carrier, dtype=np.float32)
        np.put_along_axis(support_vector, support_idx, support_weights, axis=1)
        incoming_signal = support_vector @ support_graph
        outgoing_signal = support_vector @ support_graph.T
        graph_signal = support_vector + 0.5 * (incoming_signal + outgoing_signal)
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_graph = np.take_along_axis(graph_signal, candidate_idx, axis=1)
        # Der Zusatzscore belohnt Kandidaten, die im selben keyed Teilgraphen wie das Query-Support-Set liegen.
        candidate_graph = candidate_graph - np.mean(
            candidate_graph, axis=1, keepdims=True
        )
        rerank_gain = np.maximum(0.0, 1.0 + graph_gain * candidate_graph)
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_graph_support_head_v0",
        family="state_projective_graph_support",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "graph_temperature": graph_temperature,
            "graph_gain": graph_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_consensus_graph_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    support_width = max(graph_k, int(params.get("support_width", 10)))
    rerank_width = max(support_width, int(params.get("rerank_width", 24)))
    support_temperature = float(params.get("support_temperature", 0.14))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    graph_gain = float(params.get("graph_gain", 0.004))
    hidden_view_ratio = float(params.get("hidden_view_ratio", 0.25))
    hidden_view_dim = max(8, int(hidden_dim * hidden_view_ratio))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_consensus_graph_head_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    hidden_view_mix = local_rng.normal(size=(hidden_dim, hidden_view_dim)).astype(
        np.float32
    )

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            "projective_consensus_graph_head_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _build_sparse_graph(graph_scores: np.ndarray) -> np.ndarray:
        doc_count = graph_scores.shape[0]
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        if doc_count <= 1:
            return support_graph
        keep = min(graph_k, doc_count - 1)
        if keep <= 0:
            return support_graph
        local_scores = np.array(graph_scores, copy=True)
        np.fill_diagonal(local_scores, -np.inf)
        neighbor_idx = np.argpartition(-local_scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(local_scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
        return support_graph

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        doc_count = wave_real.shape[0]
        if doc_count <= 1:
            empty_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "consensus_graph": empty_graph,
            }
        carrier_graph = _build_sparse_graph(
            _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
        )
        hidden_view = _safe_normalize((wave_real**2 + wave_imag**2) @ hidden_view_mix)
        hidden_graph = _build_sparse_graph((hidden_view @ hidden_view.T) ** 2)
        consensus_graph = np.sqrt(np.maximum(carrier_graph * hidden_graph, 0.0))
        consensus_sum = np.sum(consensus_graph, axis=1, keepdims=True)
        consensus_sum = np.where(consensus_sum == 0.0, 1.0, consensus_sum)
        consensus_graph = consensus_graph / consensus_sum
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "consensus_graph": consensus_graph.astype(np.float32),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        consensus_graph = doc_state["consensus_graph"]
        if consensus_graph.shape[0] != doc_count:
            return carrier
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_vector = np.zeros_like(carrier, dtype=np.float32)
        np.put_along_axis(support_vector, support_idx, support_weights, axis=1)
        incoming_signal = support_vector @ consensus_graph
        outgoing_signal = support_vector @ consensus_graph.T
        graph_signal = 0.5 * (incoming_signal + outgoing_signal)
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_graph = np.take_along_axis(graph_signal, candidate_idx, axis=1)
        candidate_graph = candidate_graph - np.mean(
            candidate_graph, axis=1, keepdims=True
        )
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        # Nur unsichere Queries bekommen einen Konsens-Boost aus zwei keyed Views.
        rerank_gain = np.maximum(
            0.0, 1.0 + graph_gain * uncertainty_gate * candidate_graph
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_consensus_graph_head_v0",
        family="state_projective_consensus_graph",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "graph_temperature": graph_temperature,
            "graph_gain": graph_gain,
            "hidden_view_ratio": hidden_view_ratio,
            "uncertainty_width": uncertainty_width,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_margin_graph_support_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_graph_support_head_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    support_width = max(graph_k, int(params.get("support_width", 8)))
    rerank_width = max(support_width, int(params.get("rerank_width", 12)))
    support_temperature = float(params.get("support_temperature", 0.14))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    graph_gain = float(params.get("graph_gain", 0.012))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        doc_count = wave_real.shape[0]
        if doc_count <= 1:
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "support_graph": np.zeros((doc_count, doc_count), dtype=np.float32),
            }
        graph_scores = _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
        np.fill_diagonal(graph_scores, -np.inf)
        keep = min(graph_k, doc_count - 1)
        neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "support_graph": support_graph,
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        support_graph = doc_state["support_graph"]
        if support_graph.shape[0] != doc_count:
            return carrier
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_vector = np.zeros_like(carrier, dtype=np.float32)
        np.put_along_axis(support_vector, support_idx, support_weights, axis=1)
        incoming_signal = support_vector @ support_graph
        outgoing_signal = support_vector @ support_graph.T
        graph_signal = support_vector + 0.5 * (incoming_signal + outgoing_signal)
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_graph = np.take_along_axis(graph_signal, candidate_idx, axis=1)
        candidate_graph = candidate_graph - np.mean(
            candidate_graph, axis=1, keepdims=True
        )
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        # Graph-Support greift nur bei kleiner Carrier-Marge, damit sichere Rangfolgen stabil bleiben.
        rerank_gain = np.maximum(
            0.0, 1.0 + graph_gain * uncertainty_gate * candidate_graph
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_margin_graph_support_v0",
        family="state_projective_margin_graph_support",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "graph_temperature": graph_temperature,
            "graph_gain": graph_gain,
            "uncertainty_width": uncertainty_width,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_coherence_routed_graph_support_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_graph_support_head_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    support_width = max(graph_k, int(params.get("support_width", 8)))
    rerank_width = max(support_width, int(params.get("rerank_width", 12)))
    support_temperature = float(params.get("support_temperature", 0.14))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    graph_gain = float(params.get("graph_gain", 0.012))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    relation_slots = max(8, int(params.get("relation_slots", 16)))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 2))))
    relation_temperature = float(params.get("relation_temperature", 0.08))
    coherence_gate_mix = float(params.get("coherence_gate_mix", 0.85))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    relation_basis = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _relation_profile(wave_real: np.ndarray, wave_imag: np.ndarray) -> np.ndarray:
        energy = wave_real**2 + wave_imag**2
        relation_logits = energy @ relation_basis.T
        return _topk_soft_assign(relation_logits, relation_top_k, relation_temperature)

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag)
        doc_count = wave_real.shape[0]
        if doc_count <= 1:
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "relation_profile": relation_profile.astype(np.float32),
                "support_graph": np.zeros((doc_count, doc_count), dtype=np.float32),
            }
        graph_scores = _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
        np.fill_diagonal(graph_scores, -np.inf)
        keep = min(graph_k, doc_count - 1)
        neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "relation_profile": relation_profile.astype(np.float32),
            "support_graph": support_graph,
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "relation_profile": relation_profile.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        support_graph = doc_state["support_graph"]
        if support_graph.shape[0] != doc_count:
            return carrier
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_vector = np.zeros_like(carrier, dtype=np.float32)
        np.put_along_axis(support_vector, support_idx, support_weights, axis=1)
        incoming_signal = support_vector @ support_graph
        outgoing_signal = support_vector @ support_graph.T
        graph_signal = support_vector + 0.5 * (incoming_signal + outgoing_signal)
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_graph = np.take_along_axis(graph_signal, candidate_idx, axis=1)
        candidate_graph = candidate_graph - np.mean(
            candidate_graph, axis=1, keepdims=True
        )
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        coherence_gate = np.ones((carrier.shape[0], 1), dtype=np.float32)
        relation_profile = doc_state["relation_profile"]
        if relation_profile.shape[0] == doc_count:
            support_profiles = relation_profile[support_idx]
            support_context = np.sum(
                support_profiles * support_weights[:, :, None], axis=1, dtype=np.float32
            )
            query_profile = query_state["relation_profile"]
            query_alignment = np.sum(
                query_profile * support_context, axis=1, keepdims=True, dtype=np.float32
            )
            support_focus = np.sum(
                support_context * support_context,
                axis=1,
                keepdims=True,
                dtype=np.float32,
            )
            coherence = np.sqrt(
                np.clip(query_alignment * support_focus, 0.0, 1.0)
            ).astype(np.float32)
            coherence_gate = np.clip(
                1.0 + coherence_gate_mix * (coherence - 0.5),
                0.25,
                1.75,
            ).astype(np.float32)
        # Der Graph-Rerank bleibt identisch zum Margin-Head und wird nur für kohärente Query-Familien stärker freigeschaltet.
        rerank_gain = np.maximum(
            0.0,
            1.0 + graph_gain * uncertainty_gate * coherence_gate * candidate_graph,
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_coherence_routed_graph_support_v0",
        family="state_projective_coherence_routed_graph_support",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "graph_temperature": graph_temperature,
            "graph_gain": graph_gain,
            "uncertainty_width": uncertainty_width,
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "coherence_gate_mix": coherence_gate_mix,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_semantic_prior_graph_support_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_graph_support_head_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    support_width = max(graph_k, int(params.get("support_width", 8)))
    rerank_width = max(support_width, int(params.get("rerank_width", 12)))
    support_temperature = float(params.get("support_temperature", 0.14))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    graph_gain = float(params.get("graph_gain", 0.012))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    codebook_size = max(8, int(params.get("codebook_size", 24)))
    code_top_k = min(codebook_size, max(1, int(params.get("code_top_k", 3))))
    code_temperature = float(params.get("code_temperature", 0.10))
    semantic_mix = float(params.get("semantic_mix", 0.22))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    head_seed = _method_seed(
        "projective_semantic_prior_graph_support_v0", secret_key, dim
    ) % (2**32 - 1)
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode_projective(
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        energy = (wave_real**2 + wave_imag**2).astype(np.float32)
        public = _safe_normalize(energy @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return (
            public,
            wave_real.astype(np.float32),
            wave_imag.astype(np.float32),
            energy,
        )

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _semantic_codes(energy: np.ndarray, centers: np.ndarray) -> np.ndarray:
        if centers.shape[0] == 0:
            return np.zeros((energy.shape[0], 0), dtype=np.float32)
        local_top_k = min(code_top_k, centers.shape[0])
        logits = energy @ centers.T
        return _topk_soft_assign(logits, local_top_k, code_temperature)

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, energy = _encode_projective(x)
        doc_count = wave_real.shape[0]
        empty_codes = np.zeros((doc_count, 0), dtype=np.float32)
        empty_centers = np.zeros((0, hidden_dim), dtype=np.float32)
        if doc_count <= 1:
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "energy": energy,
                "support_graph": np.zeros((doc_count, doc_count), dtype=np.float32),
                "semantic_codes": empty_codes,
                "semantic_centers": empty_centers,
            }
        graph_scores = _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
        np.fill_diagonal(graph_scores, -np.inf)
        keep = min(graph_k, doc_count - 1)
        neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
        effective_codebook_size = min(codebook_size, doc_count)
        kmeans = KMeans(
            n_clusters=effective_codebook_size,
            n_init=6,
            random_state=head_seed,
            max_iter=150,
        )
        kmeans.fit(energy)
        semantic_centers = _safe_normalize(
            np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        )
        semantic_codes = _semantic_codes(energy, semantic_centers)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "energy": energy,
            "support_graph": support_graph,
            "semantic_codes": semantic_codes.astype(np.float32),
            "semantic_centers": semantic_centers.astype(np.float32),
            "semantic_code_config": np.asarray(
                [float(code_top_k), code_temperature], dtype=np.float32
            ),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, energy = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "energy": energy,
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        support_graph = doc_state["support_graph"]
        if support_graph.shape[0] != doc_count:
            return carrier
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_vector = np.zeros_like(carrier, dtype=np.float32)
        np.put_along_axis(support_vector, support_idx, support_weights, axis=1)
        incoming_signal = support_vector @ support_graph
        outgoing_signal = support_vector @ support_graph.T
        graph_signal = support_vector + 0.5 * (incoming_signal + outgoing_signal)
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_graph = np.take_along_axis(graph_signal, candidate_idx, axis=1)
        candidate_graph = candidate_graph - np.mean(
            candidate_graph, axis=1, keepdims=True
        )
        semantic_codes = doc_state["semantic_codes"]
        semantic_centers = doc_state["semantic_centers"]
        if semantic_codes.shape[0] == doc_count and semantic_centers.shape[0] > 0:
            query_codes = _semantic_codes(query_state["energy"], semantic_centers)
            support_codes = semantic_codes[support_idx]
            support_context = np.sum(
                support_codes * support_weights[:, :, None], axis=1, dtype=np.float32
            )
            support_context = _safe_normalize(support_context)
            candidate_codes = semantic_codes[candidate_idx]
            candidate_semantic = np.sum(
                candidate_codes * support_context[:, None, :], axis=2, dtype=np.float32
            )
            candidate_semantic = candidate_semantic - np.mean(
                candidate_semantic, axis=1, keepdims=True
            )
            semantic_scale = np.max(np.abs(candidate_semantic), axis=1, keepdims=True)
            semantic_scale = np.where(semantic_scale == 0.0, 1.0, semantic_scale)
            candidate_semantic = candidate_semantic / semantic_scale
            query_confidence = np.max(query_codes, axis=1, keepdims=True)
            family_alignment = np.sum(
                query_codes * support_context, axis=1, keepdims=True, dtype=np.float32
            )
            semantic_gate = np.sqrt(
                np.clip(query_confidence * np.maximum(family_alignment, 0.0), 0.0, 1.0)
            ).astype(np.float32)
            # Semantische Codes liefern nur einen schwachen Familien-Prior, keinen zweiten Hauptscore.
            candidate_graph = (
                candidate_graph + semantic_mix * semantic_gate * candidate_semantic
            )
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        rerank_gain = np.maximum(
            0.0, 1.0 + graph_gain * uncertainty_gate * candidate_graph
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_semantic_prior_graph_support_v0",
        family="state_projective_semantic_prior_graph_support",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "graph_temperature": graph_temperature,
            "graph_gain": graph_gain,
            "uncertainty_width": uncertainty_width,
            "codebook_size": float(codebook_size),
            "code_top_k": float(code_top_k),
            "code_temperature": code_temperature,
            "semantic_mix": semantic_mix,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_semantic_listdecode_graph_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_graph_support_head_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    support_width = max(graph_k, int(params.get("support_width", 10)))
    rerank_width = max(support_width, int(params.get("rerank_width", 24)))
    support_temperature = float(params.get("support_temperature", 0.14))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    graph_gain = float(params.get("graph_gain", 0.012))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    codebook_size = max(8, int(params.get("codebook_size", 12)))
    code_top_k = min(codebook_size, max(1, int(params.get("code_top_k", 2))))
    code_temperature = float(params.get("code_temperature", 0.10))
    mode_count = max(1, int(params.get("mode_count", 2)))
    mode_floor = float(params.get("mode_floor", 0.15))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    head_seed = _method_seed(
        "projective_semantic_listdecode_graph_v0", secret_key, dim
    ) % (2**32 - 1)
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode_projective(
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        energy = (wave_real**2 + wave_imag**2).astype(np.float32)
        public = _safe_normalize(energy @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return (
            public,
            wave_real.astype(np.float32),
            wave_imag.astype(np.float32),
            energy,
        )

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _semantic_codes(energy: np.ndarray, centers: np.ndarray) -> np.ndarray:
        if centers.shape[0] == 0:
            return np.zeros((energy.shape[0], 0), dtype=np.float32)
        local_top_k = min(code_top_k, centers.shape[0])
        logits = energy @ centers.T
        return _topk_soft_assign(logits, local_top_k, code_temperature)

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, energy = _encode_projective(x)
        doc_count = wave_real.shape[0]
        empty_codes = np.zeros((doc_count, 0), dtype=np.float32)
        empty_centers = np.zeros((0, hidden_dim), dtype=np.float32)
        if doc_count <= 1:
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "energy": energy,
                "support_graph": np.zeros((doc_count, doc_count), dtype=np.float32),
                "semantic_codes": empty_codes,
                "semantic_centers": empty_centers,
            }
        graph_scores = _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
        np.fill_diagonal(graph_scores, -np.inf)
        keep = min(graph_k, doc_count - 1)
        neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
        effective_codebook_size = min(codebook_size, doc_count)
        kmeans = KMeans(
            n_clusters=effective_codebook_size,
            n_init=6,
            random_state=head_seed,
            max_iter=150,
        )
        kmeans.fit(energy)
        semantic_centers = _safe_normalize(
            np.asarray(kmeans.cluster_centers_, dtype=np.float32)
        )
        semantic_codes = _semantic_codes(energy, semantic_centers)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "energy": energy,
            "support_graph": support_graph,
            "semantic_codes": semantic_codes.astype(np.float32),
            "semantic_centers": semantic_centers.astype(np.float32),
            "semantic_code_config": np.asarray(
                [float(code_top_k), code_temperature], dtype=np.float32
            ),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, energy = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "energy": energy,
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        support_graph = doc_state["support_graph"]
        if support_graph.shape[0] != doc_count:
            return carrier
        semantic_codes = doc_state["semantic_codes"]
        semantic_centers = doc_state["semantic_centers"]
        if semantic_codes.shape[0] != doc_count:
            return carrier
        if semantic_centers.shape[0] == 0:
            return carrier
        query_codes = _semantic_codes(query_state["energy"], semantic_centers)
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_codes = semantic_codes[support_idx]
        support_context = np.sum(
            support_codes * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        mode_seed = np.sqrt(np.maximum(query_codes * support_context, 0.0)).astype(
            np.float32
        )
        effective_mode_count = min(mode_count, mode_seed.shape[1])
        if effective_mode_count <= 0:
            return carrier
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_codes = semantic_codes[candidate_idx]
        mode_idx = np.argpartition(-mode_seed, effective_mode_count - 1, axis=1)[
            :, :effective_mode_count
        ]
        candidate_signal = np.zeros_like(candidate_carrier, dtype=np.float32)
        for mode_offset in range(effective_mode_count):
            selected_mode = mode_idx[:, mode_offset : mode_offset + 1]
            support_mode_idx = np.broadcast_to(
                selected_mode[:, None, :],
                (support_codes.shape[0], support_codes.shape[1], 1),
            )
            candidate_mode_idx = np.broadcast_to(
                selected_mode[:, None, :],
                (candidate_codes.shape[0], candidate_codes.shape[1], 1),
            )
            support_mode = np.take_along_axis(support_codes, support_mode_idx, axis=2)[
                :, :, 0
            ]
            candidate_mode = np.take_along_axis(
                candidate_codes, candidate_mode_idx, axis=2
            )[:, :, 0]
            query_mode = np.take_along_axis(mode_seed, selected_mode, axis=1)
            mode_support_weights = support_weights * (
                mode_floor + (1.0 - mode_floor) * support_mode
            )
            mode_weight_sum = np.sum(mode_support_weights, axis=1, keepdims=True)
            mode_weight_sum = np.where(mode_weight_sum == 0.0, 1.0, mode_weight_sum)
            mode_support_weights = mode_support_weights / mode_weight_sum
            mode_support_vector = np.zeros_like(carrier, dtype=np.float32)
            np.put_along_axis(
                mode_support_vector, support_idx, mode_support_weights, axis=1
            )
            incoming_signal = mode_support_vector @ support_graph
            outgoing_signal = mode_support_vector @ support_graph.T
            mode_graph_signal = mode_support_vector + 0.5 * (
                incoming_signal + outgoing_signal
            )
            candidate_graph = np.take_along_axis(
                mode_graph_signal, candidate_idx, axis=1
            )
            mode_alignment = np.sqrt(np.maximum(query_mode * candidate_mode, 0.0))
            mode_signal = candidate_graph * np.maximum(mode_floor, mode_alignment)
            candidate_signal = np.maximum(
                candidate_signal, mode_signal.astype(np.float32)
            )
        # Statt einen einzigen Support-Kontext zu mitteln, behalten wir mehrere plausible Modi bis zum letzten Rerank-Schritt getrennt.
        candidate_signal = candidate_signal - np.mean(
            candidate_signal, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_signal = candidate_signal / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        rerank_gain = np.maximum(
            0.0, 1.0 + graph_gain * uncertainty_gate * candidate_signal
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_semantic_listdecode_graph_v0",
        family="state_projective_semantic_listdecode_graph",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "graph_temperature": graph_temperature,
            "graph_gain": graph_gain,
            "uncertainty_width": uncertainty_width,
            "codebook_size": float(codebook_size),
            "code_top_k": float(code_top_k),
            "code_temperature": code_temperature,
            "mode_count": float(mode_count),
            "mode_floor": mode_floor,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_observer_listdecode_graph_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_graph_support_head_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    support_width = max(graph_k, int(params.get("support_width", 10)))
    rerank_width = max(support_width, int(params.get("rerank_width", 24)))
    support_temperature = float(params.get("support_temperature", 0.14))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    graph_gain = float(params.get("graph_gain", 0.002))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    observer_channels = max(2, int(params.get("observer_channels", 2)))
    observer_dim = max(12, int(params.get("observer_dim", 24)))
    observer_gain = float(params.get("observer_gain", 0.15))
    mode_count = max(1, int(params.get("mode_count", 2)))
    mode_floor = float(params.get("mode_floor", 0.5))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    carrier_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    observer_rng = np.random.default_rng(
        _method_seed("projective_observer_listdecode_graph_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    phase_bias = carrier_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = carrier_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    observer_proj = np.stack(
        [
            _qr_orthogonal(observer_rng, dim, observer_dim)
            for _ in range(observer_channels)
        ]
    )
    observer_mod_proj = np.stack(
        [
            _qr_orthogonal(observer_rng, dim, observer_dim)
            for _ in range(observer_channels)
        ]
    )
    observer_bias = observer_rng.uniform(
        -math.pi, math.pi, size=(observer_channels, observer_dim)
    ).astype(np.float32)
    observer_gate = np.abs(
        observer_rng.normal(size=(observer_channels,)).astype(np.float32)
    )
    observer_gate = observer_gate / np.maximum(1e-6, np.sum(observer_gate))

    def _encode_all(
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        observer_hidden = []
        for channel in range(observer_channels):
            base = y @ observer_proj[channel]
            mod = np.sin(y @ observer_mod_proj[channel] + observer_bias[channel])
            hidden = _safe_normalize(base + observer_gain * mod)
            observer_hidden.append(hidden.astype(np.float32))
        return (
            public,
            wave_real.astype(np.float32),
            wave_imag.astype(np.float32),
            np.stack(observer_hidden, axis=1),
        )

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, observer_hidden = _encode_all(x)
        doc_count = wave_real.shape[0]
        if doc_count <= 1:
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "support_graph": np.zeros((doc_count, doc_count), dtype=np.float32),
                "observer_channels": observer_hidden.astype(np.float32),
                "observer_gate": observer_gate[None, :].astype(np.float32),
            }
        graph_scores = _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
        np.fill_diagonal(graph_scores, -np.inf)
        keep = min(graph_k, doc_count - 1)
        neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "support_graph": support_graph,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, observer_hidden = _encode_all(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        support_graph = doc_state["support_graph"]
        if support_graph.shape[0] != doc_count:
            return carrier
        doc_observer = doc_state["observer_channels"]
        if doc_observer.shape[0] != doc_count:
            return carrier
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        query_observer = query_state["observer_channels"]
        support_observer = doc_observer[support_idx]
        candidate_observer = doc_observer[candidate_idx]
        support_response = np.maximum(
            np.sum(query_observer[:, None, :, :] * support_observer, axis=3), 0.0
        )
        candidate_response = np.maximum(
            np.sum(query_observer[:, None, :, :] * candidate_observer, axis=3), 0.0
        )
        mode_seed = np.sum(
            support_response * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        mode_seed = mode_seed * observer_gate[None, :]
        effective_mode_count = min(mode_count, mode_seed.shape[1])
        if effective_mode_count <= 0:
            return carrier
        mode_idx = np.argpartition(-mode_seed, effective_mode_count - 1, axis=1)[
            :, :effective_mode_count
        ]
        candidate_signal = np.zeros_like(candidate_carrier, dtype=np.float32)
        for mode_offset in range(effective_mode_count):
            selected_mode = mode_idx[:, mode_offset : mode_offset + 1]
            support_mode = np.take_along_axis(
                support_response, selected_mode[:, None, :], axis=2
            )[:, :, 0]
            candidate_mode = np.take_along_axis(
                candidate_response, selected_mode[:, None, :], axis=2
            )[:, :, 0]
            query_mode = np.take_along_axis(mode_seed, selected_mode, axis=1)
            mode_support_weights = support_weights * (
                mode_floor + (1.0 - mode_floor) * support_mode
            )
            mode_weight_sum = np.sum(mode_support_weights, axis=1, keepdims=True)
            mode_weight_sum = np.where(mode_weight_sum == 0.0, 1.0, mode_weight_sum)
            mode_support_weights = mode_support_weights / mode_weight_sum
            mode_support_vector = np.zeros_like(carrier, dtype=np.float32)
            np.put_along_axis(
                mode_support_vector, support_idx, mode_support_weights, axis=1
            )
            incoming_signal = mode_support_vector @ support_graph
            outgoing_signal = mode_support_vector @ support_graph.T
            mode_graph_signal = mode_support_vector + 0.5 * (
                incoming_signal + outgoing_signal
            )
            candidate_graph = np.take_along_axis(
                mode_graph_signal, candidate_idx, axis=1
            )
            mode_alignment = np.sqrt(np.maximum(query_mode * candidate_mode, 0.0))
            mode_signal = candidate_graph * np.maximum(mode_floor, mode_alignment)
            candidate_signal = np.maximum(
                candidate_signal, mode_signal.astype(np.float32)
            )
        # Observer-Kanäle trennen nur plausible Modi; der stabile Carrier-/Graph-Pfad bleibt die eigentliche Evidenzquelle.
        candidate_signal = candidate_signal - np.mean(
            candidate_signal, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_signal = candidate_signal / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        rerank_gain = np.maximum(
            0.0, 1.0 + graph_gain * uncertainty_gate * candidate_signal
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_observer_listdecode_graph_v0",
        family="state_projective_observer_listdecode_graph",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "graph_temperature": graph_temperature,
            "graph_gain": graph_gain,
            "uncertainty_width": uncertainty_width,
            "observer_channels": float(observer_channels),
            "observer_dim": float(observer_dim),
            "observer_gain": observer_gain,
            "mode_count": float(mode_count),
            "mode_floor": mode_floor,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_query_gated_graph_support_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_graph_support_head_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    graph_k = max(4, int(params.get("graph_k", 8)))
    support_width = max(graph_k, int(params.get("support_width", 8)))
    rerank_width = max(support_width, int(params.get("rerank_width", 12)))
    support_temperature = float(params.get("support_temperature", 0.14))
    graph_temperature = float(params.get("graph_temperature", 0.24))
    graph_gain = float(params.get("graph_gain", 0.012))
    relation_slots = max(8, int(params.get("relation_slots", 24)))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 4))))
    relation_temperature = float(params.get("relation_temperature", 0.18))
    support_relation_mix = float(params.get("support_relation_mix", 0.35))
    candidate_relation_mix = float(params.get("candidate_relation_mix", 0.50))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    relation_basis = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _relation_profile(wave_real: np.ndarray, wave_imag: np.ndarray) -> np.ndarray:
        energy = wave_real**2 + wave_imag**2
        relation_logits = energy @ relation_basis.T
        return _topk_soft_assign(relation_logits, relation_top_k, relation_temperature)

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag)
        doc_count = wave_real.shape[0]
        if doc_count <= 1:
            return {
                "public": public,
                "wave_real": wave_real,
                "wave_imag": wave_imag,
                "relation_profile": relation_profile.astype(np.float32),
                "support_graph": np.zeros((doc_count, doc_count), dtype=np.float32),
            }
        graph_scores = _carrier_score(wave_real, wave_imag, wave_real, wave_imag)
        np.fill_diagonal(graph_scores, -np.inf)
        keep = min(graph_k, doc_count - 1)
        neighbor_idx = np.argpartition(-graph_scores, keep - 1, axis=1)[:, :keep]
        neighbor_scores = np.take_along_axis(graph_scores, neighbor_idx, axis=1)
        shifted = neighbor_scores - np.max(neighbor_scores, axis=1, keepdims=True)
        neighbor_weights = np.exp(shifted / max(1e-4, graph_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(neighbor_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        neighbor_weights = neighbor_weights / weight_sum
        support_graph = np.zeros((doc_count, doc_count), dtype=np.float32)
        np.put_along_axis(support_graph, neighbor_idx, neighbor_weights, axis=1)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "relation_profile": relation_profile.astype(np.float32),
            "support_graph": support_graph,
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "relation_profile": relation_profile.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        support_graph = doc_state["support_graph"]
        if support_graph.shape[0] != doc_count:
            return carrier
        relation_score = (
            query_state["relation_profile"] @ doc_state["relation_profile"].T
        )
        if relation_score.shape[1] != doc_count:
            return carrier
        relation_score = np.maximum(relation_score, 0.0).astype(np.float32)
        relation_scale = relation_score / np.maximum(
            1e-6, np.max(relation_score, axis=1, keepdims=True)
        )
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        support_relation = np.take_along_axis(relation_scale, support_idx, axis=1)
        support_weights = support_weights * (
            1.0 + support_relation_mix * support_relation
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_vector = np.zeros_like(carrier, dtype=np.float32)
        np.put_along_axis(support_vector, support_idx, support_weights, axis=1)
        incoming_signal = support_vector @ support_graph
        outgoing_signal = support_vector @ support_graph.T
        graph_signal = support_vector + 0.5 * (incoming_signal + outgoing_signal)
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_graph = np.take_along_axis(graph_signal, candidate_idx, axis=1)
        candidate_relation = np.take_along_axis(relation_scale, candidate_idx, axis=1)
        # Der Graph-Bias zählt nur dort stark, wo Query und Kandidat auch im keyed Relationenraum kompatibel sind.
        gated_graph = candidate_graph * (
            1.0 + candidate_relation_mix * candidate_relation
        )
        gated_graph = gated_graph - np.mean(gated_graph, axis=1, keepdims=True)
        rerank_gain = np.maximum(0.0, 1.0 + graph_gain * gated_graph)
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_query_gated_graph_support_v0",
        family="state_projective_query_gated_graph_support",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "graph_k": float(graph_k),
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "graph_temperature": graph_temperature,
            "graph_gain": graph_gain,
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "support_relation_mix": support_relation_mix,
            "candidate_relation_mix": candidate_relation_mix,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_query_edge_graph_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_graph_support_head_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    support_width = max(4, int(params.get("support_width", 10)))
    rerank_width = max(support_width, int(params.get("rerank_width", 24)))
    support_temperature = float(params.get("support_temperature", 0.14))
    relation_slots = max(8, int(params.get("relation_slots", 24)))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 3))))
    relation_temperature = float(params.get("relation_temperature", 0.12))
    support_relation_mix = float(params.get("support_relation_mix", 0.60))
    candidate_relation_mix = float(params.get("candidate_relation_mix", 0.35))
    edge_gain = float(params.get("edge_gain", 0.10))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    relation_basis = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _relation_profile(wave_real: np.ndarray, wave_imag: np.ndarray) -> np.ndarray:
        energy = wave_real**2 + wave_imag**2
        relation_logits = energy @ relation_basis.T
        return _topk_soft_assign(relation_logits, relation_top_k, relation_temperature)

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "relation_profile": relation_profile.astype(np.float32),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        relation_profile = _relation_profile(wave_real, wave_imag)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "relation_profile": relation_profile.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        relation_score = (
            query_state["relation_profile"] @ doc_state["relation_profile"].T
        )
        if relation_score.shape[1] != doc_count:
            return carrier
        relation_score = np.maximum(relation_score, 0.0).astype(np.float32)
        relation_scale = relation_score / np.maximum(
            1e-6, np.max(relation_score, axis=1, keepdims=True)
        )
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_seed = carrier * (1.0 + support_relation_mix * relation_scale)
        support_idx = np.argpartition(-support_seed, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_seed_scores = np.take_along_axis(support_seed, support_idx, axis=1)
        shifted = support_seed_scores - np.max(
            support_seed_scores, axis=1, keepdims=True
        )
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_relation = np.take_along_axis(relation_scale, candidate_idx, axis=1)
        support_profiles = doc_state["relation_profile"][support_idx]
        candidate_profiles = doc_state["relation_profile"][candidate_idx]
        query_slots = query_state["relation_profile"][:, None, None, :]
        shared_slots = np.minimum(
            support_profiles[:, :, None, :], candidate_profiles[:, None, :, :]
        )
        query_edge = np.max(query_slots * shared_slots, axis=3)
        edge_signal = np.sum(
            query_edge * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        candidate_signal = edge_signal + candidate_relation_mix * candidate_relation
        candidate_signal = candidate_signal - np.mean(
            candidate_signal, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_signal = candidate_signal / signal_scale
        rerank_gain = np.maximum(0.0, 1.0 + edge_gain * candidate_signal)
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_query_edge_graph_v0",
        family="state_projective_query_edge_graph",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "support_relation_mix": support_relation_mix,
            "candidate_relation_mix": candidate_relation_mix,
            "edge_gain": edge_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_observer_cograph_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_graph_support_head_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    support_width = max(4, int(params.get("support_width", 10)))
    rerank_width = max(support_width, int(params.get("rerank_width", 24)))
    support_temperature = float(params.get("support_temperature", 0.14))
    observer_channels = max(2, int(params.get("observer_channels", 6)))
    observer_dim = max(12, int(params.get("observer_dim", max(12, dim // 2))))
    observer_gain = float(params.get("observer_gain", 0.55))
    observer_mix = float(params.get("observer_mix", 0.35))
    cograph_gain = float(params.get("cograph_gain", 0.018))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    carrier_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    observer_rng = np.random.default_rng(
        _method_seed("projective_observer_cograph_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(carrier_rng, dim, hidden_dim)
    phase_bias = carrier_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = carrier_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    observer_proj = np.stack(
        [
            _qr_orthogonal(observer_rng, dim, observer_dim)
            for _ in range(observer_channels)
        ]
    )
    observer_mod_proj = np.stack(
        [
            _qr_orthogonal(observer_rng, dim, observer_dim)
            for _ in range(observer_channels)
        ]
    )
    observer_bias = observer_rng.uniform(
        -math.pi, math.pi, size=(observer_channels, observer_dim)
    ).astype(np.float32)
    observer_gate = np.abs(
        observer_rng.normal(size=(observer_channels,)).astype(np.float32)
    )
    observer_gate = observer_gate / np.maximum(1e-6, np.sum(observer_gate))

    def _encode_all(
        x: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        observer_hidden = []
        for channel in range(observer_channels):
            base = y @ observer_proj[channel]
            mod = np.sin(y @ observer_mod_proj[channel] + observer_bias[channel])
            hidden = _safe_normalize(base + observer_gain * mod)
            observer_hidden.append(hidden.astype(np.float32))
        return (
            public,
            wave_real.astype(np.float32),
            wave_imag.astype(np.float32),
            np.stack(observer_hidden, axis=1),
        )

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, observer_hidden = _encode_all(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag, observer_hidden = _encode_all(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "observer_channels": observer_hidden.astype(np.float32),
            "observer_gate": observer_gate[None, :].astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        doc_observer = doc_state["observer_channels"]
        if doc_observer.shape[0] != doc_count:
            return carrier
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        query_observer = query_state["observer_channels"]
        support_observer = doc_observer[support_idx]
        candidate_observer = doc_observer[candidate_idx]
        support_response = np.maximum(
            np.sum(query_observer[:, None, :, :] * support_observer, axis=3), 0.0
        )
        candidate_response = np.maximum(
            np.sum(query_observer[:, None, :, :] * candidate_observer, axis=3), 0.0
        )
        channel_weight = np.sum(
            support_response * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        channel_weight = channel_weight * observer_gate[None, :]
        channel_total = np.sum(channel_weight, axis=1, keepdims=True)
        channel_total = np.where(channel_total == 0.0, 1.0, channel_total)
        channel_weight = channel_weight / channel_total
        shared_response = np.sqrt(
            np.maximum(
                support_response[:, :, None, :] * candidate_response[:, None, :, :], 0.0
            )
        )
        observer_edges = np.maximum(
            np.sum(
                support_observer[:, :, None, :, :]
                * candidate_observer[:, None, :, :, :],
                axis=4,
            ),
            0.0,
        )
        observer_edges = observer_edges * shared_response
        observer_edges = np.sum(
            observer_edges * channel_weight[:, None, None, :], axis=3, dtype=np.float32
        )
        edge_signal = np.sum(
            observer_edges * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        direct_signal = np.sum(
            candidate_response * channel_weight[:, None, :], axis=2, dtype=np.float32
        )
        # Ein unabhängiger Observer-Pfad soll query-konditionierte Co-Response-Evidenz liefern,
        # statt denselben Carrier-Graphen nur erneut umzuwichten.
        candidate_signal = edge_signal + observer_mix * direct_signal
        candidate_signal = candidate_signal - np.mean(
            candidate_signal, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_signal = candidate_signal / signal_scale
        rerank_gain = np.maximum(0.0, 1.0 + cograph_gain * candidate_signal)
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_observer_cograph_v0",
        family="state_projective_observer_cograph",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "observer_channels": float(observer_channels),
            "observer_dim": float(observer_dim),
            "observer_gain": observer_gain,
            "observer_mix": observer_mix,
            "cograph_gain": cograph_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_multiquery_cograph_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_hilbert_embedding_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    support_width = max(4, int(params.get("support_width", 10)))
    rerank_width = max(support_width, int(params.get("rerank_width", 24)))
    support_temperature = float(params.get("support_temperature", 0.14))
    view_count = max(2, int(params.get("view_count", 4)))
    relation_slots = max(8, int(params.get("relation_slots", 24)))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 3))))
    relation_temperature = float(params.get("relation_temperature", 0.12))
    view_keep_ratio = float(params.get("view_keep_ratio", 0.50))
    support_relation_mix = float(params.get("support_relation_mix", 0.25))
    support_presence_mix = float(params.get("support_presence_mix", 0.25))
    disagreement_penalty = float(params.get("disagreement_penalty", 0.35))
    cograph_gain = float(params.get("cograph_gain", 0.018))
    uncertainty_width = float(params.get("uncertainty_width", 0.03))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    head_rng = np.random.default_rng(
        _method_seed("projective_multiquery_cograph_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    relation_basis = _safe_normalize(
        head_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )
    view_keep = min(hidden_dim, max(4, int(hidden_dim * max(0.05, view_keep_ratio))))
    raw_view_masks = np.abs(
        head_rng.normal(size=(view_count, hidden_dim)).astype(np.float32)
    )
    view_idx = np.argpartition(-raw_view_masks, view_keep - 1, axis=1)[:, :view_keep]
    view_masks = np.full((view_count, hidden_dim), 0.20, dtype=np.float32)
    strong_view_masks = 0.80 + np.take_along_axis(raw_view_masks, view_idx, axis=1)
    np.put_along_axis(view_masks, view_idx, strong_view_masks, axis=1)
    view_masks = view_masks / np.maximum(
        1e-6, np.mean(view_masks, axis=1, keepdims=True)
    )

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _multi_view_profile(wave_real: np.ndarray, wave_imag: np.ndarray) -> np.ndarray:
        energy = wave_real**2 + wave_imag**2
        masked_energy = energy[:, None, :] * view_masks[None, :, :]
        relation_logits = np.einsum("nvd,sd->nvs", masked_energy, relation_basis)
        flat_logits = relation_logits.reshape(-1, relation_slots)
        flat_profile = _topk_soft_assign(
            flat_logits, relation_top_k, relation_temperature
        )
        return flat_profile.reshape(-1, view_count, relation_slots).astype(np.float32)

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "view_profile": _multi_view_profile(wave_real, wave_imag),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "view_profile": _multi_view_profile(wave_real, wave_imag),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        doc_profiles = doc_state["view_profile"]
        if doc_profiles.shape[0] != doc_count:
            return carrier
        if doc_profiles.shape[1] != view_count:
            return carrier
        query_profiles = query_state["view_profile"]
        if query_profiles.shape[1] != view_count:
            return carrier
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_profiles = doc_profiles[candidate_idx]
        keep_support = min(support_width, keep_rerank)
        if keep_support <= 0:
            return carrier
        support_presence = np.zeros_like(candidate_carrier, dtype=np.float32)
        view_signal = np.zeros(
            (candidate_carrier.shape[0], keep_rerank, view_count), dtype=np.float32
        )
        for view_idx in range(view_count):
            view_query = query_profiles[:, view_idx, :]
            view_candidates = candidate_profiles[:, :, view_idx, :]
            view_relation = np.maximum(
                np.sum(view_candidates * view_query[:, None, :], axis=2), 0.0
            ).astype(np.float32)
            relation_scale = view_relation / np.maximum(
                1e-6, np.max(view_relation, axis=1, keepdims=True)
            )
            view_seed = candidate_carrier * (
                1.0 + support_relation_mix * relation_scale
            )
            support_local_idx = np.argpartition(-view_seed, keep_support - 1, axis=1)[
                :, :keep_support
            ]
            support_seed_scores = np.take_along_axis(
                view_seed, support_local_idx, axis=1
            )
            shifted = support_seed_scores - np.max(
                support_seed_scores, axis=1, keepdims=True
            )
            support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
                np.float32
            )
            weight_sum = np.sum(support_weights, axis=1, keepdims=True)
            weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
            support_weights = support_weights / weight_sum
            support_indicator = np.zeros_like(candidate_carrier, dtype=np.float32)
            np.put_along_axis(
                support_indicator, support_local_idx, support_weights, axis=1
            )
            support_presence = support_presence + support_indicator
            support_profiles = np.take_along_axis(
                view_candidates,
                support_local_idx[:, :, None],
                axis=1,
            )
            support_context = np.sum(
                support_profiles * support_weights[:, :, None], axis=1, dtype=np.float32
            )
            support_context = _safe_normalize(support_context)
            context_signal = np.maximum(
                np.sum(view_candidates * support_context[:, None, :], axis=2),
                0.0,
            ).astype(np.float32)
            # Der Zusatzscore zählt nur dort, wo ein Kandidat unter mehreren Query-Sichten
            # konsistent in denselben lokalen Antwortmodus fällt.
            view_signal[:, :, view_idx] = np.sqrt(
                np.maximum(view_relation * context_signal, 0.0)
            )
        support_presence = support_presence / float(max(1, view_count))
        candidate_signal = np.mean(view_signal, axis=2)
        if view_count > 1:
            candidate_signal = candidate_signal - disagreement_penalty * np.std(
                view_signal, axis=2
            )
        candidate_signal = candidate_signal + support_presence_mix * support_presence
        candidate_signal = candidate_signal - np.mean(
            candidate_signal, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_signal = candidate_signal / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        rerank_gain = np.maximum(
            0.0, 1.0 + cograph_gain * uncertainty_gate * candidate_signal
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_multiquery_cograph_v0",
        family="state_projective_multiquery_cograph",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "view_count": float(view_count),
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "view_keep_ratio": view_keep_ratio,
            "support_relation_mix": support_relation_mix,
            "support_presence_mix": support_presence_mix,
            "disagreement_penalty": disagreement_penalty,
            "cograph_gain": cograph_gain,
            "uncertainty_width": uncertainty_width,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_stable_codebook_head_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base_method_name = "projective_hilbert_embedding_v0"
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    phase_scale = float(params.get("phase_scale", 0.75))
    support_width = max(4, int(params.get("support_width", 10)))
    rerank_width = max(support_width, int(params.get("rerank_width", 24)))
    support_temperature = float(params.get("support_temperature", 0.14))
    codebook_size = max(8, int(params.get("codebook_size", 48)))
    code_top_k = min(codebook_size, max(1, int(params.get("code_top_k", 4))))
    code_temperature = float(params.get("code_temperature", 0.12))
    view_jitter = float(params.get("view_jitter", 0.08))
    code_gain = float(params.get("code_gain", 0.012))
    uncertainty_width = float(params.get("uncertainty_width", 0.02))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.80))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed(base_method_name, secret_key, dim))
    head_rng = np.random.default_rng(
        _method_seed("projective_stable_codebook_head_v0", secret_key, dim)
    )
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)
    codebook = _safe_normalize(
        head_rng.normal(size=(codebook_size, hidden_dim)).astype(np.float32)
    )
    base_mask = 1.0 + 0.12 * head_rng.normal(size=(hidden_dim,)).astype(np.float32)
    jitter_mask = view_jitter * head_rng.normal(size=(hidden_dim,)).astype(np.float32)
    code_mask_a = np.clip(base_mask + jitter_mask, 0.6, None)
    code_mask_b = np.clip(base_mask - jitter_mask, 0.6, None)
    code_mask_a = code_mask_a / np.maximum(1e-6, np.mean(code_mask_a))
    code_mask_b = code_mask_b / np.maximum(1e-6, np.mean(code_mask_b))

    def _encode_projective(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        real = y @ real_proj
        imag = y @ imag_proj
        phase = phase_scale * real + phase_bias[None, :]
        amp = np.sqrt(np.maximum(1e-6, 1.0 + 0.5 * np.tanh(imag))).astype(np.float32)
        wave_real = amp * np.cos(phase)
        wave_imag = amp * np.sin(phase)
        norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=1, keepdims=True))
        norm = np.where(norm == 0.0, 1.0, norm)
        wave_real = wave_real / norm
        wave_imag = wave_imag / norm
        public = _safe_normalize((wave_real**2 + wave_imag**2) @ public_mix)
        public = _mask_public_observation(
            base_method_name,
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return public, wave_real.astype(np.float32), wave_imag.astype(np.float32)

    def _carrier_score(
        wave_real_a: np.ndarray,
        wave_imag_a: np.ndarray,
        wave_real_b: np.ndarray,
        wave_imag_b: np.ndarray,
    ) -> np.ndarray:
        overlap = wave_real_a @ wave_real_b.T
        overlap = overlap + wave_imag_a @ wave_imag_b.T
        return overlap**2

    def _stable_codes(wave_real: np.ndarray, wave_imag: np.ndarray) -> np.ndarray:
        energy = wave_real**2 + wave_imag**2
        logits_a = (energy * code_mask_a[None, :]) @ codebook.T
        logits_b = (energy * code_mask_b[None, :]) @ codebook.T
        codes_a = _topk_soft_assign(logits_a, code_top_k, code_temperature)
        codes_b = _topk_soft_assign(logits_b, code_top_k, code_temperature)
        stable_codes = np.sqrt(np.maximum(codes_a * codes_b, 0.0)).astype(np.float32)
        code_sum = np.sum(stable_codes, axis=1, keepdims=True)
        code_sum = np.where(code_sum == 0.0, 1.0, code_sum)
        return stable_codes / code_sum

    def _encode_docs(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "stable_codes": _stable_codes(wave_real, wave_imag),
        }

    def _encode_queries(x: np.ndarray) -> StateMap:
        public, wave_real, wave_imag = _encode_projective(x)
        return {
            "public": public,
            "wave_real": wave_real,
            "wave_imag": wave_imag,
            "stable_codes": _stable_codes(wave_real, wave_imag),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = _carrier_score(
            query_state["wave_real"],
            query_state["wave_imag"],
            doc_state["wave_real"],
            doc_state["wave_imag"],
        )
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        stable_codes = doc_state["stable_codes"]
        if stable_codes.shape[0] != doc_count:
            return carrier
        code_score = query_state["stable_codes"] @ stable_codes.T
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier
        support_idx = np.argpartition(-carrier, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(carrier, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / max(1e-4, support_temperature)).astype(
            np.float32
        )
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum
        support_codes = stable_codes[support_idx]
        support_context = np.sum(
            support_codes * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        support_context = _safe_normalize(support_context)
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_codes = stable_codes[candidate_idx]
        candidate_code = np.take_along_axis(code_score, candidate_idx, axis=1)
        context_code = np.sum(
            candidate_codes * support_context[:, None, :], axis=2, dtype=np.float32
        )
        candidate_signal = np.sqrt(np.maximum(candidate_code * context_code, 0.0))
        candidate_signal = candidate_signal - np.mean(
            candidate_signal, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_signal = candidate_signal / signal_scale
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        rerank_gain = np.maximum(
            0.0, 1.0 + code_gain * uncertainty_gate * candidate_signal
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_stable_codebook_head_v0",
        family="state_projective_stable_codebook",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "phase_scale": phase_scale,
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "codebook_size": float(codebook_size),
            "code_top_k": float(code_top_k),
            "code_temperature": code_temperature,
            "view_jitter": view_jitter,
            "code_gain": code_gain,
            "uncertainty_width": uncertainty_width,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _projective_chart_resonance_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    charts = int(params.get("charts", 4))
    chart_dim = max(12, int(params.get("chart_dim", max(12, dim // 2))))
    phase_scale = float(params.get("phase_scale", 0.72))
    chart_gain = float(params.get("chart_gain", 1.15))
    chart_top_k = min(charts, max(1, int(params.get("chart_top_k", 2))))
    chart_temperature = float(params.get("chart_temperature", 0.22))
    amplitude_gain = float(params.get("amplitude_gain", 0.42))
    chart_weight_gain = float(params.get("chart_weight_gain", 0.16))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.82))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_chart_resonance_v0", secret_key, dim)
    )
    chart_gate = _safe_normalize(
        local_rng.normal(size=(charts, dim)).astype(np.float32)
    )
    real_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, chart_dim) for _ in range(charts)]
    )
    imag_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, chart_dim) for _ in range(charts)]
    )
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(charts, chart_dim)).astype(
        np.float32
    )
    public_mix = np.stack(
        [
            local_rng.normal(size=(chart_dim, max(4, public_dim // charts + 1))).astype(
                np.float32
            )
            for _ in range(charts)
        ]
    )

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        gate_logits = chart_gain * (y @ chart_gate.T)
        chart_weights = _topk_soft_assign(gate_logits, chart_top_k, chart_temperature)
        wave_real = []
        wave_imag = []
        public_chunks = []
        for chart in range(charts):
            real = y @ real_proj[chart]
            imag = y @ imag_proj[chart]
            phase = phase_scale * real + phase_bias[chart][None, :]
            amplitude = np.sqrt(
                np.maximum(1e-6, 1.0 + amplitude_gain * np.tanh(imag))
            ).astype(np.float32)
            chart_real = amplitude * np.cos(phase)
            chart_imag = amplitude * np.sin(phase)
            norm = np.sqrt(np.sum(chart_real**2 + chart_imag**2, axis=1, keepdims=True))
            norm = np.where(norm == 0.0, 1.0, norm)
            chart_real = chart_real / norm
            chart_imag = chart_imag / norm
            wave_real.append(chart_real.astype(np.float32))
            wave_imag.append(chart_imag.astype(np.float32))
            chart_energy = chart_real**2 + chart_imag**2
            chart_public = chart_energy @ public_mix[chart]
            chart_public = chart_weights[:, chart : chart + 1] * chart_public
            public_chunks.append(chart_public.astype(np.float32))
        public = _safe_normalize(np.hstack(public_chunks))
        public = _mask_public_observation(
            "projective_chart_resonance_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public,
            "chart_weights": chart_weights.astype(np.float32),
            "wave_real": np.stack(wave_real, axis=1),
            "wave_imag": np.stack(wave_imag, axis=1),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = np.zeros(
            (query_state["public"].shape[0], doc_state["public"].shape[0]),
            dtype=np.float32,
        )
        for chart in range(charts):
            overlap = (
                query_state["wave_real"][:, chart] @ doc_state["wave_real"][:, chart].T
            )
            overlap = (
                overlap
                + query_state["wave_imag"][:, chart]
                @ doc_state["wave_imag"][:, chart].T
            )
            carrier = carrier + (overlap**2)
        carrier = carrier / float(max(1, charts))
        chart_score = query_state["chart_weights"] @ doc_state["chart_weights"].T
        return (1.0 - chart_weight_gain) * carrier + chart_weight_gain * chart_score

    return EmbeddingStateMethod(
        method_name="projective_chart_resonance_v0",
        family="state_projective_chart",
        params={
            "dim": dim,
            "charts": float(charts),
            "chart_dim": float(chart_dim),
            "phase_scale": phase_scale,
            "chart_gain": chart_gain,
            "chart_top_k": float(chart_top_k),
            "chart_temperature": chart_temperature,
            "amplitude_gain": amplitude_gain,
            "chart_weight_gain": chart_weight_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _kernel_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    features = max(24, int(params.get("features", dim)))
    gamma = float(params.get("gamma", 1.10))
    public_ratio = float(params.get("public_ratio", 0.20))
    public_mask = float(params.get("public_mask", 0.72))
    public_chunk = int(params.get("public_chunk", 4))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("kernel_observable_embedding_v0", secret_key, dim)
    )
    omega = local_rng.normal(0.0, gamma, size=(dim, features)).astype(np.float32)
    phase = local_rng.uniform(0.0, 2.0 * math.pi, size=(features,)).astype(np.float32)
    gate = np.abs(local_rng.normal(size=(features,)).astype(np.float32))
    gate = gate / np.sum(gate)
    public_mix = local_rng.normal(size=(features, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        hidden = math.sqrt(2.0 / float(features)) * np.cos(y @ omega + phase[None, :])
        hidden = _safe_normalize(hidden.astype(np.float32))
        public = _safe_normalize(np.abs(hidden) @ public_mix)
        public = _mask_public_observation(
            "kernel_observable_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {"public": public, "hidden": hidden}

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        weighted_query = query_state["hidden"] * gate[None, :]
        return weighted_query @ doc_state["hidden"].T

    return EmbeddingStateMethod(
        method_name="kernel_observable_embedding_v0",
        family="state_kernel_observable",
        params={
            "dim": dim,
            "features": float(features),
            "gamma": gamma,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _calabi_yau_chart_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    charts = int(params.get("charts", 4))
    chart_dim = max(8, int(params.get("chart_dim", max(8, dim // 2))))
    fiber_gain = float(params.get("fiber_gain", 0.32))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.82))
    public_chunk = int(params.get("public_chunk", 5))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("calabi_yau_chart_embedding_v0", secret_key, dim)
    )
    chart_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, chart_dim) for _ in range(charts)]
    )
    fiber_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, chart_dim) for _ in range(charts)]
    )
    chart_gate = _safe_normalize(
        local_rng.normal(size=(charts, dim)).astype(np.float32)
    )
    public_mix = np.stack(
        [
            local_rng.normal(size=(chart_dim, max(4, public_dim // charts + 1))).astype(
                np.float32
            )
            for _ in range(charts)
        ]
    )

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        gate_logits = y @ chart_gate.T
        gate_logits = gate_logits - np.max(gate_logits, axis=1, keepdims=True)
        chart_weights = np.exp(gate_logits).astype(np.float32)
        chart_weights = chart_weights / np.sum(chart_weights, axis=1, keepdims=True)
        coords = []
        public_chunks = []
        for chart in range(charts):
            base = y @ chart_proj[chart]
            fiber = np.sin(y @ fiber_proj[chart])
            hidden = _safe_normalize(base + fiber_gain * fiber)
            coords.append(hidden.astype(np.float32))
            public_chunks.append(
                (np.abs(hidden) @ public_mix[chart]).astype(np.float32)
            )
        public = _safe_normalize(np.hstack(public_chunks))
        public = _mask_public_observation(
            "calabi_yau_chart_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            "public": public,
            "chart_coords": np.stack(coords, axis=1),
            "chart_weights": chart_weights.astype(np.float32),
        }

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        sim = np.zeros(
            (query_state["public"].shape[0], doc_state["public"].shape[0]),
            dtype=np.float32,
        )
        for chart in range(charts):
            chart_sim = (
                query_state["chart_coords"][:, chart]
                @ doc_state["chart_coords"][:, chart].T
            )
            weight_sim = np.outer(
                query_state["chart_weights"][:, chart],
                doc_state["chart_weights"][:, chart],
            )
            sim = sim + weight_sim * chart_sim
        return sim / float(max(1, charts))

    return EmbeddingStateMethod(
        method_name="calabi_yau_chart_embedding_v0",
        family="state_chart_manifold",
        params={
            "dim": dim,
            "charts": float(charts),
            "chart_dim": float(chart_dim),
            "fiber_gain": fiber_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _kahler_symplectic_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    symplectic_weight = float(params.get("symplectic_weight", 0.28))
    public_ratio = float(params.get("public_ratio", 0.20))
    public_mask = float(params.get("public_mask", 0.76))
    public_chunk = int(params.get("public_chunk", 4))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("kahler_symplectic_embedding_v0", secret_key, dim)
    )
    q_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    p_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    public_mix = local_rng.normal(size=(hidden_dim, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        q = _safe_normalize(y @ q_proj)
        p = _safe_normalize(y @ p_proj)
        energy = np.sqrt(np.maximum(1e-6, q**2 + p**2)).astype(np.float32)
        public = _safe_normalize(energy @ public_mix)
        public = _mask_public_observation(
            "kahler_symplectic_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {"public": public, "q": q.astype(np.float32), "p": p.astype(np.float32)}

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        metric = query_state["q"] @ doc_state["q"].T
        metric = metric + query_state["p"] @ doc_state["p"].T
        metric = metric / 2.0
        form = query_state["q"] @ doc_state["p"].T
        form = form - query_state["p"] @ doc_state["q"].T
        return metric - symplectic_weight * np.abs(form)

    return EmbeddingStateMethod(
        method_name="kahler_symplectic_embedding_v0",
        family="state_symplectic",
        params={
            "dim": dim,
            "hidden_dim": float(hidden_dim),
            "symplectic_weight": symplectic_weight,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _bregman_entropy_distortion_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    temperature = max(1e-4, float(params.get("temperature", 0.82)))
    curvature = max(0.05, float(params.get("curvature", 0.68)))
    noise_scale = max(0.0, float(params.get("noise_scale", 0.10)))
    recover_gain = float(params.get("recover_gain", 0.28))
    recover_gain = min(1.0, max(0.0, recover_gain))
    dual_mix = max(0.0, float(params.get("dual_mix", 0.04)))
    uncertainty_width = max(0.0, float(params.get("uncertainty_width", 0.015)))
    public_ratio = float(params.get("public_ratio", 0.18))
    public_mask = float(params.get("public_mask", 0.86))
    public_chunk = int(params.get("public_chunk", 6))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("bregman_entropy_distortion_v0", secret_key, dim)
    )
    dual_map = local_rng.normal(size=(hidden_dim, hidden_dim)).astype(np.float32)
    key_field = local_rng.uniform(0.7, 1.5, size=(hidden_dim,)).astype(np.float32)
    key_phase = local_rng.uniform(-math.pi, math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    key_order = np.argsort(key_field)
    bias = local_rng.normal(0.0, 0.08, size=(hidden_dim,)).astype(np.float32)
    noise_bank = local_rng.normal(0.0, 1.0, size=(1, hidden_dim)).astype(np.float32)
    public_mix = local_rng.normal(size=(2 * hidden_dim, public_dim)).astype(np.float32)

    def _energy_simplex(state: StateMap) -> np.ndarray:
        energy = np.maximum(
            1e-6, state["wave_real"] ** 2 + state["wave_imag"] ** 2
        ).astype(np.float32)
        energy_sum = np.sum(energy, axis=1, keepdims=True)
        energy_sum = np.where(energy_sum == 0.0, 1.0, energy_sum)
        return energy / energy_sum

    def _augment(state: StateMap, recover: float) -> StateMap:
        simplex = _energy_simplex(state)
        log_simplex = np.log(np.maximum(simplex, 1e-6)).astype(np.float32)
        dual = log_simplex @ dual_map
        dual = dual + bias[None, :]
        dual = dual + (1.0 - recover) * noise_scale * noise_bank
        dual = np.sign(dual) * np.log1p(curvature * np.abs(dual))
        primal = np.exp(dual / temperature).astype(np.float32)
        primal_sum = np.sum(primal, axis=1, keepdims=True)
        primal_sum = np.where(primal_sum == 0.0, 1.0, primal_sum)
        primal = primal / primal_sum
        recovered = (1.0 - recover) * dual + recover * log_simplex
        keyed_dual = np.sin(recovered * key_field[None, :] + key_phase[None, :]).astype(
            np.float32
        )
        sample_order = np.argsort(keyed_dual, axis=1)
        sorted_primal = np.take_along_axis(primal, sample_order, axis=1)
        sorted_dual = np.take_along_axis(keyed_dual, sample_order, axis=1)
        cumulative_primal = np.cumsum(sorted_primal, axis=1).astype(np.float32)
        aux_profile = _safe_normalize(
            np.hstack(
                [
                    np.sqrt(np.maximum(primal[:, key_order], 1e-6)),
                    keyed_dual[:, key_order],
                ]
            ).astype(np.float32)
        )
        public = _safe_normalize(
            np.hstack([cumulative_primal, np.tanh(sorted_dual)]).astype(np.float32)
            @ public_mix
        )
        public = _mask_public_observation(
            "bregman_entropy_distortion_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {
            **state,
            "public": public.astype(np.float32),
            "entropy_dual": recovered.astype(np.float32),
            "entropy_primal": primal.astype(np.float32),
            "entropy_keyed_dual": keyed_dual.astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _augment(base.encode_docs(x), 0.0)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _augment(base.encode_queries(x), recover_gain)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base.score(doc_state, query_state)
        dual_alignment = query_state["entropy_primal"] @ doc_state["entropy_primal"].T
        dual_alignment = dual_alignment / float(
            max(1, query_state["entropy_primal"].shape[1])
        )
        dual_gate = 0.5 + 0.5 * np.tanh(dual_alignment)
        uncertainty_gate = _relative_uncertainty_gate(carrier, uncertainty_width)
        mixed = (1.0 - dual_mix * uncertainty_gate) * carrier
        mixed = mixed + dual_mix * uncertainty_gate * dual_gate
        return mixed.astype(np.float32)

    return EmbeddingStateMethod(
        method_name="bregman_entropy_distortion_v0",
        family="state_entropy_dual",
        params={
            **base.params,
            "temperature": temperature,
            "curvature": curvature,
            "noise_scale": noise_scale,
            "recover_gain": recover_gain,
            "dual_mix": dual_mix,
            "uncertainty_width": uncertainty_width,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
    )


def _piecewise_isometry_breaker_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    regions = max(3, int(params.get("regions", 6)))
    region_top_k = min(regions, max(1, int(params.get("region_top_k", 2))))
    temperature = max(1e-4, float(params.get("temperature", 0.12)))
    boundary_power = max(0.5, float(params.get("boundary_power", 2.2)))
    shift_scale = max(0.0, float(params.get("shift_scale", 0.16)))
    noise_scale = max(0.0, float(params.get("noise_scale", 0.05)))
    recover_gain = min(1.0, max(0.0, float(params.get("recover_gain", 0.06))))
    public_ratio = float(params.get("public_ratio", 0.14))
    public_mask = float(params.get("public_mask", 0.94))
    public_chunk = int(params.get("public_chunk", 8))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("piecewise_isometry_breaker_v0", secret_key, dim)
    )
    region_centers = np.abs(
        local_rng.normal(size=(regions, hidden_dim)).astype(np.float32)
    )
    center_sum = np.sum(region_centers, axis=1, keepdims=True)
    center_sum = np.where(center_sum == 0.0, 1.0, center_sum)
    region_centers = region_centers / center_sum
    region_rot = np.stack(
        [_qr_orthogonal(local_rng, hidden_dim, hidden_dim) for _ in range(regions)]
    )
    region_shift = _safe_normalize(
        local_rng.normal(size=(regions, hidden_dim)).astype(np.float32)
    )
    route_mix = local_rng.normal(size=(regions, hidden_dim)).astype(np.float32)
    public_mix = local_rng.normal(size=(2 * hidden_dim + regions, public_dim)).astype(
        np.float32
    )
    noise_bank = local_rng.normal(size=(1, hidden_dim)).astype(np.float32)

    def _energy_simplex(state: StateMap) -> np.ndarray:
        energy = np.maximum(
            1e-6, state["wave_real"] ** 2 + state["wave_imag"] ** 2
        ).astype(np.float32)
        energy_sum = np.sum(energy, axis=1, keepdims=True)
        energy_sum = np.where(energy_sum == 0.0, 1.0, energy_sum)
        return energy / energy_sum

    def _piecewise_route(
        simplex: np.ndarray, recover: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        region_logits = simplex @ region_centers.T
        route_temp = max(1e-4, temperature * (1.0 - 0.35 * recover))
        region_weights = _topk_soft_assign(region_logits, region_top_k, route_temp)
        region_weights = np.power(
            np.clip(region_weights, 0.0, 1.0), boundary_power
        ).astype(np.float32)
        weight_sum = np.sum(region_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        region_weights = region_weights / weight_sum
        routed = np.zeros_like(simplex, dtype=np.float32)
        for region in range(regions):
            local = simplex - region_centers[region][None, :]
            rotated = local @ region_rot[region].T
            shifted = rotated + shift_scale * region_shift[region][None, :]
            routed = routed + region_weights[:, region : region + 1] * shifted
        routed = routed + (1.0 - recover) * noise_scale * noise_bank
        if recover > 0.0:
            routed = (1.0 - recover) * routed + recover * simplex
        return region_weights.astype(np.float32), routed.astype(np.float32)

    def _augment(state: StateMap, recover: float) -> StateMap:
        simplex = _energy_simplex(state)
        region_weights, routed = _piecewise_route(simplex, recover)
        order_basis = routed + 0.35 * (region_weights @ route_mix)
        sample_order = np.argsort(order_basis, axis=1)
        sorted_simplex = np.take_along_axis(simplex, sample_order, axis=1)
        sorted_routed = np.take_along_axis(routed, sample_order, axis=1)
        cumulative_simplex = np.cumsum(sorted_simplex, axis=1).astype(np.float32)
        public = _safe_normalize(
            np.hstack(
                [cumulative_simplex, np.tanh(sorted_routed), region_weights]
            ).astype(np.float32)
            @ public_mix
        )
        public = _mask_public_observation(
            "piecewise_isometry_breaker_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        aux_profile = _safe_normalize(
            np.hstack([region_weights, _safe_normalize(routed)]).astype(np.float32)
        )
        return {
            **state,
            "public": public.astype(np.float32),
            "piecewise_region_weights": region_weights.astype(np.float32),
            "piecewise_routed": routed.astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _augment(base.encode_docs(x), 0.0)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _augment(base.encode_queries(x), recover_gain)

    return EmbeddingStateMethod(
        method_name="piecewise_isometry_breaker_v0",
        family="state_projective_piecewise_isometry_breaker",
        params={
            **base.params,
            "regions": float(regions),
            "region_top_k": float(region_top_k),
            "temperature": temperature,
            "boundary_power": boundary_power,
            "shift_scale": shift_scale,
            "noise_scale": noise_scale,
            "recover_gain": recover_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base.score,
    )


def _keyed_phase_vortex_flow_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    vortices = max(3, int(params.get("vortices", 6)))
    vortex_top_k = min(vortices, max(1, int(params.get("vortex_top_k", 2))))
    swirl = max(0.05, float(params.get("swirl", 1.10)))
    chirality = float(params.get("chirality", 0.55))
    radial_power = max(0.25, float(params.get("radial_power", 1.45)))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.10)))
    noise_scale = max(0.0, float(params.get("noise_scale", 0.05)))
    recover_gain = min(1.0, max(0.0, float(params.get("recover_gain", 0.10))))
    public_ratio = float(params.get("public_ratio", 0.14))
    public_mask = float(params.get("public_mask", 0.92))
    public_chunk = int(params.get("public_chunk", 8))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("keyed_phase_vortex_flow_v0", secret_key, dim)
    )
    route_bank = _safe_normalize(
        np.abs(local_rng.normal(size=(vortices, hidden_dim)).astype(np.float32)) + 0.05
    )
    route_bias = local_rng.uniform(-0.18, 0.18, size=(vortices,)).astype(np.float32)
    center_real = local_rng.normal(0.0, 0.55, size=(vortices, hidden_dim)).astype(
        np.float32
    )
    center_imag = local_rng.normal(0.0, 0.55, size=(vortices, hidden_dim)).astype(
        np.float32
    )
    charge = local_rng.choice([-1.0, 1.0], size=(vortices, hidden_dim)).astype(
        np.float32
    )
    phase_bias = local_rng.uniform(
        -math.pi, math.pi, size=(vortices, hidden_dim)
    ).astype(np.float32)
    route_mix = local_rng.normal(size=(vortices, hidden_dim)).astype(np.float32)
    public_mix = local_rng.normal(size=(3 * hidden_dim + vortices, public_dim)).astype(
        np.float32
    )
    noise_real = local_rng.normal(size=(1, hidden_dim)).astype(np.float32)
    noise_imag = local_rng.normal(size=(1, hidden_dim)).astype(np.float32)

    def _energy_simplex(state: StateMap) -> np.ndarray:
        energy = np.maximum(
            1e-6, state["wave_real"] ** 2 + state["wave_imag"] ** 2
        ).astype(np.float32)
        energy_sum = np.sum(energy, axis=1, keepdims=True)
        energy_sum = np.where(energy_sum == 0.0, 1.0, energy_sum)
        return energy / energy_sum

    # Vortex singularities only touch the public and aux channels; score stays projective.
    def _augment(state: StateMap, recover: float) -> StateMap:
        wave_real = state["wave_real"]
        wave_imag = state["wave_imag"]
        simplex = _energy_simplex(state)
        base_phase = np.arctan2(wave_imag, wave_real).astype(np.float32)
        route_logits = simplex @ route_bank.T
        route_logits = route_logits + 0.10 * (np.cos(base_phase) @ charge.T) / float(
            max(1, hidden_dim)
        )
        route_logits = route_logits + route_bias[None, :]
        local_temperature = max(1e-4, route_temperature * (1.0 - 0.35 * recover))
        route_weights = _topk_soft_assign(
            route_logits, vortex_top_k, local_temperature
        ).astype(np.float32)

        local_real = wave_real[:, None, :] - center_real[None, :, :]
        local_imag = wave_imag[:, None, :] - center_imag[None, :, :]
        radius = np.sqrt(np.maximum(local_real**2 + local_imag**2, 1e-6)).astype(
            np.float32
        )
        circulation = np.arctan2(local_imag, local_real).astype(np.float32)
        flow = base_phase[:, None, :] + chirality * circulation
        flow = flow + swirl * charge[None, :, :] / np.power(1.0 + radius, radial_power)
        flow = flow + phase_bias[None, :, :]
        if recover > 0.0:
            flow = (1.0 - recover) * flow + recover * base_phase[:, None, :]

        vortex_real = np.sum(route_weights[:, :, None] * np.cos(flow), axis=1).astype(
            np.float32
        )
        vortex_imag = np.sum(route_weights[:, :, None] * np.sin(flow), axis=1).astype(
            np.float32
        )
        singularity = np.sum(
            route_weights[:, :, None] * charge[None, :, :] / np.maximum(radius, 1e-3),
            axis=1,
        )
        singularity = np.tanh(singularity / float(max(1, vortices))).astype(np.float32)
        circulation_profile = np.sum(
            route_weights[:, :, None] * np.sin(circulation) * charge[None, :, :],
            axis=1,
        ).astype(np.float32)

        routed_real = vortex_real + 0.25 * circulation_profile
        routed_imag = vortex_imag + 0.20 * singularity
        if noise_scale > 0.0:
            routed_real = routed_real + (1.0 - recover) * noise_scale * noise_real
            routed_imag = routed_imag + (1.0 - recover) * noise_scale * noise_imag
        if recover > 0.0:
            recover_real = simplex * np.cos(base_phase)
            recover_imag = simplex * np.sin(base_phase)
            routed_real = (1.0 - recover) * routed_real + recover * recover_real
            routed_imag = (1.0 - recover) * routed_imag + recover * recover_imag

        order_basis = (
            routed_real + 0.35 * routed_imag + 0.15 * (route_weights @ route_mix)
        )
        sample_order = np.argsort(order_basis, axis=1)
        sorted_simplex = np.take_along_axis(simplex, sample_order, axis=1)
        sorted_real = np.take_along_axis(routed_real, sample_order, axis=1)
        sorted_imag = np.take_along_axis(routed_imag, sample_order, axis=1)
        cumulative_simplex = np.cumsum(sorted_simplex, axis=1).astype(np.float32)
        public = _safe_normalize(
            np.hstack(
                [
                    cumulative_simplex,
                    np.tanh(sorted_real),
                    np.tanh(sorted_imag),
                    route_weights,
                ]
            ).astype(np.float32)
            @ public_mix
        )
        public = _mask_public_observation(
            "keyed_phase_vortex_flow_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        vortex_flux = _safe_normalize(routed_real.astype(np.float32, copy=False))
        vortex_charge = _safe_normalize(
            (0.65 * circulation_profile + 0.35 * singularity).astype(
                np.float32, copy=False
            )
        )
        aux_profile = _safe_normalize(
            np.hstack([route_weights, vortex_flux, vortex_charge]).astype(np.float32)
        )
        return {
            **state,
            "public": public.astype(np.float32),
            "vortex_route_weights": route_weights,
            "vortex_flux": vortex_flux.astype(np.float32),
            "vortex_charge": vortex_charge.astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _augment(base.encode_docs(x), 0.0)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _augment(base.encode_queries(x), recover_gain)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        route_align = np.sqrt(
            np.maximum(
                query_state["vortex_route_weights"]
                @ doc_state["vortex_route_weights"].T,
                0.0,
            )
        ).astype(np.float32)
        flux_align = query_state["vortex_flux"] @ doc_state["vortex_flux"].T
        charge_align = query_state["vortex_charge"] @ doc_state["vortex_charge"].T
        aux_scores = (0.55 * flux_align + 0.45 * charge_align) * (
            0.30 + 0.70 * route_align
        )
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return aux_scores / scale

    return EmbeddingStateMethod(
        method_name="keyed_phase_vortex_flow_v0",
        family="state_projective_phase_vortex_breaker",
        params={
            **base.params,
            "vortices": float(vortices),
            "vortex_top_k": float(vortex_top_k),
            "swirl": swirl,
            "chirality": chirality,
            "radial_power": radial_power,
            "route_temperature": route_temperature,
            "noise_scale": noise_scale,
            "recover_gain": recover_gain,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base.score,
        aux_score=_aux_score,
    )


def _projective_bregman_vortex_hybrid_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    temperature = max(1e-4, float(params.get("temperature", 0.62)))
    curvature = max(0.05, float(params.get("curvature", 1.10)))
    entropy_noise_scale = max(0.0, float(params.get("entropy_noise_scale", 0.18)))
    entropy_recover_gain = min(
        1.0, max(0.0, float(params.get("entropy_recover_gain", 0.12)))
    )
    vortices = max(3, int(params.get("vortices", 8)))
    vortex_top_k = min(vortices, max(1, int(params.get("vortex_top_k", 2))))
    swirl = max(0.05, float(params.get("swirl", 1.45)))
    chirality = float(params.get("chirality", 0.72))
    radial_power = max(0.25, float(params.get("radial_power", 1.75)))
    route_temperature = max(1e-4, float(params.get("route_temperature", 0.08)))
    vortex_noise_scale = max(0.0, float(params.get("vortex_noise_scale", 0.04)))
    vortex_recover_gain = min(
        1.0, max(0.0, float(params.get("vortex_recover_gain", 0.10)))
    )
    route_dual_mix = max(0.0, float(params.get("route_dual_mix", 0.24)))
    phase_dual_mix = float(params.get("phase_dual_mix", 0.30))
    public_ratio = float(params.get("public_ratio", 0.12))
    public_mask = float(params.get("public_mask", 0.95))
    public_chunk = int(params.get("public_chunk", 10))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_bregman_vortex_hybrid_v0", secret_key, dim)
    )
    dual_map = local_rng.normal(size=(hidden_dim, hidden_dim)).astype(np.float32)
    key_field = local_rng.uniform(0.7, 1.5, size=(hidden_dim,)).astype(np.float32)
    key_phase = local_rng.uniform(-math.pi, math.pi, size=(hidden_dim,)).astype(
        np.float32
    )
    key_order = np.argsort(key_field)
    dual_bias = local_rng.normal(0.0, 0.08, size=(hidden_dim,)).astype(np.float32)
    entropy_noise_bank = local_rng.normal(0.0, 1.0, size=(1, hidden_dim)).astype(
        np.float32
    )
    route_bank = _safe_normalize(
        np.abs(local_rng.normal(size=(vortices, hidden_dim)).astype(np.float32)) + 0.05
    )
    route_bias = local_rng.uniform(-0.18, 0.18, size=(vortices,)).astype(np.float32)
    center_real = local_rng.normal(0.0, 0.55, size=(vortices, hidden_dim)).astype(
        np.float32
    )
    center_imag = local_rng.normal(0.0, 0.55, size=(vortices, hidden_dim)).astype(
        np.float32
    )
    charge = local_rng.choice([-1.0, 1.0], size=(vortices, hidden_dim)).astype(
        np.float32
    )
    phase_bias = local_rng.uniform(
        -math.pi, math.pi, size=(vortices, hidden_dim)
    ).astype(np.float32)
    route_mix = local_rng.normal(size=(vortices, hidden_dim)).astype(np.float32)
    public_mix = local_rng.normal(size=(4 * hidden_dim + vortices, public_dim)).astype(
        np.float32
    )
    vortex_noise_real = local_rng.normal(size=(1, hidden_dim)).astype(np.float32)
    vortex_noise_imag = local_rng.normal(size=(1, hidden_dim)).astype(np.float32)

    def _energy_simplex(state: StateMap) -> np.ndarray:
        energy = np.maximum(
            1e-6, state["wave_real"] ** 2 + state["wave_imag"] ** 2
        ).astype(np.float32)
        energy_sum = np.sum(energy, axis=1, keepdims=True)
        energy_sum = np.where(energy_sum == 0.0, 1.0, energy_sum)
        return energy / energy_sum

    def _augment(
        state: StateMap, entropy_recover: float, vortex_recover: float
    ) -> StateMap:
        simplex = _energy_simplex(state)
        log_simplex = np.log(np.maximum(simplex, 1e-6)).astype(np.float32)
        dual = log_simplex @ dual_map
        dual = dual + dual_bias[None, :]
        dual = dual + (1.0 - entropy_recover) * entropy_noise_scale * entropy_noise_bank
        dual = np.sign(dual) * np.log1p(curvature * np.abs(dual))
        primal = np.exp(dual / temperature).astype(np.float32)
        primal_sum = np.sum(primal, axis=1, keepdims=True)
        primal_sum = np.where(primal_sum == 0.0, 1.0, primal_sum)
        primal = primal / primal_sum
        recovered = (1.0 - entropy_recover) * dual + entropy_recover * log_simplex
        keyed_dual = np.sin(recovered * key_field[None, :] + key_phase[None, :]).astype(
            np.float32
        )
        dual_phase = np.tanh(keyed_dual).astype(np.float32)

        wave_real = state["wave_real"]
        wave_imag = state["wave_imag"]
        base_phase = np.arctan2(wave_imag, wave_real).astype(np.float32)
        route_source = _safe_normalize(
            primal + route_dual_mix * np.abs(dual_phase)
        ).astype(np.float32)
        route_logits = route_source @ route_bank.T
        route_logits = route_logits + 0.10 * (np.cos(base_phase) @ charge.T) / float(
            max(1, hidden_dim)
        )
        route_logits = route_logits + 0.08 * (dual_phase @ charge.T) / float(
            max(1, hidden_dim)
        )
        route_logits = route_logits + route_bias[None, :]
        local_temperature = max(1e-4, route_temperature * (1.0 - 0.35 * vortex_recover))
        route_weights = _topk_soft_assign(
            route_logits, vortex_top_k, local_temperature
        ).astype(np.float32)

        local_real = wave_real[:, None, :] - center_real[None, :, :]
        local_imag = wave_imag[:, None, :] - center_imag[None, :, :]
        radius = np.sqrt(np.maximum(local_real**2 + local_imag**2, 1e-6)).astype(
            np.float32
        )
        circulation = np.arctan2(local_imag, local_real).astype(np.float32)
        flow = base_phase[:, None, :] + phase_dual_mix * dual_phase[:, None, :]
        flow = flow + chirality * circulation
        flow = flow + swirl * charge[None, :, :] / np.power(1.0 + radius, radial_power)
        flow = flow + phase_bias[None, :, :]
        if vortex_recover > 0.0:
            recover_phase = base_phase + 0.5 * phase_dual_mix * dual_phase
            flow = (1.0 - vortex_recover) * flow + vortex_recover * recover_phase[
                :, None, :
            ]

        vortex_real = np.sum(route_weights[:, :, None] * np.cos(flow), axis=1).astype(
            np.float32
        )
        vortex_imag = np.sum(route_weights[:, :, None] * np.sin(flow), axis=1).astype(
            np.float32
        )
        singularity = np.sum(
            route_weights[:, :, None] * charge[None, :, :] / np.maximum(radius, 1e-3),
            axis=1,
        )
        singularity = np.tanh(singularity / float(max(1, vortices))).astype(np.float32)
        circulation_profile = np.sum(
            route_weights[:, :, None] * np.sin(circulation) * charge[None, :, :],
            axis=1,
        ).astype(np.float32)

        routed_real = vortex_real + 0.25 * circulation_profile + 0.18 * dual_phase
        routed_imag = (
            vortex_imag + 0.20 * singularity + 0.12 * np.sqrt(np.maximum(primal, 1e-6))
        )
        if vortex_noise_scale > 0.0:
            routed_real = (
                routed_real
                + (1.0 - vortex_recover) * vortex_noise_scale * vortex_noise_real
            )
            routed_imag = (
                routed_imag
                + (1.0 - vortex_recover) * vortex_noise_scale * vortex_noise_imag
            )
        if vortex_recover > 0.0:
            recover_real = primal * np.cos(
                base_phase + 0.5 * phase_dual_mix * dual_phase
            )
            recover_imag = primal * np.sin(
                base_phase + 0.5 * phase_dual_mix * dual_phase
            )
            routed_real = (
                1.0 - vortex_recover
            ) * routed_real + vortex_recover * recover_real
            routed_imag = (
                1.0 - vortex_recover
            ) * routed_imag + vortex_recover * recover_imag

        order_basis = routed_real + 0.35 * routed_imag + 0.18 * dual_phase
        order_basis = order_basis + 0.15 * (route_weights @ route_mix)
        sample_order = np.argsort(order_basis, axis=1)
        sorted_primal = np.take_along_axis(primal, sample_order, axis=1)
        sorted_dual = np.take_along_axis(dual_phase, sample_order, axis=1)
        sorted_real = np.take_along_axis(routed_real, sample_order, axis=1)
        sorted_imag = np.take_along_axis(routed_imag, sample_order, axis=1)
        cumulative_primal = np.cumsum(sorted_primal, axis=1).astype(np.float32)
        public = _safe_normalize(
            np.hstack(
                [
                    cumulative_primal,
                    sorted_dual,
                    np.tanh(sorted_real),
                    np.tanh(sorted_imag),
                    route_weights,
                ]
            ).astype(np.float32)
            @ public_mix
        )
        public = _mask_public_observation(
            "projective_bregman_vortex_hybrid_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        vortex_flux = _safe_normalize(routed_real.astype(np.float32, copy=False))
        vortex_charge = _safe_normalize(
            (0.65 * circulation_profile + 0.35 * singularity).astype(
                np.float32, copy=False
            )
        )
        aux_profile = _safe_normalize(
            np.hstack(
                [
                    route_weights,
                    np.sqrt(np.maximum(primal[:, key_order], 1e-6)),
                    vortex_flux,
                    vortex_charge,
                ]
            ).astype(np.float32)
        )
        return {
            **state,
            "public": public.astype(np.float32),
            "entropy_primal": primal.astype(np.float32),
            "entropy_keyed_dual": dual_phase.astype(np.float32),
            "vortex_route_weights": route_weights.astype(np.float32),
            "vortex_flux": vortex_flux.astype(np.float32),
            "vortex_charge": vortex_charge.astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _augment(base.encode_docs(x), 0.0, 0.0)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _augment(
            base.encode_queries(x), entropy_recover_gain, vortex_recover_gain
        )

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        route_align = np.sqrt(
            np.maximum(
                query_state["vortex_route_weights"]
                @ doc_state["vortex_route_weights"].T,
                0.0,
            )
        ).astype(np.float32)
        flux_align = query_state["vortex_flux"] @ doc_state["vortex_flux"].T
        charge_align = query_state["vortex_charge"] @ doc_state["vortex_charge"].T
        entropy_align = np.sqrt(
            np.maximum(
                query_state["entropy_primal"] @ doc_state["entropy_primal"].T,
                0.0,
            )
            / float(max(1, hidden_dim))
        ).astype(np.float32)
        dual_align = (
            query_state["entropy_keyed_dual"] @ doc_state["entropy_keyed_dual"].T
        ) / float(max(1, hidden_dim))
        dual_gate = 0.5 + 0.5 * np.tanh(dual_align)
        aux_scores = 0.40 * flux_align + 0.28 * charge_align + 0.32 * entropy_align
        aux_scores = aux_scores * (0.30 + 0.70 * route_align)
        aux_scores = aux_scores * (0.55 + 0.45 * dual_gate)
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return (aux_scores / scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_bregman_vortex_hybrid_v0",
        family="state_projective_bregman_vortex_hybrid",
        params={
            **base.params,
            "temperature": temperature,
            "curvature": curvature,
            "entropy_noise_scale": entropy_noise_scale,
            "entropy_recover_gain": entropy_recover_gain,
            "vortices": float(vortices),
            "vortex_top_k": float(vortex_top_k),
            "swirl": swirl,
            "chirality": chirality,
            "radial_power": radial_power,
            "route_temperature": route_temperature,
            "vortex_noise_scale": vortex_noise_scale,
            "vortex_recover_gain": vortex_recover_gain,
            "route_dual_mix": route_dual_mix,
            "phase_dual_mix": phase_dual_mix,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base.score,
        aux_score=_aux_score,
    )


def _projective_holographic_boundary_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    base = _projective_hilbert_build(rng, params)
    dim = int(params.get("dim", 0))
    hidden_dim = max(12, int(params.get("hidden_dim", dim)))
    boundary_ratio = min(max(float(params.get("boundary_ratio", 0.22)), 0.08), 0.55)
    fringe = max(0.0, float(params.get("fringe", 0.22)))
    recover_gain = min(max(float(params.get("recover_gain", 0.22)), 0.0), 1.0)
    shell_mix = min(max(float(params.get("shell_mix", 0.18)), 0.0), 1.0)
    public_ratio = float(params.get("public_ratio", 0.12))
    public_mask = float(params.get("public_mask", 0.95))
    public_chunk = int(params.get("public_chunk", 10))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("projective_holographic_boundary_v0", secret_key, dim)
    )

    rotate = _qr_orthogonal(local_rng, hidden_dim, hidden_dim)
    freq_len = hidden_dim // 2 + 1
    keep = max(2, min(freq_len, int(freq_len * boundary_ratio)))
    boundary_phase = np.exp(
        1j * local_rng.uniform(-math.pi, math.pi, size=(keep,))
    ).astype(np.complex64)
    boundary_gain = local_rng.uniform(0.75, 1.25, size=(keep,)).astype(np.float32)
    fringe_mix = local_rng.normal(size=(keep, hidden_dim)).astype(np.float32)
    boundary_mix = local_rng.normal(size=(hidden_dim + 3 * keep, public_dim)).astype(
        np.float32
    )

    def _energy_simplex(state: StateMap) -> np.ndarray:
        energy = np.maximum(
            1e-6, state["wave_real"] ** 2 + state["wave_imag"] ** 2
        ).astype(np.float32)
        energy_sum = np.sum(energy, axis=1, keepdims=True)
        energy_sum = np.where(energy_sum == 0.0, 1.0, energy_sum)
        return energy / energy_sum

    def _bulk_summary(spectrum: np.ndarray) -> np.ndarray:
        if spectrum.shape[1] <= keep:
            return np.zeros((spectrum.shape[0], keep), dtype=np.float32)
        bulk = np.abs(spectrum[:, keep:]).astype(np.float32)
        splits = np.array_split(np.arange(bulk.shape[1]), keep)
        summary = np.zeros((bulk.shape[0], keep), dtype=np.float32)
        for idx, split in enumerate(splits):
            if len(split) == 0:
                continue
            summary[:, idx] = np.mean(bulk[:, split], axis=1)
        return summary

    def _augment(state: StateMap, recover: float) -> StateMap:
        simplex = _energy_simplex(state)
        rotated = _safe_normalize(simplex @ rotate)
        spectrum = np.fft.rfft(rotated, axis=1).astype(np.complex64)
        shell = np.array(spectrum[:, :keep], copy=True)
        shell = shell * boundary_phase[None, :] * boundary_gain[None, :]
        recover_mix = recover_gain * recover
        if recover_mix > 0.0:
            shell = (1.0 - recover_mix) * shell + recover_mix * spectrum[:, :keep]

        boundary_spectrum = np.zeros_like(spectrum)
        boundary_spectrum[:, :keep] = shell
        boundary_trace = np.fft.irfft(boundary_spectrum, n=hidden_dim, axis=1).astype(
            np.float32
        )
        boundary_trace = _safe_normalize(boundary_trace)

        shell_mag = np.abs(shell).astype(np.float32)
        shell_real = np.real(shell).astype(np.float32)
        shell_imag = np.imag(shell).astype(np.float32)
        shell_order = np.argsort(shell_real + shell_mix * shell_imag, axis=1)
        sorted_mag = np.take_along_axis(shell_mag, shell_order, axis=1)
        sorted_phase = np.take_along_axis(
            np.angle(shell).astype(np.float32), shell_order, axis=1
        )
        cumulative_mag = np.cumsum(sorted_mag, axis=1).astype(np.float32)

        fringe_summary = _bulk_summary(spectrum)
        fringe_hidden = np.tanh(fringe_summary @ fringe_mix).astype(np.float32)
        if fringe > 0.0:
            fringe_hidden = fringe * fringe_hidden
        if recover_mix > 0.0:
            fringe_hidden = (1.0 - recover_mix) * fringe_hidden + recover_mix * rotated

        public_context = np.hstack(
            [boundary_trace, cumulative_mag, np.tanh(sorted_phase), fringe_summary]
        ).astype(np.float32)
        public = _safe_normalize(np.tanh(public_context @ boundary_mix))
        public = _mask_public_observation(
            "projective_holographic_boundary_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        aux_profile = _safe_normalize(
            np.hstack([boundary_trace, sorted_mag, fringe_summary]).astype(np.float32)
        )
        return {
            **state,
            "public": public.astype(np.float32),
            "holographic_boundary": boundary_trace.astype(np.float32),
            "holographic_fringe": _safe_normalize(fringe_hidden).astype(np.float32),
            "aux_operator_profile": aux_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        return _augment(base.encode_docs(x), 0.0)

    def _encode_queries(x: np.ndarray) -> StateMap:
        return _augment(base.encode_queries(x), 1.0)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        boundary_align = (
            query_state["holographic_boundary"] @ doc_state["holographic_boundary"].T
        )
        fringe_align = (
            query_state["holographic_fringe"] @ doc_state["holographic_fringe"].T
        )
        aux_scores = 0.7 * boundary_align + 0.3 * fringe_align
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return (aux_scores / scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_holographic_boundary_v0",
        family="state_projective_holographic_boundary",
        params={
            **base.params,
            "boundary_ratio": boundary_ratio,
            "fringe": fringe,
            "recover_gain": recover_gain,
            "shell_mix": shell_mix,
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base.score,
        aux_score=_aux_score,
    )


def _projective_holographic_support_bipartite_state_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    role_count = max(8, min(48, int(params.get("role_count", 20))))
    role_top_k = min(role_count, max(1, int(params.get("role_top_k", 2))))
    role_temperature = max(1e-4, float(params.get("role_temperature", 0.08)))
    support_width = max(3, int(params.get("support_width", 8)))
    rerank_width = max(4, int(params.get("rerank_width", 24)))
    support_temperature = max(1e-4, float(params.get("support_temperature", 0.12)))
    support_floor = min(max(float(params.get("support_floor", 0.20)), 0.0), 0.9)
    wave_mix = min(max(float(params.get("wave_mix", 0.26)), 0.0), 1.0)
    boundary_mix = min(max(float(params.get("boundary_mix", 0.46)), 0.0), 1.0)
    fringe_mix = min(max(float(params.get("fringe_mix", 0.16)), 0.0), 1.0)
    support_mix = min(max(float(params.get("support_mix", 0.34)), 0.0), 1.0)
    bridge_mix = min(max(float(params.get("bridge_mix", 0.28)), 0.0), 1.0)
    base_aux_mix = min(max(float(params.get("base_aux_mix", 0.24)), 0.0), 1.0)
    profile_mix = max(0.0, float(params.get("profile_mix", 0.20)))
    duality_gain = max(0.0, float(params.get("duality_gain", 0.006)))
    uncertainty_width = max(1e-4, float(params.get("uncertainty_width", 0.022)))
    dim = int(params.get("dim", 0))
    secret_key = str(params.get("secret_key", ""))

    base_method = _projective_holographic_boundary_build(rng, params)
    base_aux_score = base_method.aux_score
    sample_state = base_method.encode_docs(np.zeros((1, dim), dtype=np.float32))
    hidden_dim = sample_state["wave_real"].shape[1]
    fringe_dim = sample_state["holographic_fringe"].shape[1]
    local_rng = np.random.default_rng(
        _method_seed(
            "projective_holographic_support_bipartite_state_v0", secret_key, dim
        )
    )

    left_wave_real = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_wave_imag = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_wave_real = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_wave_imag = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_boundary = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_boundary = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_fringe = _safe_normalize(
        local_rng.normal(size=(role_count, fringe_dim)).astype(np.float32)
    )
    right_fringe = _safe_normalize(
        local_rng.normal(size=(role_count, fringe_dim)).astype(np.float32)
    )
    role_weight = np.abs(local_rng.normal(size=(role_count,)).astype(np.float32))
    role_weight = role_weight / np.maximum(1e-6, np.sum(role_weight))
    role_permutation = local_rng.permutation(role_count)

    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        total = np.sum(x, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        return (x / total).astype(np.float32)

    def _merge_aux_profile(
        base_profile: np.ndarray | None, bridge_profile: np.ndarray
    ) -> np.ndarray:
        if base_profile is None:
            return _safe_normalize(bridge_profile.astype(np.float32, copy=False))
        if base_profile.shape[0] != bridge_profile.shape[0]:
            return base_profile.astype(np.float32, copy=False)
        merged = np.concatenate(
            [
                base_profile.astype(np.float32, copy=False),
                profile_mix * bridge_profile.astype(np.float32, copy=False),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _role_profiles(state: StateMap, recover: float) -> Dict[str, np.ndarray]:
        left_wave = np.maximum(
            state["wave_real"] @ left_wave_real.T
            + state["wave_imag"] @ left_wave_imag.T,
            0.0,
        ).astype(np.float32) / float(max(1, hidden_dim))
        right_wave = np.maximum(
            state["wave_real"] @ right_wave_real.T
            + state["wave_imag"] @ right_wave_imag.T,
            0.0,
        ).astype(np.float32) / float(max(1, hidden_dim))
        left_boundary_match = np.clip(
            state["holographic_boundary"] @ left_boundary.T, 0.0, 1.0
        ).astype(np.float32)
        right_boundary_match = np.clip(
            state["holographic_boundary"] @ right_boundary.T, 0.0, 1.0
        ).astype(np.float32)
        left_fringe_match = np.clip(
            state["holographic_fringe"] @ left_fringe.T, 0.0, 1.0
        ).astype(np.float32)
        right_fringe_match = np.clip(
            state["holographic_fringe"] @ right_fringe.T, 0.0, 1.0
        ).astype(np.float32)

        left_logits = wave_mix * left_wave + boundary_mix * left_boundary_match
        left_logits = left_logits + fringe_mix * left_fringe_match
        right_logits = wave_mix * right_wave + boundary_mix * right_boundary_match
        right_logits = right_logits + fringe_mix * right_fringe_match
        if recover > 0.0:
            left_logits = left_logits + recover * (
                left_boundary_match
                - np.mean(left_boundary_match, axis=1, keepdims=True)
            )
            right_logits = right_logits + recover * (
                right_boundary_match
                - np.mean(right_boundary_match, axis=1, keepdims=True)
            )

        local_temperature = max(1e-4, role_temperature * (1.0 - 0.20 * recover))
        left_profile = _topk_soft_assign(left_logits, role_top_k, local_temperature)
        right_profile = _topk_soft_assign(right_logits, role_top_k, local_temperature)
        left_profile = _normalize_rows(left_profile)
        right_profile = _normalize_rows(right_profile)
        bridge_delta = np.abs(left_profile - right_profile[:, role_permutation]).astype(
            np.float32
        )
        bridge_profile = np.hstack(
            [left_profile[:, role_permutation], right_profile, bridge_delta]
        ).astype(np.float32)
        bridge_profile = bridge_profile - np.mean(bridge_profile, axis=1, keepdims=True)
        bridge_profile = _safe_normalize(bridge_profile)
        return {
            "left_role_profile": left_profile.astype(np.float32),
            "right_role_profile": right_profile.astype(np.float32),
            "bridge_profile": bridge_profile.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_docs(x))
        role_state = _role_profiles(state, 0.0)
        state.update(role_state)
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), role_state["bridge_profile"]
        )
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_queries(x))
        role_state = _role_profiles(state, 1.0)
        state.update(role_state)
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), role_state["bridge_profile"]
        )
        return state

    def _base_aux(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        if base_aux_score is None:
            return np.zeros(
                (query_state["wave_real"].shape[0], doc_state["wave_real"].shape[0]),
                dtype=np.float32,
            )
        return base_aux_score(doc_state, query_state).astype(np.float32)

    def _bipartite_signal(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base_method.score(doc_state, query_state).astype(np.float32)
        base_aux = _base_aux(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return base_aux

        doc_left = doc_state.get("left_role_profile")
        doc_right = doc_state.get("right_role_profile")
        doc_bridge = doc_state.get("bridge_profile")
        query_left = query_state.get("left_role_profile")
        query_right = query_state.get("right_role_profile")
        query_bridge = query_state.get("bridge_profile")
        if doc_left is None or doc_right is None or doc_bridge is None:
            return base_aux
        if query_left is None or query_right is None or query_bridge is None:
            return base_aux
        if doc_left.shape[0] != doc_count:
            return base_aux

        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return base_aux

        support_seed = carrier + 0.10 * np.maximum(base_aux, 0.0)
        support_idx = np.argpartition(-support_seed, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(support_seed, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        support_weights = _normalize_rows(support_weights)

        support_left = doc_left[support_idx]
        support_right = doc_right[support_idx]
        support_bridge = doc_bridge[support_idx]
        left_align = np.sum(
            query_left[:, None, :] * support_left,
            axis=2,
            dtype=np.float32,
        )
        right_align = np.sum(
            query_right[:, None, :] * support_right,
            axis=2,
            dtype=np.float32,
        )
        cross_align = np.sum(
            query_left[:, None, :] * support_right,
            axis=2,
            dtype=np.float32,
        )
        cross_align = cross_align + np.sum(
            query_right[:, None, :] * support_left,
            axis=2,
            dtype=np.float32,
        )
        support_gate = np.sqrt(
            np.clip(
                np.maximum(left_align, 0.0) * np.maximum(right_align, 0.0), 0.0, 1.0
            )
        ).astype(np.float32)
        support_gate = support_gate * (
            0.75 + 0.25 * np.sqrt(np.clip(0.5 * np.maximum(cross_align, 0.0), 0.0, 1.0))
        )
        support_gate = support_weights * (
            support_floor + (1.0 - support_floor) * support_gate
        )
        support_gate = _normalize_rows(support_gate)

        support_left_context = np.sum(
            support_gate[:, :, None] * support_left,
            axis=1,
            dtype=np.float32,
        )
        support_right_context = np.sum(
            support_gate[:, :, None] * support_right,
            axis=1,
            dtype=np.float32,
        )
        support_bridge_context = np.sum(
            support_gate[:, :, None] * support_bridge,
            axis=1,
            dtype=np.float32,
        )
        support_left_context = _safe_normalize(support_left_context)
        support_right_context = _safe_normalize(support_right_context)
        support_bridge_context = _safe_normalize(support_bridge_context)

        query_left_target = _safe_normalize(
            (1.0 - support_mix) * query_left + support_mix * support_left_context
        )
        query_right_target = _safe_normalize(
            (1.0 - support_mix) * query_right + support_mix * support_right_context
        )
        query_bridge_target = _safe_normalize(
            (1.0 - support_mix) * query_bridge + support_mix * support_bridge_context
        )

        left_cross = np.maximum(
            (query_left_target * role_weight[None, :]) @ doc_right.T,
            0.0,
        )
        right_cross = np.maximum(
            (query_right_target * role_weight[None, :]) @ doc_left.T,
            0.0,
        )
        bridge_cross = np.maximum(query_bridge_target @ doc_bridge.T, 0.0)
        bridge_cross = bridge_cross / float(max(1, query_bridge_target.shape[1]))
        dual_core = 0.5 * (left_cross + right_cross)
        bipartite = (1.0 - bridge_mix) * dual_core + bridge_mix * np.sqrt(
            np.maximum(dual_core * bridge_cross, 0.0)
        )
        bipartite = bipartite - np.mean(bipartite, axis=1, keepdims=True)
        scale = np.max(np.abs(bipartite), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        bipartite = bipartite / scale
        combined = base_aux_mix * base_aux + (1.0 - base_aux_mix) * bipartite
        combined = combined - np.mean(combined, axis=1, keepdims=True)
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return (combined / combined_scale).astype(np.float32)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return _bipartite_signal(doc_state, query_state)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base_method.score(doc_state, query_state).astype(np.float32)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier
        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier
        signal = _bipartite_signal(doc_state, query_state)
        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_signal = np.take_along_axis(signal, candidate_idx, axis=1)
        uncertainty_gate = _relative_uncertainty_gate(
            candidate_carrier, uncertainty_width
        )
        rerank_gain = np.maximum(
            0.0,
            1.0 + duality_gain * uncertainty_gate * candidate_signal,
        )
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    return EmbeddingStateMethod(
        method_name="projective_holographic_support_bipartite_state_v0",
        family="state_projective_holographic_support_bipartite",
        params={
            **base_method.params,
            "role_count": float(role_count),
            "role_top_k": float(role_top_k),
            "role_temperature": role_temperature,
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "support_floor": support_floor,
            "wave_mix": wave_mix,
            "boundary_mix": boundary_mix,
            "fringe_mix": fringe_mix,
            "support_mix": support_mix,
            "bridge_mix": bridge_mix,
            "base_aux_mix": base_aux_mix,
            "profile_mix": profile_mix,
            "duality_gain": duality_gain,
            "uncertainty_width": uncertainty_width,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _projective_holographic_noncommutative_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    role_count = max(8, min(48, int(params.get("role_count", 18))))
    role_top_k = min(role_count, max(1, int(params.get("role_top_k", 2))))
    role_temperature = max(1e-4, float(params.get("role_temperature", 0.08)))
    support_width = max(3, int(params.get("support_width", 8)))
    support_temperature = max(1e-4, float(params.get("support_temperature", 0.12)))
    support_floor = min(max(float(params.get("support_floor", 0.18)), 0.0), 0.9)
    wave_mix = min(max(float(params.get("wave_mix", 0.22)), 0.0), 1.0)
    boundary_mix = min(max(float(params.get("boundary_mix", 0.50)), 0.0), 1.0)
    fringe_mix = min(max(float(params.get("fringe_mix", 0.16)), 0.0), 1.0)
    support_mix = min(max(float(params.get("support_mix", 0.24)), 0.0), 1.0)
    bridge_mix = min(max(float(params.get("bridge_mix", 0.28)), 0.0), 1.0)
    order_mix = min(max(float(params.get("order_mix", 0.32)), 0.0), 1.5)
    phase_mix = min(max(float(params.get("phase_mix", 0.20)), 0.0), 1.5)
    base_aux_mix = min(max(float(params.get("base_aux_mix", 0.10)), 0.0), 1.0)
    profile_mix = max(0.0, float(params.get("profile_mix", 0.24)))
    dim = int(params.get("dim", 0))
    secret_key = str(params.get("secret_key", ""))

    base_method = _projective_holographic_boundary_build(rng, params)
    base_aux_score = base_method.aux_score
    sample_state = base_method.encode_docs(np.zeros((1, dim), dtype=np.float32))
    hidden_dim = sample_state["wave_real"].shape[1]
    fringe_dim = sample_state["holographic_fringe"].shape[1]
    local_rng = np.random.default_rng(
        _method_seed(
            "projective_holographic_noncommutative_observable_v0", secret_key, dim
        )
    )

    left_wave_real = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_wave_imag = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_wave_real = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_wave_imag = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_boundary = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_boundary = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_fringe = _safe_normalize(
        local_rng.normal(size=(role_count, fringe_dim)).astype(np.float32)
    )
    right_fringe = _safe_normalize(
        local_rng.normal(size=(role_count, fringe_dim)).astype(np.float32)
    )
    role_permutation = local_rng.permutation(role_count)
    order_operator_a = _qr_orthogonal(local_rng, role_count, role_count)
    order_operator_b = _qr_orthogonal(local_rng, role_count, role_count)
    phase_operator = _qr_orthogonal(local_rng, role_count, role_count)
    order_bias = local_rng.normal(scale=0.05, size=(role_count,)).astype(np.float32)

    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        total = np.sum(x, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        return (x / total).astype(np.float32)

    def _order_profile(state: StateMap) -> np.ndarray:
        order_state = state.get("order_state")
        phase_state = state.get("phase_state")
        if order_state is None:
            return np.zeros((state["public"].shape[0], 0), dtype=np.float32)
        if phase_state is None:
            phase_state = np.zeros_like(order_state)
        commutator = np.tanh(order_state @ (order_operator_a - order_operator_b).T)
        profile = np.concatenate([order_state, phase_state, commutator], axis=1).astype(
            np.float32,
            copy=False,
        )
        profile = profile - np.mean(profile, axis=1, keepdims=True)
        return _safe_normalize(profile)

    def _merge_aux_profile(
        base_profile: np.ndarray | None, state: StateMap
    ) -> np.ndarray:
        order_profile = _order_profile(state)
        if base_profile is None:
            return order_profile.astype(np.float32, copy=False)
        if base_profile.shape[0] != order_profile.shape[0]:
            return base_profile.astype(np.float32, copy=False)
        merged = np.concatenate(
            [
                base_profile.astype(np.float32, copy=False),
                profile_mix * order_profile.astype(np.float32, copy=False),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _role_profiles(state: StateMap, recover: float) -> Dict[str, np.ndarray]:
        left_wave = np.maximum(
            state["wave_real"] @ left_wave_real.T
            + state["wave_imag"] @ left_wave_imag.T,
            0.0,
        ).astype(np.float32) / float(max(1, hidden_dim))
        right_wave = np.maximum(
            state["wave_real"] @ right_wave_real.T
            + state["wave_imag"] @ right_wave_imag.T,
            0.0,
        ).astype(np.float32) / float(max(1, hidden_dim))
        left_boundary_match = np.clip(
            state["holographic_boundary"] @ left_boundary.T, 0.0, 1.0
        ).astype(np.float32)
        right_boundary_match = np.clip(
            state["holographic_boundary"] @ right_boundary.T, 0.0, 1.0
        ).astype(np.float32)
        left_fringe_match = np.clip(
            state["holographic_fringe"] @ left_fringe.T, 0.0, 1.0
        ).astype(np.float32)
        right_fringe_match = np.clip(
            state["holographic_fringe"] @ right_fringe.T, 0.0, 1.0
        ).astype(np.float32)

        left_logits = wave_mix * left_wave + boundary_mix * left_boundary_match
        left_logits = left_logits + fringe_mix * left_fringe_match
        right_logits = wave_mix * right_wave + boundary_mix * right_boundary_match
        right_logits = right_logits + fringe_mix * right_fringe_match
        if recover > 0.0:
            left_logits = left_logits + recover * (
                left_boundary_match
                - np.mean(left_boundary_match, axis=1, keepdims=True)
            )
            right_logits = right_logits + recover * (
                right_boundary_match
                - np.mean(right_boundary_match, axis=1, keepdims=True)
            )

        local_temperature = max(1e-4, role_temperature * (1.0 - 0.20 * recover))
        left_profile = _topk_soft_assign(left_logits, role_top_k, local_temperature)
        right_profile = _topk_soft_assign(right_logits, role_top_k, local_temperature)
        left_profile = _normalize_rows(left_profile)
        right_profile = _normalize_rows(right_profile)
        bridge_delta = np.abs(left_profile - right_profile[:, role_permutation]).astype(
            np.float32
        )
        bridge_profile = np.hstack(
            [left_profile[:, role_permutation], right_profile, bridge_delta]
        ).astype(np.float32)
        bridge_profile = bridge_profile - np.mean(bridge_profile, axis=1, keepdims=True)
        bridge_profile = _safe_normalize(bridge_profile)

        bridge_left = bridge_profile[:, :role_count]
        bridge_right = bridge_profile[:, role_count : 2 * role_count]
        bridge_tail = bridge_profile[:, 2 * role_count :]
        order_state = 0.5 * (left_profile + right_profile[:, role_permutation])
        order_state = (1.0 - bridge_mix) * order_state + bridge_mix * (
            0.35 * bridge_left
            + 0.35 * bridge_right[:, role_permutation]
            + 0.30 * bridge_tail
        )
        order_state = _safe_normalize(order_state.astype(np.float32))
        phase_state = np.tanh(
            left_profile
            - right_profile[:, role_permutation]
            + 0.5 * (bridge_left - bridge_right[:, role_permutation])
        ).astype(np.float32)
        phase_state = _safe_normalize(phase_state)
        return {
            "left_role_profile": left_profile.astype(np.float32),
            "right_role_profile": right_profile.astype(np.float32),
            "bridge_profile": bridge_profile.astype(np.float32),
            "order_state": order_state.astype(np.float32),
            "phase_state": phase_state.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_docs(x))
        role_state = _role_profiles(state, 0.0)
        state.update(role_state)
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), state
        )
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_queries(x))
        role_state = _role_profiles(state, 1.0)
        state.update(role_state)
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), state
        )
        return state

    def _base_aux(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        if base_aux_score is None:
            return np.zeros(
                (query_state["wave_real"].shape[0], doc_state["wave_real"].shape[0]),
                dtype=np.float32,
            )
        return base_aux_score(doc_state, query_state).astype(np.float32)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base_method.score(doc_state, query_state).astype(np.float32)
        base_aux = _base_aux(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return base_aux

        doc_left = doc_state.get("left_role_profile")
        doc_right = doc_state.get("right_role_profile")
        doc_bridge = doc_state.get("bridge_profile")
        doc_order = doc_state.get("order_state")
        doc_phase = doc_state.get("phase_state")
        query_left = query_state.get("left_role_profile")
        query_right = query_state.get("right_role_profile")
        query_bridge = query_state.get("bridge_profile")
        query_order = query_state.get("order_state")
        query_phase = query_state.get("phase_state")
        if doc_left is None or doc_right is None or doc_bridge is None:
            return base_aux
        if doc_order is None or doc_phase is None:
            return base_aux
        if query_left is None or query_right is None or query_bridge is None:
            return base_aux
        if query_order is None or query_phase is None:
            return base_aux
        if doc_left.shape[0] != doc_count:
            return base_aux

        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return base_aux

        support_seed = carrier + 0.10 * np.maximum(base_aux, 0.0)
        support_idx = np.argpartition(-support_seed, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(support_seed, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        support_weights = _normalize_rows(support_weights)

        support_left = doc_left[support_idx]
        support_right = doc_right[support_idx]
        support_bridge = doc_bridge[support_idx]
        support_order = doc_order[support_idx]
        support_phase = doc_phase[support_idx]

        left_align = np.sum(
            query_left[:, None, :] * support_left,
            axis=2,
            dtype=np.float32,
        )
        right_align = np.sum(
            query_right[:, None, :] * support_right,
            axis=2,
            dtype=np.float32,
        )
        cross_align = np.sum(
            query_left[:, None, :] * support_right,
            axis=2,
            dtype=np.float32,
        )
        cross_align = cross_align + np.sum(
            query_right[:, None, :] * support_left,
            axis=2,
            dtype=np.float32,
        )
        support_gate = np.sqrt(
            np.clip(
                np.maximum(left_align, 0.0) * np.maximum(right_align, 0.0), 0.0, 1.0
            )
        ).astype(np.float32)
        support_gate = support_gate * (
            0.75 + 0.25 * np.sqrt(np.clip(0.5 * np.maximum(cross_align, 0.0), 0.0, 1.0))
        )
        support_gate = support_weights * (
            support_floor + (1.0 - support_floor) * support_gate
        )
        support_gate = _normalize_rows(support_gate)

        support_left_context = np.sum(
            support_gate[:, :, None] * support_left,
            axis=1,
            dtype=np.float32,
        )
        support_right_context = np.sum(
            support_gate[:, :, None] * support_right,
            axis=1,
            dtype=np.float32,
        )
        support_bridge_context = np.sum(
            support_gate[:, :, None] * support_bridge,
            axis=1,
            dtype=np.float32,
        )
        support_order_context = np.sum(
            support_gate[:, :, None] * support_order,
            axis=1,
            dtype=np.float32,
        )
        support_phase_context = np.sum(
            support_gate[:, :, None] * support_phase,
            axis=1,
            dtype=np.float32,
        )
        support_left_context = _safe_normalize(support_left_context)
        support_right_context = _safe_normalize(support_right_context)
        support_bridge_context = _safe_normalize(support_bridge_context)
        support_order_context = _safe_normalize(support_order_context)
        support_phase_context = _safe_normalize(support_phase_context)

        query_left_target = _safe_normalize(
            (1.0 - support_mix) * query_left + support_mix * support_left_context
        )
        query_right_target = _safe_normalize(
            (1.0 - support_mix) * query_right + support_mix * support_right_context
        )
        query_bridge_target = _safe_normalize(
            (1.0 - support_mix) * query_bridge + support_mix * support_bridge_context
        )
        query_order_target = _safe_normalize(
            (1.0 - support_mix) * query_order + support_mix * support_order_context
        )
        query_phase_target = _safe_normalize(
            (1.0 - support_mix) * query_phase + support_mix * support_phase_context
        )

        bridge_left = query_bridge_target[:, :role_count]
        bridge_right = query_bridge_target[:, role_count : 2 * role_count]
        bridge_tail = query_bridge_target[:, 2 * role_count :]
        order_logits = order_mix * (
            query_left_target - query_right_target[:, role_permutation]
        )
        order_logits = order_logits + 0.5 * bridge_mix * (
            bridge_tail + bridge_left - bridge_right[:, role_permutation]
        )
        order_logits = order_logits + phase_mix * query_phase_target
        order_logits = order_logits + order_bias[None, :]
        order_idx = np.argsort(order_logits, axis=1)

        signal = np.zeros_like(carrier, dtype=np.float32)
        # Die geheime Operator-Reihenfolge ist hier selbst der Kollaps:
        # gleicher Carrier, aber query-spezifische Rollenordnung nur im Aux-Kanal.
        for query_idx in range(carrier.shape[0]):
            order = order_idx[query_idx]
            ordered_query = query_order_target[query_idx, order]
            ordered_phase = query_phase_target[query_idx, order]
            ordered_doc = doc_order[:, order]
            ordered_doc_phase = doc_phase[:, order]
            query_drive = ordered_query[None, :]
            forward_stage = np.tanh(
                ordered_doc @ order_operator_a.T + order_mix * query_drive
            ).astype(np.float32)
            forward_stage = np.tanh(
                forward_stage @ order_operator_b.T
                + phase_mix * ordered_doc_phase
                + 0.5 * ordered_phase[None, :]
            ).astype(np.float32)
            backward_stage = np.tanh(
                ordered_doc @ order_operator_b.T + order_mix * query_drive
            ).astype(np.float32)
            backward_stage = np.tanh(
                backward_stage @ order_operator_a.T
                + phase_mix * ordered_doc_phase[:, ::-1]
                + 0.5 * ordered_phase[::-1][None, :]
            ).astype(np.float32)
            commutator = np.sum(
                (forward_stage - backward_stage)
                * (0.6 * ordered_phase[None, :] + 0.4 * ordered_doc_phase),
                axis=1,
            )
            phase_term = np.sum(
                (forward_stage @ phase_operator.T) * backward_stage,
                axis=1,
            )
            row = commutator + phase_mix * phase_term
            signal[query_idx] = row.astype(np.float32, copy=False)

        signal = signal - np.mean(signal, axis=1, keepdims=True)
        scale = np.max(np.abs(signal), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        signal = signal / scale
        combined = base_aux_mix * base_aux + (1.0 - base_aux_mix) * signal
        combined = combined - np.mean(combined, axis=1, keepdims=True)
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return (combined / combined_scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_holographic_noncommutative_observable_v0",
        family="state_projective_holographic_noncommutative_observable",
        params={
            **base_method.params,
            "role_count": float(role_count),
            "role_top_k": float(role_top_k),
            "role_temperature": role_temperature,
            "support_width": float(support_width),
            "support_temperature": support_temperature,
            "support_floor": support_floor,
            "wave_mix": wave_mix,
            "boundary_mix": boundary_mix,
            "fringe_mix": fringe_mix,
            "support_mix": support_mix,
            "bridge_mix": bridge_mix,
            "order_mix": order_mix,
            "phase_mix": phase_mix,
            "base_aux_mix": base_aux_mix,
            "profile_mix": profile_mix,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base_method.score,
        aux_score=_aux_score,
    )


def _projective_holographic_codebook_order_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    role_count = max(8, min(48, int(params.get("role_count", 18))))
    role_top_k = min(role_count, max(1, int(params.get("role_top_k", 2))))
    role_temperature = max(1e-4, float(params.get("role_temperature", 0.08)))
    support_width = max(3, int(params.get("support_width", 8)))
    support_temperature = max(1e-4, float(params.get("support_temperature", 0.12)))
    support_floor = min(max(float(params.get("support_floor", 0.18)), 0.0), 0.9)
    wave_mix = min(max(float(params.get("wave_mix", 0.22)), 0.0), 1.0)
    boundary_mix = min(max(float(params.get("boundary_mix", 0.50)), 0.0), 1.0)
    fringe_mix = min(max(float(params.get("fringe_mix", 0.16)), 0.0), 1.0)
    support_mix = min(max(float(params.get("support_mix", 0.24)), 0.0), 1.0)
    bridge_mix = min(max(float(params.get("bridge_mix", 0.28)), 0.0), 1.0)
    order_mix = min(max(float(params.get("order_mix", 0.32)), 0.0), 1.5)
    phase_mix = min(max(float(params.get("phase_mix", 0.20)), 0.0), 1.5)
    base_aux_mix = min(max(float(params.get("base_aux_mix", 0.10)), 0.0), 1.0)
    profile_mix = max(0.0, float(params.get("profile_mix", 0.24)))
    modes = max(6, min(24, int(params.get("modes", 12))))
    code_top_k = min(modes, max(1, int(params.get("code_top_k", 3))))
    code_temperature = max(1e-4, float(params.get("code_temperature", 0.10)))
    code_mix = min(max(float(params.get("code_mix", 0.42)), 0.0), 1.5)
    bridge_code_mix = min(max(float(params.get("bridge_code_mix", 0.18)), 0.0), 1.5)
    collapse_floor = min(max(float(params.get("collapse_floor", 0.14)), 0.0), 0.8)
    dim = int(params.get("dim", 0))
    secret_key = str(params.get("secret_key", ""))

    base_method = _projective_holographic_boundary_build(rng, params)
    base_aux_score = base_method.aux_score
    sample_state = base_method.encode_docs(np.zeros((1, dim), dtype=np.float32))
    hidden_dim = sample_state["wave_real"].shape[1]
    fringe_dim = sample_state["holographic_fringe"].shape[1]
    local_rng = np.random.default_rng(
        _method_seed(
            "projective_holographic_codebook_order_observable_v0", secret_key, dim
        )
    )

    left_wave_real = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_wave_imag = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_wave_real = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_wave_imag = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_boundary = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    right_boundary = _safe_normalize(
        local_rng.normal(size=(role_count, hidden_dim)).astype(np.float32)
    )
    left_fringe = _safe_normalize(
        local_rng.normal(size=(role_count, fringe_dim)).astype(np.float32)
    )
    right_fringe = _safe_normalize(
        local_rng.normal(size=(role_count, fringe_dim)).astype(np.float32)
    )
    role_permutation = local_rng.permutation(role_count)
    order_operator_a = _qr_orthogonal(local_rng, role_count, role_count)
    order_operator_b = _qr_orthogonal(local_rng, role_count, role_count)
    phase_operator = _qr_orthogonal(local_rng, role_count, role_count)
    order_bias = local_rng.normal(scale=0.05, size=(role_count,)).astype(np.float32)

    code_order = (
        np.abs(local_rng.normal(size=(modes, role_count)).astype(np.float32)) + 0.05
    )
    code_order = code_order / np.maximum(
        1e-6, np.sum(code_order, axis=1, keepdims=True)
    )
    code_phase = (
        np.abs(local_rng.normal(size=(modes, role_count)).astype(np.float32)) + 0.05
    )
    code_phase = code_phase / np.maximum(
        1e-6, np.sum(code_phase, axis=1, keepdims=True)
    )
    code_bridge = (
        np.abs(local_rng.normal(size=(modes, 3 * role_count)).astype(np.float32)) + 0.05
    )
    code_bridge = code_bridge / np.maximum(
        1e-6, np.sum(code_bridge, axis=1, keepdims=True)
    )
    code_bias = local_rng.normal(scale=0.05, size=(modes,)).astype(np.float32)
    semantic_centers = np.eye(modes, dtype=np.float32)
    semantic_code_config = np.array(
        [float(code_top_k), code_temperature], dtype=np.float32
    )

    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        total = np.sum(x, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        return (x / total).astype(np.float32)

    def _order_profile(state: StateMap) -> np.ndarray:
        order_state = state.get("order_state")
        phase_state = state.get("phase_state")
        if order_state is None:
            return np.zeros((state["public"].shape[0], 0), dtype=np.float32)
        if phase_state is None:
            phase_state = np.zeros_like(order_state)
        commutator = np.tanh(order_state @ (order_operator_a - order_operator_b).T)
        profile = np.concatenate([order_state, phase_state, commutator], axis=1).astype(
            np.float32,
            copy=False,
        )
        profile = profile - np.mean(profile, axis=1, keepdims=True)
        return _safe_normalize(profile)

    def _collapse_profile(state: StateMap, recover: float) -> Dict[str, np.ndarray]:
        order_state = state.get("order_state")
        phase_state = state.get("phase_state")
        bridge_profile = state.get("bridge_profile")
        if order_state is None or phase_state is None or bridge_profile is None:
            rows = state["public"].shape[0]
            zero_modes = np.zeros((rows, modes), dtype=np.float32)
            zero_profile = np.zeros((rows, 3 * modes), dtype=np.float32)
            return {
                "energy": zero_modes,
                "mode_weight": zero_modes,
                "mode_energy": zero_modes,
                "mode_phase_support": zero_modes,
                "semantic_codes": zero_modes,
                "semantic_centers": semantic_centers,
                "semantic_code_config": semantic_code_config,
                "collapse_profile": zero_profile,
            }

        order_match = np.maximum(order_state @ code_order.T, 0.0).astype(np.float32)
        phase_match = np.maximum(phase_state @ code_phase.T, 0.0).astype(np.float32)
        bridge_match = np.maximum(bridge_profile @ code_bridge.T, 0.0).astype(
            np.float32
        )
        logits = code_mix * order_match + phase_mix * phase_match
        logits = logits + bridge_code_mix * bridge_match + code_bias[None, :]
        if recover > 0.0:
            logits = logits + 0.30 * recover * (
                order_match - np.mean(order_match, axis=1, keepdims=True)
            )
            logits = logits + 0.15 * recover * phase_match

        local_temperature = max(1e-4, code_temperature * (1.0 - 0.25 * recover))
        mode_weight = _topk_soft_assign(logits, code_top_k, local_temperature)
        mode_weight = (
            collapse_floor / float(modes) + (1.0 - collapse_floor) * mode_weight
        )
        mode_weight = _normalize_rows(mode_weight)
        mode_energy = mode_weight * np.maximum(
            logits - np.min(logits, axis=1, keepdims=True), 0.0
        )
        collapse_profile = np.hstack([mode_weight, mode_energy, phase_match]).astype(
            np.float32
        )
        collapse_profile = collapse_profile - np.mean(
            collapse_profile, axis=1, keepdims=True
        )
        collapse_profile = _safe_normalize(collapse_profile)
        return {
            "energy": mode_weight.astype(np.float32),
            "mode_weight": mode_weight.astype(np.float32),
            "mode_energy": mode_energy.astype(np.float32),
            "mode_phase_support": phase_match.astype(np.float32),
            "semantic_codes": mode_weight.astype(np.float32),
            "semantic_centers": semantic_centers,
            "semantic_code_config": semantic_code_config,
            "collapse_profile": collapse_profile.astype(np.float32),
        }

    def _merge_aux_profile(
        base_profile: np.ndarray | None, state: StateMap
    ) -> np.ndarray:
        order_profile = _order_profile(state)
        collapse_profile = state.get("collapse_profile")
        parts: List[np.ndarray] = []
        if base_profile is not None:
            parts.append(base_profile.astype(np.float32, copy=False))
        if order_profile.shape[1] > 0:
            parts.append(profile_mix * order_profile.astype(np.float32, copy=False))
        if collapse_profile is not None:
            parts.append(profile_mix * collapse_profile.astype(np.float32, copy=False))
        if not parts:
            return np.zeros((state["public"].shape[0], 0), dtype=np.float32)
        merged = np.concatenate(parts, axis=1).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _role_profiles(state: StateMap, recover: float) -> Dict[str, np.ndarray]:
        left_wave = np.maximum(
            state["wave_real"] @ left_wave_real.T
            + state["wave_imag"] @ left_wave_imag.T,
            0.0,
        ).astype(np.float32) / float(max(1, hidden_dim))
        right_wave = np.maximum(
            state["wave_real"] @ right_wave_real.T
            + state["wave_imag"] @ right_wave_imag.T,
            0.0,
        ).astype(np.float32) / float(max(1, hidden_dim))
        left_boundary_match = np.clip(
            state["holographic_boundary"] @ left_boundary.T, 0.0, 1.0
        ).astype(np.float32)
        right_boundary_match = np.clip(
            state["holographic_boundary"] @ right_boundary.T, 0.0, 1.0
        ).astype(np.float32)
        left_fringe_match = np.clip(
            state["holographic_fringe"] @ left_fringe.T, 0.0, 1.0
        ).astype(np.float32)
        right_fringe_match = np.clip(
            state["holographic_fringe"] @ right_fringe.T, 0.0, 1.0
        ).astype(np.float32)

        left_logits = wave_mix * left_wave + boundary_mix * left_boundary_match
        left_logits = left_logits + fringe_mix * left_fringe_match
        right_logits = wave_mix * right_wave + boundary_mix * right_boundary_match
        right_logits = right_logits + fringe_mix * right_fringe_match
        if recover > 0.0:
            left_logits = left_logits + recover * (
                left_boundary_match
                - np.mean(left_boundary_match, axis=1, keepdims=True)
            )
            right_logits = right_logits + recover * (
                right_boundary_match
                - np.mean(right_boundary_match, axis=1, keepdims=True)
            )

        local_temperature = max(1e-4, role_temperature * (1.0 - 0.20 * recover))
        left_profile = _topk_soft_assign(left_logits, role_top_k, local_temperature)
        right_profile = _topk_soft_assign(right_logits, role_top_k, local_temperature)
        left_profile = _normalize_rows(left_profile)
        right_profile = _normalize_rows(right_profile)
        bridge_delta = np.abs(left_profile - right_profile[:, role_permutation]).astype(
            np.float32
        )
        bridge_profile = np.hstack(
            [left_profile[:, role_permutation], right_profile, bridge_delta]
        ).astype(np.float32)
        bridge_profile = bridge_profile - np.mean(bridge_profile, axis=1, keepdims=True)
        bridge_profile = _safe_normalize(bridge_profile)

        bridge_left = bridge_profile[:, :role_count]
        bridge_right = bridge_profile[:, role_count : 2 * role_count]
        bridge_tail = bridge_profile[:, 2 * role_count :]
        order_state = 0.5 * (left_profile + right_profile[:, role_permutation])
        order_state = (1.0 - bridge_mix) * order_state + bridge_mix * (
            0.35 * bridge_left
            + 0.35 * bridge_right[:, role_permutation]
            + 0.30 * bridge_tail
        )
        order_state = _safe_normalize(order_state.astype(np.float32))
        phase_state = np.tanh(
            left_profile
            - right_profile[:, role_permutation]
            + 0.5 * (bridge_left - bridge_right[:, role_permutation])
        ).astype(np.float32)
        phase_state = _safe_normalize(phase_state)
        return {
            "left_role_profile": left_profile.astype(np.float32),
            "right_role_profile": right_profile.astype(np.float32),
            "bridge_profile": bridge_profile.astype(np.float32),
            "order_state": order_state.astype(np.float32),
            "phase_state": phase_state.astype(np.float32),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_docs(x))
        state.update(_role_profiles(state, 0.0))
        state.update(_collapse_profile(state, 0.0))
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), state
        )
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_queries(x))
        state.update(_role_profiles(state, 1.0))
        state.update(_collapse_profile(state, 1.0))
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), state
        )
        return state

    def _base_aux(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        if base_aux_score is None:
            return np.zeros(
                (query_state["wave_real"].shape[0], doc_state["wave_real"].shape[0]),
                dtype=np.float32,
            )
        return base_aux_score(doc_state, query_state).astype(np.float32)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base_method.score(doc_state, query_state).astype(np.float32)
        base_aux = _base_aux(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return base_aux

        doc_left = doc_state.get("left_role_profile")
        doc_right = doc_state.get("right_role_profile")
        doc_bridge = doc_state.get("bridge_profile")
        doc_order = doc_state.get("order_state")
        doc_phase = doc_state.get("phase_state")
        doc_mode = doc_state.get("mode_weight")
        doc_mode_energy = doc_state.get("mode_energy")
        doc_mode_phase = doc_state.get("mode_phase_support")
        query_left = query_state.get("left_role_profile")
        query_right = query_state.get("right_role_profile")
        query_bridge = query_state.get("bridge_profile")
        query_order = query_state.get("order_state")
        query_phase = query_state.get("phase_state")
        query_mode = query_state.get("mode_weight")
        query_mode_energy = query_state.get("mode_energy")
        query_mode_phase = query_state.get("mode_phase_support")
        if doc_left is None or doc_right is None or doc_bridge is None:
            return base_aux
        if doc_order is None or doc_phase is None:
            return base_aux
        if doc_mode is None or doc_mode_energy is None or doc_mode_phase is None:
            return base_aux
        if query_left is None or query_right is None or query_bridge is None:
            return base_aux
        if query_order is None or query_phase is None:
            return base_aux
        if query_mode is None or query_mode_energy is None or query_mode_phase is None:
            return base_aux
        if doc_left.shape[0] != doc_count:
            return base_aux

        mode_seed = np.sqrt(np.maximum(query_mode @ doc_mode.T, 0.0)).astype(np.float32)
        support_seed = carrier + 0.10 * np.maximum(base_aux, 0.0) + 0.08 * mode_seed
        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return base_aux

        support_idx = np.argpartition(-support_seed, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(support_seed, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        support_weights = _normalize_rows(support_weights)

        support_left = doc_left[support_idx]
        support_right = doc_right[support_idx]
        support_bridge = doc_bridge[support_idx]
        support_order = doc_order[support_idx]
        support_phase = doc_phase[support_idx]
        support_mode = doc_mode[support_idx]
        support_mode_energy = doc_mode_energy[support_idx]
        support_mode_phase = doc_mode_phase[support_idx]

        left_align = np.sum(
            query_left[:, None, :] * support_left,
            axis=2,
            dtype=np.float32,
        )
        right_align = np.sum(
            query_right[:, None, :] * support_right,
            axis=2,
            dtype=np.float32,
        )
        mode_align = np.sum(
            query_mode[:, None, :] * support_mode,
            axis=2,
            dtype=np.float32,
        )
        support_gate = np.sqrt(
            np.clip(
                np.maximum(left_align, 0.0)
                * np.maximum(right_align, 0.0)
                * np.maximum(mode_align, 0.0),
                0.0,
                1.0,
            )
        ).astype(np.float32)
        support_gate = support_weights * (
            support_floor + (1.0 - support_floor) * support_gate
        )
        support_gate = _normalize_rows(support_gate)

        support_left_context = np.sum(
            support_gate[:, :, None] * support_left,
            axis=1,
            dtype=np.float32,
        )
        support_right_context = np.sum(
            support_gate[:, :, None] * support_right,
            axis=1,
            dtype=np.float32,
        )
        support_bridge_context = np.sum(
            support_gate[:, :, None] * support_bridge,
            axis=1,
            dtype=np.float32,
        )
        support_order_context = np.sum(
            support_gate[:, :, None] * support_order,
            axis=1,
            dtype=np.float32,
        )
        support_phase_context = np.sum(
            support_gate[:, :, None] * support_phase,
            axis=1,
            dtype=np.float32,
        )
        support_mode_context = np.sum(
            support_gate[:, :, None] * support_mode,
            axis=1,
            dtype=np.float32,
        )
        support_mode_energy_context = np.sum(
            support_gate[:, :, None] * support_mode_energy,
            axis=1,
            dtype=np.float32,
        )
        support_mode_phase_context = np.sum(
            support_gate[:, :, None] * support_mode_phase,
            axis=1,
            dtype=np.float32,
        )
        support_left_context = _safe_normalize(support_left_context)
        support_right_context = _safe_normalize(support_right_context)
        support_bridge_context = _safe_normalize(support_bridge_context)
        support_order_context = _safe_normalize(support_order_context)
        support_phase_context = _safe_normalize(support_phase_context)
        support_mode_context = _safe_normalize(support_mode_context)
        support_mode_energy_context = _safe_normalize(support_mode_energy_context)
        support_mode_phase_context = _safe_normalize(support_mode_phase_context)

        query_left_target = _safe_normalize(
            (1.0 - support_mix) * query_left + support_mix * support_left_context
        )
        query_right_target = _safe_normalize(
            (1.0 - support_mix) * query_right + support_mix * support_right_context
        )
        query_bridge_target = _safe_normalize(
            (1.0 - support_mix) * query_bridge + support_mix * support_bridge_context
        )
        query_order_target = _safe_normalize(
            (1.0 - support_mix) * query_order + support_mix * support_order_context
        )
        query_phase_target = _safe_normalize(
            (1.0 - support_mix) * query_phase + support_mix * support_phase_context
        )
        query_mode_target = _safe_normalize(
            (1.0 - support_mix) * query_mode + support_mix * support_mode_context
        )
        query_mode_energy_target = _safe_normalize(
            (1.0 - support_mix) * query_mode_energy
            + support_mix * support_mode_energy_context
        )
        query_mode_phase_target = _safe_normalize(
            (1.0 - support_mix) * query_mode_phase
            + support_mix * support_mode_phase_context
        )

        bridge_left = query_bridge_target[:, :role_count]
        bridge_right = query_bridge_target[:, role_count : 2 * role_count]
        bridge_tail = query_bridge_target[:, 2 * role_count :]
        order_logits = order_mix * (
            query_left_target - query_right_target[:, role_permutation]
        )
        order_logits = order_logits + 0.5 * bridge_mix * (
            bridge_tail + bridge_left - bridge_right[:, role_permutation]
        )
        order_logits = order_logits + phase_mix * query_phase_target
        order_logits = order_logits + order_bias[None, :]
        order_idx = np.argsort(order_logits, axis=1)

        signal = np.zeros_like(carrier, dtype=np.float32)
        for query_idx in range(carrier.shape[0]):
            order = order_idx[query_idx]
            ordered_query = query_order_target[query_idx, order]
            ordered_phase = query_phase_target[query_idx, order]
            ordered_doc = doc_order[:, order]
            ordered_doc_phase = doc_phase[:, order]
            query_drive = ordered_query[None, :]
            forward_stage = np.tanh(
                ordered_doc @ order_operator_a.T + order_mix * query_drive
            ).astype(np.float32)
            forward_stage = np.tanh(
                forward_stage @ order_operator_b.T
                + phase_mix * ordered_doc_phase
                + 0.5 * ordered_phase[None, :]
            ).astype(np.float32)
            backward_stage = np.tanh(
                ordered_doc @ order_operator_b.T + order_mix * query_drive
            ).astype(np.float32)
            backward_stage = np.tanh(
                backward_stage @ order_operator_a.T
                + phase_mix * ordered_doc_phase[:, ::-1]
                + 0.5 * ordered_phase[::-1][None, :]
            ).astype(np.float32)
            commutator = np.sum(
                (forward_stage - backward_stage)
                * (0.6 * ordered_phase[None, :] + 0.4 * ordered_doc_phase),
                axis=1,
            )
            phase_term = np.sum(
                (forward_stage @ phase_operator.T) * backward_stage,
                axis=1,
            )
            signal[query_idx] = (commutator + phase_mix * phase_term).astype(
                np.float32,
                copy=False,
            )

        signal = signal - np.mean(signal, axis=1, keepdims=True)
        signal_scale = np.max(np.abs(signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        signal = signal / signal_scale

        mode_gate = np.sqrt(np.maximum(query_mode_target @ doc_mode.T, 0.0)).astype(
            np.float32
        )
        energy_match = (query_mode_energy_target @ doc_mode_energy.T) / float(
            max(1, modes)
        )
        phase_match = (query_mode_phase_target @ doc_mode_phase.T) / float(
            max(1, modes)
        )
        code_signal = 0.55 * mode_gate + 0.25 * energy_match + 0.20 * phase_match
        code_signal = code_signal - np.mean(code_signal, axis=1, keepdims=True)
        code_scale = np.max(np.abs(code_signal), axis=1, keepdims=True)
        code_scale = np.where(code_scale == 0.0, 1.0, code_scale)
        code_signal = code_signal / code_scale

        gated_signal = signal * (0.20 + 0.80 * mode_gate)
        combined_signal = 0.55 * gated_signal + 0.45 * code_signal
        combined = base_aux_mix * base_aux + (1.0 - base_aux_mix) * combined_signal
        combined = combined - np.mean(combined, axis=1, keepdims=True)
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return (combined / combined_scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_holographic_codebook_order_observable_v0",
        family="state_projective_holographic_codebook_order_observable",
        params={
            **base_method.params,
            "role_count": float(role_count),
            "role_top_k": float(role_top_k),
            "role_temperature": role_temperature,
            "support_width": float(support_width),
            "support_temperature": support_temperature,
            "support_floor": support_floor,
            "wave_mix": wave_mix,
            "boundary_mix": boundary_mix,
            "fringe_mix": fringe_mix,
            "support_mix": support_mix,
            "bridge_mix": bridge_mix,
            "order_mix": order_mix,
            "phase_mix": phase_mix,
            "base_aux_mix": base_aux_mix,
            "profile_mix": profile_mix,
            "modes": float(modes),
            "code_top_k": float(code_top_k),
            "code_temperature": code_temperature,
            "code_mix": code_mix,
            "bridge_code_mix": bridge_code_mix,
            "collapse_floor": collapse_floor,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base_method.score,
        aux_score=_aux_score,
    )


def _projective_holographic_gauge_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    links = max(6, min(32, int(params.get("links", 12))))
    link_top_k = min(links, max(1, int(params.get("link_top_k", 3))))
    link_temperature = max(1e-4, float(params.get("link_temperature", 0.11)))
    link_floor = min(max(float(params.get("link_floor", 0.10)), 0.0), 0.5)
    plaquettes = max(4, min(links, int(params.get("plaquettes", max(6, links // 2)))))
    plaquette_top_k = min(plaquettes, max(1, int(params.get("plaquette_top_k", 2))))
    plaquette_temperature = max(1e-4, float(params.get("plaquette_temperature", 0.09)))
    spin_floor = min(max(float(params.get("spin_floor", 0.08)), 0.0), 0.5)
    gauge_recover_gain = min(
        max(float(params.get("gauge_recover_gain", 0.30)), 0.0), 1.0
    )
    boundary_mix = min(max(float(params.get("boundary_mix", 0.44)), 0.0), 1.5)
    fringe_mix = min(max(float(params.get("fringe_mix", 0.24)), 0.0), 1.5)
    wave_mix = min(max(float(params.get("wave_mix", 0.18)), 0.0), 1.5)
    phase_mix = min(max(float(params.get("phase_mix", 0.14)), 0.0), 1.5)
    flux_gain = max(0.05, float(params.get("flux_gain", 1.15)))
    charge_gain = max(0.05, float(params.get("charge_gain", 1.0)))
    profile_mix = min(max(float(params.get("profile_mix", 0.30)), 0.0), 1.0)
    base_aux_mix = min(max(float(params.get("base_aux_mix", 0.20)), 0.0), 1.0)
    dim = int(params.get("dim", 0))
    secret_key = str(params.get("secret_key", ""))

    base_method = _projective_holographic_boundary_build(rng, params)
    base_aux_score = base_method.aux_score
    sample_state = base_method.encode_docs(np.zeros((1, dim), dtype=np.float32))
    hidden_dim = sample_state["wave_real"].shape[1]
    fringe_dim = sample_state["holographic_fringe"].shape[1]
    local_rng = np.random.default_rng(
        _method_seed("projective_holographic_gauge_observable_v0", secret_key, dim)
    )

    link_boundary = _safe_normalize(
        local_rng.normal(size=(links, hidden_dim)).astype(np.float32)
    )
    link_fringe = _safe_normalize(
        local_rng.normal(size=(links, fringe_dim)).astype(np.float32)
    )
    link_wave_real = _safe_normalize(
        local_rng.normal(size=(links, hidden_dim)).astype(np.float32)
    )
    link_wave_imag = _safe_normalize(
        local_rng.normal(size=(links, hidden_dim)).astype(np.float32)
    )
    phase_probe = local_rng.uniform(-math.pi, math.pi, size=(links, hidden_dim)).astype(
        np.float32
    )
    phase_cos = np.cos(phase_probe).astype(np.float32)
    phase_sin = np.sin(phase_probe).astype(np.float32)
    link_bias = local_rng.uniform(-0.08, 0.08, size=(links,)).astype(np.float32)

    forward_incidence = (
        np.abs(local_rng.normal(size=(plaquettes, links)).astype(np.float32)) + 0.05
    )
    forward_incidence = forward_incidence / np.maximum(
        1e-6, np.sum(forward_incidence, axis=1, keepdims=True)
    )
    reverse_incidence = (
        np.abs(local_rng.normal(size=(plaquettes, links)).astype(np.float32)) + 0.05
    )
    reverse_incidence = reverse_incidence / np.maximum(
        1e-6, np.sum(reverse_incidence, axis=1, keepdims=True)
    )
    link_permutation = local_rng.permutation(links)
    plaquette_permutation = local_rng.permutation(plaquettes)
    plaquette_sign = local_rng.choice(
        np.array([-1.0, 1.0], dtype=np.float32), size=(plaquettes,)
    ).astype(np.float32)

    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        total = np.sum(x, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        return (x / total).astype(np.float32)

    def _merge_aux_profile(
        base_profile: np.ndarray | None, gauge_profile: np.ndarray
    ) -> np.ndarray:
        if base_profile is None:
            return _safe_normalize(gauge_profile.astype(np.float32, copy=False))
        if base_profile.shape[0] != gauge_profile.shape[0]:
            return base_profile.astype(np.float32, copy=False)
        merged = np.concatenate(
            [
                base_profile.astype(np.float32, copy=False),
                profile_mix * gauge_profile.astype(np.float32, copy=False),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _gauge_profile(state: StateMap, recover: float) -> Dict[str, np.ndarray]:
        boundary_match = np.maximum(
            state["holographic_boundary"] @ link_boundary.T, 0.0
        ).astype(np.float32)
        fringe_match = np.maximum(
            state["holographic_fringe"] @ link_fringe.T, 0.0
        ).astype(np.float32)
        wave_match = np.maximum(
            state["wave_real"] @ link_wave_real.T
            + state["wave_imag"] @ link_wave_imag.T,
            0.0,
        ).astype(np.float32) / float(max(1, hidden_dim))
        phase = np.arctan2(state["wave_imag"], state["wave_real"]).astype(np.float32)
        phase_match = np.maximum(
            np.cos(phase) @ phase_cos.T + np.sin(phase) @ phase_sin.T,
            0.0,
        ).astype(np.float32) / float(max(1, hidden_dim))

        link_logits = boundary_mix * boundary_match + fringe_mix * fringe_match
        link_logits = link_logits + wave_mix * wave_match + phase_mix * phase_match
        link_logits = link_logits + link_bias[None, :]
        if recover > 0.0:
            recover_delta = boundary_match - np.mean(
                boundary_match, axis=1, keepdims=True
            )
            link_logits = link_logits + gauge_recover_gain * recover * recover_delta

        local_link_temp = max(
            1e-4, link_temperature * (1.0 - 0.55 * gauge_recover_gain * recover)
        )
        link_code = _topk_soft_assign(link_logits, link_top_k, local_link_temp)
        link_code = link_floor / float(links) + (1.0 - link_floor) * link_code
        link_code = _normalize_rows(link_code)
        focus_top_k = max(1, min(links, link_top_k - 1 if link_top_k > 1 else 1))
        focus_scale = 1.0 + recover * (1.0 + gauge_recover_gain)
        link_focus = _topk_soft_assign(
            link_logits * focus_scale,
            focus_top_k,
            max(1e-4, local_link_temp * (1.0 - 0.35 * recover)),
        )
        link_focus = link_floor / float(links) + (1.0 - link_floor) * link_focus
        link_focus = _normalize_rows(link_focus)

        plaquette_forward = (link_focus @ forward_incidence.T).astype(np.float32)
        plaquette_reverse = (link_focus @ reverse_incidence.T).astype(np.float32)
        plaquette_mass = 0.5 * (plaquette_forward + plaquette_reverse)
        plaquette_flux = np.tanh(
            flux_gain * (plaquette_forward - plaquette_reverse)
        ).astype(np.float32)
        gauge_charge = np.tanh(
            charge_gain * (plaquette_mass - np.roll(plaquette_mass, -1, axis=1))
        ).astype(np.float32)

        spin_logits = np.abs(plaquette_flux) + 0.35 * plaquette_mass
        spin_logits = spin_logits + 0.65 * np.maximum(gauge_charge, 0.0)
        if recover > 0.0:
            spin_logits = spin_logits + recover * np.maximum(plaquette_flux, 0.0)

        local_spin_temp = max(
            1e-4,
            plaquette_temperature * (1.0 - 0.45 * gauge_recover_gain * recover),
        )
        spin_code = _topk_soft_assign(spin_logits, plaquette_top_k, local_spin_temp)
        spin_code = spin_floor / float(plaquettes) + (1.0 - spin_floor) * spin_code
        spin_code = _normalize_rows(spin_code)
        spin_focus_top_k = max(
            1, min(plaquettes, plaquette_top_k - 1 if plaquette_top_k > 1 else 1)
        )
        spin_focus = _topk_soft_assign(
            spin_logits * focus_scale,
            spin_focus_top_k,
            max(1e-4, local_spin_temp * (1.0 - 0.30 * recover)),
        )
        spin_focus = spin_floor / float(plaquettes) + (1.0 - spin_floor) * spin_focus
        spin_focus = _normalize_rows(spin_focus)

        ordered_links = link_code[:, link_permutation].astype(np.float32)
        ordered_link_focus = link_focus[:, link_permutation].astype(np.float32)
        link_cascade = np.cumsum(ordered_links, axis=1).astype(np.float32)
        link_focus_cascade = np.cumsum(ordered_link_focus, axis=1).astype(np.float32)
        ordered_spin = spin_code[:, plaquette_permutation].astype(np.float32)
        ordered_spin_focus = spin_focus[:, plaquette_permutation].astype(np.float32)
        spin_cascade = np.cumsum(ordered_spin, axis=1).astype(np.float32)
        spin_focus_cascade = np.cumsum(ordered_spin_focus, axis=1).astype(np.float32)
        signed_flux = (
            plaquette_flux[:, plaquette_permutation] * plaquette_sign[None, :]
        ).astype(np.float32)
        signed_charge = (
            gauge_charge[:, plaquette_permutation] * plaquette_sign[None, :]
        ).astype(np.float32)

        gauge_profile = np.hstack(
            [
                ordered_links,
                link_cascade,
                ordered_spin,
                spin_cascade,
                signed_flux,
                signed_charge,
                plaquette_mass[:, plaquette_permutation],
            ]
        ).astype(np.float32)
        return {
            "gauge_link_code": link_code.astype(np.float32),
            "gauge_link_focus": link_focus.astype(np.float32),
            "gauge_link_cascade": link_cascade,
            "gauge_link_focus_cascade": link_focus_cascade,
            "gauge_spin_code": spin_code.astype(np.float32),
            "gauge_spin_focus": spin_focus.astype(np.float32),
            "gauge_spin_cascade": spin_cascade,
            "gauge_spin_focus_cascade": spin_focus_cascade,
            "gauge_flux": plaquette_flux.astype(np.float32),
            "gauge_charge": gauge_charge.astype(np.float32),
            "gauge_mass": plaquette_mass.astype(np.float32),
            "gauge_profile": _safe_normalize(gauge_profile),
        }

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_docs(x))
        gauge_state = _gauge_profile(state, 0.0)
        state.update(gauge_state)
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), gauge_state["gauge_profile"]
        )
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_queries(x))
        gauge_state = _gauge_profile(state, 1.0)
        state.update(gauge_state)
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), gauge_state["gauge_profile"]
        )
        return state

    def _base_aux(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        if base_aux_score is None:
            return np.zeros(
                (query_state["wave_real"].shape[0], doc_state["wave_real"].shape[0]),
                dtype=np.float32,
            )
        return base_aux_score(doc_state, query_state).astype(np.float32)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_aux = _base_aux(doc_state, query_state)
        doc_links = doc_state.get("gauge_link_code")
        query_links = query_state.get("gauge_link_code")
        doc_link_focus = doc_state.get("gauge_link_focus")
        query_link_focus = query_state.get("gauge_link_focus")
        doc_link_cascade = doc_state.get("gauge_link_cascade")
        query_link_cascade = query_state.get("gauge_link_focus_cascade")
        doc_spins = doc_state.get("gauge_spin_code")
        query_spins = query_state.get("gauge_spin_code")
        doc_spin_focus = doc_state.get("gauge_spin_focus")
        query_spin_focus = query_state.get("gauge_spin_focus")
        doc_spin_cascade = doc_state.get("gauge_spin_cascade")
        query_spin_cascade = query_state.get("gauge_spin_focus_cascade")
        doc_flux = doc_state.get("gauge_flux")
        query_flux = query_state.get("gauge_flux")
        doc_charge = doc_state.get("gauge_charge")
        query_charge = query_state.get("gauge_charge")
        doc_mass = doc_state.get("gauge_mass")
        query_mass = query_state.get("gauge_mass")
        if doc_links is None or query_links is None:
            return base_aux
        if doc_spins is None or query_spins is None:
            return base_aux
        if doc_link_focus is None or query_link_focus is None:
            return base_aux
        if doc_link_cascade is None or query_link_cascade is None:
            return base_aux
        if doc_spin_focus is None or query_spin_focus is None:
            return base_aux
        if doc_spin_cascade is None or query_spin_cascade is None:
            return base_aux
        if doc_flux is None or query_flux is None:
            return base_aux
        if doc_charge is None or query_charge is None:
            return base_aux
        if doc_mass is None or query_mass is None:
            return base_aux

        link_match = query_link_focus @ doc_links.T
        spin_match = query_spin_focus @ doc_spins.T
        focus_match = query_links @ doc_link_focus.T
        spin_focus_match = query_spins @ doc_spin_focus.T
        link_cascade_match = (query_link_cascade @ doc_link_cascade.T) / float(
            max(1, links)
        )
        spin_cascade_match = (query_spin_cascade @ doc_spin_cascade.T) / float(
            max(1, plaquettes)
        )
        flux_match = (query_flux @ doc_flux.T) / float(max(1, plaquettes))
        charge_match = (query_charge @ doc_charge.T) / float(max(1, plaquettes))
        mass_match = (query_mass @ doc_mass.T) / float(max(1, plaquettes))
        gauge_scores = 0.18 * link_match + 0.16 * spin_match + 0.14 * focus_match
        gauge_scores = gauge_scores + 0.12 * spin_focus_match
        gauge_scores = gauge_scores + 0.12 * link_cascade_match
        gauge_scores = gauge_scores + 0.10 * spin_cascade_match
        gauge_scores = gauge_scores + 0.08 * flux_match + 0.05 * charge_match
        gauge_scores = gauge_scores + 0.05 * mass_match
        gauge_scores = gauge_scores * (
            0.20 + 0.80 * np.sqrt(np.maximum(link_match + focus_match, 0.0))
        )
        aux_scores = base_aux_mix * base_aux + (1.0 - base_aux_mix) * gauge_scores
        aux_scores = aux_scores - np.mean(aux_scores, axis=1, keepdims=True)
        scale = np.max(np.abs(aux_scores), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        return (aux_scores / scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_holographic_gauge_observable_v0",
        family="state_projective_holographic_gauge_observable",
        params={
            **base_method.params,
            "links": float(links),
            "link_top_k": float(link_top_k),
            "link_temperature": link_temperature,
            "link_floor": link_floor,
            "plaquettes": float(plaquettes),
            "plaquette_top_k": float(plaquette_top_k),
            "plaquette_temperature": plaquette_temperature,
            "spin_floor": spin_floor,
            "gauge_recover_gain": gauge_recover_gain,
            "boundary_mix": boundary_mix,
            "fringe_mix": fringe_mix,
            "wave_mix": wave_mix,
            "phase_mix": phase_mix,
            "flux_gain": flux_gain,
            "charge_gain": charge_gain,
            "profile_mix": profile_mix,
            "base_aux_mix": base_aux_mix,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base_method.score,
        aux_score=_aux_score,
    )


def _projective_holographic_coreasonance_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    slots = max(4, min(24, int(params.get("slots", 10))))
    slot_top_k = min(slots, max(1, int(params.get("slot_top_k", 2))))
    slot_temperature = max(1e-4, float(params.get("slot_temperature", 0.16)))
    support_width = max(2, int(params.get("support_width", 8)))
    support_temperature = max(1e-4, float(params.get("support_temperature", 0.12)))
    support_floor = min(max(float(params.get("support_floor", 0.18)), 0.0), 0.85)
    wave_mix = min(max(float(params.get("wave_mix", 0.34)), 0.0), 1.0)
    boundary_mix = min(max(float(params.get("boundary_mix", 0.28)), 0.0), 1.0)
    fringe_mix = min(max(float(params.get("fringe_mix", 0.18)), 0.0), 1.0)
    code_mix = min(max(float(params.get("code_mix", 0.20)), 0.0), 1.0)
    aux_mix = min(max(float(params.get("aux_mix", 0.32)), 0.0), 1.0)
    profile_mix = float(params.get("profile_mix", 0.28))
    dim = int(params.get("dim", 0))
    secret_key = str(params.get("secret_key", ""))

    base_method = _projective_holographic_boundary_build(rng, params)
    base_aux_score = base_method.aux_score
    sample_state = base_method.encode_docs(np.zeros((1, dim), dtype=np.float32))
    hidden_dim = sample_state["wave_real"].shape[1]
    fringe_dim = sample_state["holographic_fringe"].shape[1]
    local_rng = np.random.default_rng(
        _method_seed(
            "projective_holographic_coreasonance_observable_v0", secret_key, dim
        )
    )

    slot_real = _safe_normalize(
        local_rng.normal(size=(slots, hidden_dim)).astype(np.float32)
    )
    slot_imag = _safe_normalize(
        local_rng.normal(size=(slots, hidden_dim)).astype(np.float32)
    )
    slot_boundary = _safe_normalize(
        local_rng.normal(size=(slots, hidden_dim)).astype(np.float32)
    )
    slot_fringe = _safe_normalize(
        local_rng.normal(size=(slots, fringe_dim)).astype(np.float32)
    )

    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        total = np.sum(x, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        return (x / total).astype(np.float32)

    def _merge_aux_profile(
        base_profile: np.ndarray | None, resonance_codes: np.ndarray
    ) -> np.ndarray:
        if base_profile is None:
            return _safe_normalize(resonance_codes.astype(np.float32, copy=False))
        if base_profile.shape[0] != resonance_codes.shape[0]:
            return base_profile.astype(np.float32, copy=False)
        merged = np.concatenate(
            [
                base_profile.astype(np.float32, copy=False),
                profile_mix * resonance_codes.astype(np.float32, copy=False),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _resonance_codes(state: StateMap) -> np.ndarray:
        wave_match = state["wave_real"] @ slot_real.T
        wave_match = wave_match + state["wave_imag"] @ slot_imag.T
        wave_match = np.clip(wave_match**2, 0.0, 1.0).astype(np.float32)
        boundary_match = np.clip(
            state["holographic_boundary"] @ slot_boundary.T, 0.0, 1.0
        ).astype(np.float32)
        fringe_match = np.clip(
            state["holographic_fringe"] @ slot_fringe.T, 0.0, 1.0
        ).astype(np.float32)
        logits = wave_mix * wave_match + boundary_mix * boundary_match
        logits = logits + fringe_mix * fringe_match
        return _topk_soft_assign(logits, slot_top_k, slot_temperature).astype(
            np.float32
        )

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_docs(x))
        resonance_codes = _resonance_codes(state)
        state["resonance_codes"] = resonance_codes
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), resonance_codes
        )
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_queries(x))
        resonance_codes = _resonance_codes(state)
        state["resonance_codes"] = resonance_codes
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), resonance_codes
        )
        return state

    def _base_aux(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        if base_aux_score is None:
            return np.zeros(
                (query_state["wave_real"].shape[0], doc_state["wave_real"].shape[0]),
                dtype=np.float32,
            )
        return base_aux_score(doc_state, query_state).astype(np.float32)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base_method.score(doc_state, query_state).astype(np.float32)
        base_aux = _base_aux(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return base_aux

        doc_codes = doc_state.get("resonance_codes")
        query_codes = query_state.get("resonance_codes")
        if doc_codes is None:
            return base_aux
        if query_codes is None:
            return base_aux
        if doc_codes.shape[0] != doc_count:
            return base_aux

        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return base_aux

        support_seed = carrier + 0.10 * np.maximum(base_aux, 0.0)
        support_idx = np.argpartition(-support_seed, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(support_seed, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        support_weights = _normalize_rows(support_weights)

        support_wave_real = doc_state["wave_real"][support_idx]
        support_wave_imag = doc_state["wave_imag"][support_idx]
        support_boundary = doc_state["holographic_boundary"][support_idx]
        support_fringe = doc_state["holographic_fringe"][support_idx]
        support_codes = doc_codes[support_idx]

        query_support_wave = np.sum(
            query_state["wave_real"][:, None, :] * support_wave_real,
            axis=2,
            dtype=np.float32,
        )
        query_support_wave = query_support_wave + np.sum(
            query_state["wave_imag"][:, None, :] * support_wave_imag,
            axis=2,
            dtype=np.float32,
        )
        query_support_wave = np.clip(query_support_wave**2, 0.0, 1.0)
        query_support_boundary = np.clip(
            np.sum(
                query_state["holographic_boundary"][:, None, :] * support_boundary,
                axis=2,
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        query_support_fringe = np.clip(
            np.sum(
                query_state["holographic_fringe"][:, None, :] * support_fringe,
                axis=2,
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        query_support_code = np.clip(
            np.sum(query_codes[:, None, :] * support_codes, axis=2, dtype=np.float32),
            0.0,
            1.0,
        )
        support_gate = (
            wave_mix * query_support_wave + boundary_mix * query_support_boundary
        )
        support_gate = support_gate + fringe_mix * query_support_fringe
        support_gate = support_gate + code_mix * query_support_code
        support_gate = support_gate / np.maximum(
            1e-6, np.max(support_gate, axis=1, keepdims=True)
        )
        support_gate = support_weights * (
            support_floor + (1.0 - support_floor) * support_gate
        )
        support_gate = _normalize_rows(support_gate)

        candidate_wave = np.einsum(
            "qsh,dh->qsd", support_wave_real, doc_state["wave_real"], dtype=np.float32
        )
        candidate_wave = candidate_wave + np.einsum(
            "qsh,dh->qsd", support_wave_imag, doc_state["wave_imag"], dtype=np.float32
        )
        candidate_wave = np.clip(candidate_wave**2, 0.0, 1.0)
        candidate_boundary = np.clip(
            np.einsum(
                "qsh,dh->qsd",
                support_boundary,
                doc_state["holographic_boundary"],
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        candidate_fringe = np.clip(
            np.einsum(
                "qsh,dh->qsd",
                support_fringe,
                doc_state["holographic_fringe"],
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        candidate_code = np.clip(
            np.einsum("qsk,dk->qsd", support_codes, doc_codes, dtype=np.float32),
            0.0,
            1.0,
        )
        support_signal = wave_mix * candidate_wave + boundary_mix * candidate_boundary
        support_signal = support_signal + fringe_mix * candidate_fringe
        support_signal = support_signal + code_mix * candidate_code
        support_signal = np.sum(
            support_gate[:, :, None] * support_signal,
            axis=1,
            dtype=np.float32,
        )

        direct_boundary = np.clip(
            query_state["holographic_boundary"] @ doc_state["holographic_boundary"].T,
            0.0,
            1.0,
        )
        direct_fringe = np.clip(
            query_state["holographic_fringe"] @ doc_state["holographic_fringe"].T,
            0.0,
            1.0,
        )
        direct_code = np.clip(query_codes @ doc_codes.T, 0.0, 1.0)
        direct_signal = wave_mix * carrier + boundary_mix * direct_boundary
        direct_signal = direct_signal + fringe_mix * direct_fringe
        direct_signal = direct_signal + code_mix * direct_code

        coherent = 0.70 * support_signal + 0.30 * direct_signal
        coherent = coherent - np.mean(coherent, axis=1, keepdims=True)
        coherent_scale = np.max(np.abs(coherent), axis=1, keepdims=True)
        coherent_scale = np.where(coherent_scale == 0.0, 1.0, coherent_scale)
        coherent = coherent / coherent_scale
        focus = np.max(support_gate, axis=1, keepdims=True)
        focus_gate = np.clip(
            (focus - support_floor) / max(1e-4, 1.0 - support_floor),
            0.0,
            1.0,
        )
        combined = base_aux + aux_mix * focus_gate * coherent
        combined = combined - np.mean(combined, axis=1, keepdims=True)
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return (combined / combined_scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_holographic_coreasonance_observable_v0",
        family="state_projective_holographic_coreasonance_observable",
        params={
            **base_method.params,
            "slots": float(slots),
            "slot_top_k": float(slot_top_k),
            "slot_temperature": slot_temperature,
            "support_width": float(support_width),
            "support_temperature": support_temperature,
            "support_floor": support_floor,
            "wave_mix": wave_mix,
            "boundary_mix": boundary_mix,
            "fringe_mix": fringe_mix,
            "code_mix": code_mix,
            "aux_mix": aux_mix,
            "profile_mix": profile_mix,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base_method.score,
        aux_score=_aux_score,
    )


def _projective_holographic_query_edge_graph_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    relation_slots = max(8, min(32, int(params.get("relation_slots", 8))))
    relation_top_k = min(relation_slots, max(1, int(params.get("relation_top_k", 1))))
    relation_temperature = max(1e-4, float(params.get("relation_temperature", 0.05)))
    support_width = max(4, int(params.get("support_width", 6)))
    rerank_width = max(support_width, int(params.get("rerank_width", 16)))
    support_temperature = max(1e-4, float(params.get("support_temperature", 0.16)))
    support_relation_mix = max(0.0, float(params.get("support_relation_mix", 0.0)))
    support_aux_mix = max(0.0, float(params.get("support_aux_mix", 0.0)))
    candidate_relation_mix = max(0.0, float(params.get("candidate_relation_mix", 0.01)))
    candidate_aux_mix = max(0.0, float(params.get("candidate_aux_mix", 0.0)))
    edge_gain = max(0.0, float(params.get("edge_gain", 0.001)))
    wave_mix = min(max(float(params.get("wave_mix", 0.18)), 0.0), 1.0)
    boundary_mix = min(max(float(params.get("boundary_mix", 0.52)), 0.0), 1.0)
    fringe_mix = min(max(float(params.get("fringe_mix", 0.12)), 0.0), 1.0)
    profile_mix = max(0.0, float(params.get("profile_mix", 0.10)))
    dim = int(params.get("dim", 0))
    secret_key = str(params.get("secret_key", ""))

    base_method = _projective_holographic_boundary_build(rng, params)
    base_aux_score = base_method.aux_score
    sample_state = base_method.encode_docs(np.zeros((1, dim), dtype=np.float32))
    hidden_dim = sample_state["wave_real"].shape[1]
    fringe_dim = sample_state["holographic_fringe"].shape[1]
    local_rng = np.random.default_rng(
        _method_seed("projective_holographic_query_edge_graph_v0", secret_key, dim)
    )

    relation_wave = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )
    relation_boundary = _safe_normalize(
        local_rng.normal(size=(relation_slots, hidden_dim)).astype(np.float32)
    )
    relation_fringe = _safe_normalize(
        local_rng.normal(size=(relation_slots, fringe_dim)).astype(np.float32)
    )

    def _merge_aux_profile(
        base_profile: np.ndarray | None, relation_profile: np.ndarray
    ) -> np.ndarray:
        if base_profile is None:
            return _safe_normalize(relation_profile.astype(np.float32, copy=False))
        if base_profile.shape[0] != relation_profile.shape[0]:
            return base_profile.astype(np.float32, copy=False)
        merged = np.concatenate(
            [
                base_profile.astype(np.float32, copy=False),
                profile_mix * relation_profile.astype(np.float32, copy=False),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _relation_profile(state: StateMap) -> np.ndarray:
        wave_energy = np.maximum(
            1e-6, state["wave_real"] ** 2 + state["wave_imag"] ** 2
        ).astype(np.float32)
        boundary_energy = np.maximum(1e-6, state["holographic_boundary"] ** 2).astype(
            np.float32
        )
        fringe_energy = np.maximum(1e-6, state["holographic_fringe"] ** 2).astype(
            np.float32
        )
        logits = wave_mix * (wave_energy @ relation_wave.T)
        logits = logits + boundary_mix * (boundary_energy @ relation_boundary.T)
        logits = logits + fringe_mix * (fringe_energy @ relation_fringe.T)
        return _topk_soft_assign(logits, relation_top_k, relation_temperature).astype(
            np.float32
        )

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_docs(x))
        relation_profile = _relation_profile(state)
        state["relation_profile"] = relation_profile
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), relation_profile
        )
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_queries(x))
        relation_profile = _relation_profile(state)
        state["relation_profile"] = relation_profile
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), relation_profile
        )
        return state

    def _base_aux(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        if base_aux_score is None:
            return np.zeros(
                (query_state["public"].shape[0], doc_state["public"].shape[0]),
                dtype=np.float32,
            )
        return base_aux_score(doc_state, query_state).astype(np.float32)

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base_method.score(doc_state, query_state).astype(np.float32)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return carrier

        doc_relation = doc_state.get("relation_profile")
        query_relation = query_state.get("relation_profile")
        if doc_relation is None:
            return carrier
        if query_relation is None:
            return carrier
        if doc_relation.shape[0] != doc_count:
            return carrier

        relation_score = query_relation @ doc_relation.T
        relation_score = np.maximum(relation_score, 0.0).astype(np.float32)
        relation_scale = relation_score / np.maximum(
            1e-6, np.max(relation_score, axis=1, keepdims=True)
        )
        base_aux = np.maximum(_base_aux(doc_state, query_state), 0.0)

        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return carrier

        support_seed = carrier * (1.0 + support_relation_mix * relation_scale)
        if support_aux_mix > 0.0:
            support_seed = support_seed + support_aux_mix * base_aux
        support_idx = np.argpartition(-support_seed, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(support_seed, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        weight_sum = np.sum(support_weights, axis=1, keepdims=True)
        weight_sum = np.where(weight_sum == 0.0, 1.0, weight_sum)
        support_weights = support_weights / weight_sum

        keep_rerank = min(rerank_width, doc_count)
        if keep_rerank <= 0:
            return carrier

        candidate_idx = np.argpartition(-carrier, keep_rerank - 1, axis=1)[
            :, :keep_rerank
        ]
        candidate_carrier = np.take_along_axis(carrier, candidate_idx, axis=1)
        candidate_relation = np.take_along_axis(relation_scale, candidate_idx, axis=1)
        candidate_aux = np.take_along_axis(base_aux, candidate_idx, axis=1)

        support_profiles = doc_relation[support_idx]
        candidate_profiles = doc_relation[candidate_idx]
        query_slots = query_relation[:, None, None, :]
        shared_slots = np.minimum(
            support_profiles[:, :, None, :], candidate_profiles[:, None, :, :]
        )
        query_edge = np.max(query_slots * shared_slots, axis=3)
        edge_signal = np.sum(
            query_edge * support_weights[:, :, None], axis=1, dtype=np.float32
        )
        candidate_signal = edge_signal + candidate_relation_mix * candidate_relation
        if candidate_aux_mix > 0.0:
            candidate_signal = candidate_signal + candidate_aux_mix * candidate_aux
        candidate_signal = candidate_signal - np.mean(
            candidate_signal, axis=1, keepdims=True
        )
        signal_scale = np.max(np.abs(candidate_signal), axis=1, keepdims=True)
        signal_scale = np.where(signal_scale == 0.0, 1.0, signal_scale)
        candidate_signal = candidate_signal / signal_scale
        rerank_gain = np.maximum(0.0, 1.0 + edge_gain * candidate_signal)
        reranked = candidate_carrier * rerank_gain
        scores = np.array(carrier, copy=True)
        np.put_along_axis(scores, candidate_idx, reranked, axis=1)
        return scores

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        base_aux = _base_aux(doc_state, query_state)
        relation = query_state["relation_profile"] @ doc_state["relation_profile"].T
        relation = relation - np.mean(relation, axis=1, keepdims=True)
        scale = np.max(np.abs(relation), axis=1, keepdims=True)
        scale = np.where(scale == 0.0, 1.0, scale)
        relation = relation / scale
        combined = base_aux + profile_mix * relation
        combined = combined - np.mean(combined, axis=1, keepdims=True)
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return (combined / combined_scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_holographic_query_edge_graph_v0",
        family="state_projective_holographic_query_edge_graph",
        params={
            **base_method.params,
            "relation_slots": float(relation_slots),
            "relation_top_k": float(relation_top_k),
            "relation_temperature": relation_temperature,
            "support_width": float(support_width),
            "rerank_width": float(rerank_width),
            "support_temperature": support_temperature,
            "support_relation_mix": support_relation_mix,
            "support_aux_mix": support_aux_mix,
            "candidate_relation_mix": candidate_relation_mix,
            "candidate_aux_mix": candidate_aux_mix,
            "edge_gain": edge_gain,
            "wave_mix": wave_mix,
            "boundary_mix": boundary_mix,
            "fringe_mix": fringe_mix,
            "profile_mix": profile_mix,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=_score,
        aux_score=_aux_score,
    )


def _projective_holographic_coresponse_observable_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    probes = max(4, min(24, int(params.get("probes", 8))))
    probe_top_k = min(probes, max(1, int(params.get("probe_top_k", 2))))
    probe_temperature = max(1e-4, float(params.get("probe_temperature", 0.11)))
    support_width = max(3, int(params.get("support_width", 8)))
    support_temperature = max(1e-4, float(params.get("support_temperature", 0.12)))
    support_floor = min(max(float(params.get("support_floor", 0.16)), 0.0), 0.85)
    wave_mix = min(max(float(params.get("wave_mix", 0.30)), 0.0), 1.0)
    boundary_mix = min(max(float(params.get("boundary_mix", 0.34)), 0.0), 1.0)
    fringe_mix = min(max(float(params.get("fringe_mix", 0.18)), 0.0), 1.0)
    code_mix = min(max(float(params.get("code_mix", 0.18)), 0.0), 1.0)
    query_mix = min(max(float(params.get("query_mix", 0.24)), 0.0), 1.0)
    aux_mix = min(max(float(params.get("aux_mix", 0.34)), 0.0), 1.5)
    profile_mix = max(0.0, float(params.get("profile_mix", 0.24)))
    dim = int(params.get("dim", 0))
    secret_key = str(params.get("secret_key", ""))

    base_method = _projective_holographic_boundary_build(rng, params)
    base_aux_score = base_method.aux_score
    sample_state = base_method.encode_docs(np.zeros((1, dim), dtype=np.float32))
    hidden_dim = sample_state["wave_real"].shape[1]
    fringe_dim = sample_state["holographic_fringe"].shape[1]
    local_rng = np.random.default_rng(
        _method_seed("projective_holographic_coresponse_observable_v0", secret_key, dim)
    )

    probe_wave_real = _safe_normalize(
        local_rng.normal(size=(probes, hidden_dim)).astype(np.float32)
    )
    probe_wave_imag = _safe_normalize(
        local_rng.normal(size=(probes, hidden_dim)).astype(np.float32)
    )
    probe_boundary = _safe_normalize(
        local_rng.normal(size=(probes, hidden_dim)).astype(np.float32)
    )
    probe_fringe = _safe_normalize(
        local_rng.normal(size=(probes, fringe_dim)).astype(np.float32)
    )
    support_probe_mix = (
        np.abs(local_rng.normal(size=(probes, support_width)).astype(np.float32)) + 0.05
    )
    support_probe_mix = support_probe_mix / np.maximum(
        1e-6, np.sum(support_probe_mix, axis=1, keepdims=True)
    )
    probe_permutation = local_rng.permutation(probes)

    def _normalize_rows(x: np.ndarray) -> np.ndarray:
        total = np.sum(x, axis=1, keepdims=True)
        total = np.where(total == 0.0, 1.0, total)
        return (x / total).astype(np.float32)

    def _normalize_probe_tensor(x: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(x, axis=2, keepdims=True)
        norm = np.where(norm == 0.0, 1.0, norm)
        return (x / norm).astype(np.float32)

    def _merge_aux_profile(
        base_profile: np.ndarray | None, coresponse_codes: np.ndarray
    ) -> np.ndarray:
        probe_codes = coresponse_codes[:, probe_permutation].astype(
            np.float32, copy=False
        )
        probe_profile = np.concatenate(
            [probe_codes, np.cumsum(probe_codes, axis=1).astype(np.float32)],
            axis=1,
        ).astype(np.float32, copy=False)
        if base_profile is None:
            return _safe_normalize(probe_profile)
        if base_profile.shape[0] != probe_profile.shape[0]:
            return base_profile.astype(np.float32, copy=False)
        merged = np.concatenate(
            [
                base_profile.astype(np.float32, copy=False),
                profile_mix * probe_profile,
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        merged = merged - np.mean(merged, axis=1, keepdims=True)
        return _safe_normalize(merged)

    def _coresponse_codes(state: StateMap) -> np.ndarray:
        wave_match = state["wave_real"] @ probe_wave_real.T
        wave_match = wave_match + state["wave_imag"] @ probe_wave_imag.T
        wave_match = np.clip(wave_match**2, 0.0, 1.0).astype(np.float32)
        boundary_match = np.clip(
            state["holographic_boundary"] @ probe_boundary.T,
            0.0,
            1.0,
        ).astype(np.float32)
        fringe_match = np.clip(
            state["holographic_fringe"] @ probe_fringe.T,
            0.0,
            1.0,
        ).astype(np.float32)
        logits = wave_mix * wave_match + boundary_mix * boundary_match
        logits = logits + fringe_mix * fringe_match
        return _topk_soft_assign(logits, probe_top_k, probe_temperature).astype(
            np.float32
        )

    def _encode_docs(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_docs(x))
        coresponse_codes = _coresponse_codes(state)
        state["coresponse_codes"] = coresponse_codes
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), coresponse_codes
        )
        return state

    def _encode_queries(x: np.ndarray) -> StateMap:
        state = dict(base_method.encode_queries(x))
        coresponse_codes = _coresponse_codes(state)
        state["coresponse_codes"] = coresponse_codes
        state["aux_operator_profile"] = _merge_aux_profile(
            state.get("aux_operator_profile"), coresponse_codes
        )
        return state

    def _base_aux(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        if base_aux_score is None:
            return np.zeros(
                (query_state["wave_real"].shape[0], doc_state["wave_real"].shape[0]),
                dtype=np.float32,
            )
        return base_aux_score(doc_state, query_state).astype(np.float32)

    def _aux_score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        carrier = base_method.score(doc_state, query_state).astype(np.float32)
        base_aux = _base_aux(doc_state, query_state)
        doc_count = carrier.shape[1]
        if doc_count == 0:
            return base_aux

        doc_codes = doc_state.get("coresponse_codes")
        query_codes = query_state.get("coresponse_codes")
        if doc_codes is None:
            return base_aux
        if query_codes is None:
            return base_aux
        if doc_codes.shape[0] != doc_count:
            return base_aux

        keep_support = min(support_width, doc_count)
        if keep_support <= 0:
            return base_aux

        support_seed = carrier + 0.10 * np.maximum(base_aux, 0.0)
        support_idx = np.argpartition(-support_seed, keep_support - 1, axis=1)[
            :, :keep_support
        ]
        support_scores = np.take_along_axis(support_seed, support_idx, axis=1)
        shifted = support_scores - np.max(support_scores, axis=1, keepdims=True)
        support_weights = np.exp(shifted / support_temperature).astype(np.float32)
        support_weights = _normalize_rows(support_weights)

        support_codes = doc_codes[support_idx]
        support_query_align = np.sum(
            query_codes[:, None, :] * support_codes,
            axis=2,
            dtype=np.float32,
        )
        support_query_scale = np.max(support_query_align, axis=1, keepdims=True)
        support_query_scale = np.where(
            support_query_scale == 0.0, 1.0, support_query_scale
        )
        support_query_align = support_query_align / support_query_scale
        support_gate = support_weights * (
            support_floor + (1.0 - support_floor) * support_query_align
        )
        support_gate = _normalize_rows(support_gate)

        probe_mix = support_probe_mix[:, :keep_support].astype(np.float32, copy=False)
        probe_support = support_gate[:, None, :] * probe_mix[None, :, :]
        probe_norm = np.sum(probe_support, axis=2, keepdims=True)
        probe_norm = np.where(probe_norm == 0.0, 1.0, probe_norm)
        probe_support = (probe_support / probe_norm).astype(np.float32)

        support_wave_real = doc_state["wave_real"][support_idx]
        support_wave_imag = doc_state["wave_imag"][support_idx]
        support_boundary = doc_state["holographic_boundary"][support_idx]
        support_fringe = doc_state["holographic_fringe"][support_idx]
        probe_wave_real_state = np.einsum(
            "qps,qsd->qpd", probe_support, support_wave_real, dtype=np.float32
        )
        probe_wave_imag_state = np.einsum(
            "qps,qsd->qpd", probe_support, support_wave_imag, dtype=np.float32
        )
        probe_boundary_state = np.einsum(
            "qps,qsd->qpd", probe_support, support_boundary, dtype=np.float32
        )
        probe_fringe_state = np.einsum(
            "qps,qsd->qpd", probe_support, support_fringe, dtype=np.float32
        )
        probe_code_state = np.einsum(
            "qps,qsk->qpk", probe_support, support_codes, dtype=np.float32
        )
        probe_wave_real_state = _normalize_probe_tensor(probe_wave_real_state)
        probe_wave_imag_state = _normalize_probe_tensor(probe_wave_imag_state)
        probe_boundary_state = _normalize_probe_tensor(probe_boundary_state)
        probe_fringe_state = _normalize_probe_tensor(probe_fringe_state)
        probe_code_state = _normalize_probe_tensor(probe_code_state)

        query_wave = np.sum(
            probe_wave_real_state * query_state["wave_real"][:, None, :],
            axis=2,
            dtype=np.float32,
        )
        query_wave = query_wave + np.sum(
            probe_wave_imag_state * query_state["wave_imag"][:, None, :],
            axis=2,
            dtype=np.float32,
        )
        query_wave = np.clip(query_wave**2, 0.0, 1.0)
        query_boundary = np.clip(
            np.sum(
                probe_boundary_state * query_state["holographic_boundary"][:, None, :],
                axis=2,
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        query_fringe = np.clip(
            np.sum(
                probe_fringe_state * query_state["holographic_fringe"][:, None, :],
                axis=2,
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        query_code = np.clip(
            np.sum(
                probe_code_state * query_codes[:, None, :],
                axis=2,
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        query_logits = wave_mix * query_wave + boundary_mix * query_boundary
        query_logits = query_logits + fringe_mix * query_fringe + code_mix * query_code
        query_target = _topk_soft_assign(query_logits, probe_top_k, probe_temperature)
        support_target = np.sum(
            probe_support * support_query_align[:, None, :],
            axis=2,
            dtype=np.float32,
        )
        query_target = (1.0 - query_mix) * query_target + query_mix * support_target
        query_target = _normalize_rows(query_target)

        candidate_wave = np.einsum(
            "qpd,nd->qpn",
            probe_wave_real_state,
            doc_state["wave_real"],
            dtype=np.float32,
        )
        candidate_wave = candidate_wave + np.einsum(
            "qpd,nd->qpn",
            probe_wave_imag_state,
            doc_state["wave_imag"],
            dtype=np.float32,
        )
        candidate_wave = np.clip(candidate_wave**2, 0.0, 1.0)
        candidate_boundary = np.clip(
            np.einsum(
                "qpd,nd->qpn",
                probe_boundary_state,
                doc_state["holographic_boundary"],
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        candidate_fringe = np.clip(
            np.einsum(
                "qpd,nd->qpn",
                probe_fringe_state,
                doc_state["holographic_fringe"],
                dtype=np.float32,
            ),
            0.0,
            1.0,
        )
        candidate_code = np.clip(
            np.einsum("qpk,nk->qpn", probe_code_state, doc_codes, dtype=np.float32),
            0.0,
            1.0,
        )
        candidate_signal = wave_mix * candidate_wave + boundary_mix * candidate_boundary
        candidate_signal = candidate_signal + fringe_mix * candidate_fringe
        candidate_signal = candidate_signal + code_mix * candidate_code
        candidate_signal = candidate_signal - np.mean(
            candidate_signal, axis=1, keepdims=True
        )
        target_signal = query_target - np.mean(query_target, axis=1, keepdims=True)
        candidate_norm = np.sqrt(
            np.sum(candidate_signal**2, axis=1, dtype=np.float32)
        ).astype(np.float32)
        target_norm = np.sqrt(
            np.sum(target_signal**2, axis=1, keepdims=True, dtype=np.float32)
        ).astype(np.float32)
        denom = np.maximum(1e-6, target_norm * candidate_norm)
        pattern_match = np.einsum(
            "qp,qpn->qn", target_signal, candidate_signal, dtype=np.float32
        )
        pattern_match = pattern_match / denom

        direct_code = np.clip(query_codes @ doc_codes.T, 0.0, 1.0)
        combined = base_aux + aux_mix * (0.78 * pattern_match + 0.22 * direct_code)
        combined = combined - np.mean(combined, axis=1, keepdims=True)
        combined_scale = np.max(np.abs(combined), axis=1, keepdims=True)
        combined_scale = np.where(combined_scale == 0.0, 1.0, combined_scale)
        return (combined / combined_scale).astype(np.float32)

    return EmbeddingStateMethod(
        method_name="projective_holographic_coresponse_observable_v0",
        family="state_projective_holographic_coresponse_observable",
        params={
            **base_method.params,
            "probes": float(probes),
            "probe_top_k": float(probe_top_k),
            "probe_temperature": probe_temperature,
            "support_width": float(support_width),
            "support_temperature": support_temperature,
            "support_floor": support_floor,
            "wave_mix": wave_mix,
            "boundary_mix": boundary_mix,
            "fringe_mix": fringe_mix,
            "code_mix": code_mix,
            "query_mix": query_mix,
            "aux_mix": aux_mix,
            "profile_mix": profile_mix,
        },
        encode_docs=_encode_docs,
        encode_queries=_encode_queries,
        score=base_method.score,
        aux_score=_aux_score,
    )


def _grassmann_stiefel_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    dim = int(params.get("dim", 0))
    frame_rank = int(params.get("frame_rank", 3))
    subspace_dim = max(
        frame_rank + 2, int(params.get("subspace_dim", max(8, dim // 4)))
    )
    public_ratio = float(params.get("public_ratio", 0.22))
    public_mask = float(params.get("public_mask", 0.78))
    public_chunk = int(params.get("public_chunk", 4))
    public_dim = max(8, int(dim * public_ratio))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(
        _method_seed("grassmann_stiefel_embedding_v0", secret_key, dim)
    )
    frame_proj = np.stack(
        [_qr_orthogonal(local_rng, dim, subspace_dim) for _ in range(frame_rank)]
    )
    public_mix = local_rng.normal(size=(subspace_dim, public_dim)).astype(np.float32)

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        raw = np.einsum("nd,rdh->nrh", y, frame_proj)
        frames = []
        projectors = []
        for sample in raw:
            q, _ = np.linalg.qr(sample.T)
            frame = q[:, :frame_rank].T.astype(np.float32)
            projector = frame.T @ frame
            frames.append(frame)
            projectors.append(projector.astype(np.float32))
        frame_stack = np.stack(frames, axis=0)
        projector_stack = np.stack(projectors, axis=0)
        diag = np.diagonal(projector_stack, axis1=1, axis2=2)
        public = _safe_normalize(diag @ public_mix)
        public = _mask_public_observation(
            "grassmann_stiefel_embedding_v0",
            secret_key,
            public,
            public_mask,
            public_chunk,
        )
        return {"public": public, "projector": projector_stack, "frame": frame_stack}

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        overlap = np.einsum(
            "nij,mij->nm", query_state["projector"], doc_state["projector"]
        )
        return overlap / float(max(1, frame_rank))

    return EmbeddingStateMethod(
        method_name="grassmann_stiefel_embedding_v0",
        family="state_subspace_manifold",
        params={
            "dim": dim,
            "frame_rank": float(frame_rank),
            "subspace_dim": float(subspace_dim),
            "public_ratio": public_ratio,
            "public_mask": public_mask,
            "public_chunk": float(public_chunk),
        },
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


def _dcpe_sap_build(
    rng: np.random.Generator, params: Dict[str, float]
) -> EmbeddingStateMethod:
    """DCPE/SAP Baseline (Fuchsbauer et al. 2022): Scale-And-Perturb.

    Verschlüsselt Vektoren durch key-abhängige Permutation + Skalierung + Gauss-Noise.
    Der verschlüsselte Vektor IST das öffentliche Embedding (kein separater Public Layer).
    Approximate Distance Comparison bleibt erhalten — das ist die bewusste Leakage.
    """
    dim = int(params.get("dim", 0))
    scale = float(params.get("scale", 1.0))
    noise_std = float(params.get("noise_std", 0.1))
    secret_key = str(params.get("secret_key", ""))
    local_rng = np.random.default_rng(_method_seed("dcpe_sap_v0", secret_key, dim))
    perm = local_rng.permutation(dim)
    perm_matrix = np.eye(dim, dtype=np.float32)[perm]
    rotation = _qr_orthogonal(local_rng, dim, dim)
    from cryptography.hazmat.primitives.kdf.hkdf import HKDF as _HKDF
    from cryptography.hazmat.primitives import hashes as _hashes
    noise_seed_base = int.from_bytes(
        _HKDF(algorithm=_hashes.SHA512(), length=64, salt=b"kpt-dcpe-noise",
               info=f"{dim}".encode()).derive(secret_key.encode("utf-8")),
        "big",
    )

    def _encode(x: np.ndarray) -> StateMap:
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        transformed = scale * (y @ perm_matrix.T @ rotation.T)
        n = transformed.shape[0]
        noise_rng = np.random.default_rng(noise_seed_base)
        noise = noise_rng.normal(0.0, noise_std, size=(n, dim)).astype(np.float32)
        encrypted = transformed + noise
        return {"public": encrypted.astype(np.float32)}

    def _score(doc_state: StateMap, query_state: StateMap) -> np.ndarray:
        return query_state["public"] @ doc_state["public"].T

    return EmbeddingStateMethod(
        method_name="dcpe_sap_v0",
        family="state_cryptographic_baseline",
        params={"dim": dim, "scale": scale, "noise_std": noise_std},
        encode_docs=_encode,
        encode_queries=_encode,
        score=_score,
    )


METHOD_REGISTRY = {
    "baseline_vector_state": _baseline_vector_build,
    "dcpe_sap_v0": _dcpe_sap_build,
    "complex_wavepacket_v0": _complex_wavepacket_build,
    "complex_wavepacket_v1": _complex_wavepacket_v1_build,
    "complex_observer_subgraph_transport_observable_v0": _complex_observer_subgraph_transport_observable_build,
    "complex_observer_prototype_support_observable_v0": _complex_observer_prototype_support_observable_build,
    "complex_parallel_holonomy_observable_v0": _complex_parallel_holonomy_observable_build,
    "complex_transport_holonomy_observable_v0": _complex_transport_holonomy_observable_build,
    "observer_resonance_v0": _observer_resonance_build,
    "tensor_graph_v0": _tensor_graph_build,
    "holonomy_loop_embedding_v0": _holonomy_loop_build,
    "codebook_superposition_embedding_v0": _codebook_superposition_build,
    "keyed_wave_superpose_embedding_v0": _keyed_wave_superpose_build,
    "dual_carrier_superposition_v0": _dual_carrier_superposition_build,
    "spin_glass_embedding_v0": _spin_glass_build,
    "spread_spectrum_carrier_v0": _spread_spectrum_carrier_build,
    "broadcast_noise_embedding_v0": _broadcast_noise_embedding_build,
    "matched_filter_embedding_v0": _matched_filter_embedding_build,
    "keyed_collapse_carrier_v0": _keyed_collapse_carrier_build,
    "fountain_code_carrier_v0": _fountain_code_carrier_build,
    "ecc_syndrome_embedding_v0": _ecc_syndrome_embedding_build,
    "dual_observer_embedding_v0": _dual_observer_build,
    "hilbert_space_embedding_v0": _hilbert_space_build,
    "projective_hilbert_embedding_v0": _projective_hilbert_build,
    "projective_duality_bipartite_state_v0": _projective_duality_bipartite_state_build,
    "projective_phase_codebook_superposition_v0": _projective_phase_codebook_superposition_build,
    "projective_resonance_window_decode_v0": _projective_resonance_window_decode_build,
    "projective_pairwise_tournament_collapse_v0": _projective_pairwise_tournament_collapse_build,
    "projective_support_hypothesis_anchor_v0": _projective_support_hypothesis_anchor_build,
    "protein_folding_minima_embedding_v0": _protein_folding_minima_embedding_build,
    "metamaterial_channel_embedding_v0": _metamaterial_channel_embedding_build,
    "immune_self_nonself_embedding_v0": _immune_self_nonself_embedding_build,
    "projective_kahler_symplectic_hybrid_v0": _projective_kahler_symplectic_hybrid_build,
    "projective_observer_resonance_hybrid_v0": _projective_observer_resonance_hybrid_build,
    "projective_chart_observable_v0": _projective_chart_observable_build,
    "projective_topological_code_observable_v0": _projective_topological_code_observable_build,
    "projective_keyed_collapse_observable_v0": _projective_keyed_collapse_observable_build,
    "causal_set_percolation_embedding_v0": _causal_set_percolation_embedding_build,
    "complex_wave_kahler_symplectic_hybrid_v0": _complex_wave_kahler_symplectic_hybrid_build,
    "projective_phase_holonomy_observable_v0": _projective_phase_holonomy_observable_build,
    "projective_parallel_holonomy_observable_v0": _projective_parallel_holonomy_observable_build,
    "projective_residual_holonomy_observable_v0": _projective_residual_holonomy_observable_build,
    "projective_graph_transport_observable_v0": _projective_graph_transport_observable_build,
    "projective_observer_subgraph_transport_observable_v0": _projective_observer_subgraph_transport_observable_build,
    "projective_observer_community_support_observable_v0": _projective_observer_community_support_observable_build,
    "projective_observer_prototype_support_observable_v0": _projective_observer_prototype_support_observable_build,
    "projective_observer_prototype_routing_observable_v0": _projective_observer_prototype_routing_observable_build,
    "projective_observer_prototype_residual_routing_observable_v0": _projective_observer_prototype_residual_routing_observable_build,
    "projective_observer_subgraph_head_v0": _projective_observer_subgraph_head_build,
    "projective_observer_ambiguity_head_v0": _projective_observer_ambiguity_head_build,
    "projective_observer_margin_coherence_head_v0": _projective_observer_margin_coherence_head_build,
    "projective_observer_prototype_margin_coherence_head_v0": _projective_observer_prototype_margin_coherence_head_build,
    "projective_observer_family_selective_residual_head_v0": _projective_observer_family_selective_residual_head_build,
    "projective_observer_family_order_flip_head_v0": _projective_observer_family_order_flip_head_build,
    "projective_semantic_codebook_head_v0": _projective_semantic_codebook_head_build,
    "projective_relational_head_v0": _projective_relational_head_build,
    "projective_anchor_profile_v0": _projective_anchor_profile_build,
    "projective_bipartite_profile_v0": _projective_bipartite_profile_build,
    "projective_graph_support_head_v0": _projective_graph_support_head_build,
    "projective_consensus_graph_head_v0": _projective_consensus_graph_head_build,
    "projective_margin_graph_support_v0": _projective_margin_graph_support_build,
    "projective_coherence_routed_graph_support_v0": _projective_coherence_routed_graph_support_build,
    "projective_semantic_prior_graph_support_v0": _projective_semantic_prior_graph_support_build,
    "projective_semantic_listdecode_graph_v0": _projective_semantic_listdecode_graph_build,
    "projective_observer_listdecode_graph_v0": _projective_observer_listdecode_graph_build,
    "projective_query_gated_graph_support_v0": _projective_query_gated_graph_support_build,
    "projective_query_edge_graph_v0": _projective_query_edge_graph_build,
    "projective_observer_cograph_v0": _projective_observer_cograph_build,
    "projective_multiquery_cograph_v0": _projective_multiquery_cograph_build,
    "projective_stable_codebook_head_v0": _projective_stable_codebook_head_build,
    "projective_chart_resonance_v0": _projective_chart_resonance_build,
    "kernel_observable_embedding_v0": _kernel_observable_build,
    "calabi_yau_chart_embedding_v0": _calabi_yau_chart_build,
    "kahler_symplectic_embedding_v0": _kahler_symplectic_build,
    "bregman_entropy_distortion_v0": _bregman_entropy_distortion_build,
    "piecewise_isometry_breaker_v0": _piecewise_isometry_breaker_build,
    "keyed_phase_vortex_flow_v0": _keyed_phase_vortex_flow_build,
    "projective_bregman_vortex_hybrid_v0": _projective_bregman_vortex_hybrid_build,
    "projective_holographic_boundary_v0": _projective_holographic_boundary_build,
    "projective_holographic_support_bipartite_state_v0": _projective_holographic_support_bipartite_state_build,
    "projective_holographic_noncommutative_observable_v0": _projective_holographic_noncommutative_observable_build,
    "projective_holographic_codebook_order_observable_v0": _projective_holographic_codebook_order_observable_build,
    "projective_holographic_gauge_observable_v0": _projective_holographic_gauge_observable_build,
    "projective_holographic_coreasonance_observable_v0": _projective_holographic_coreasonance_observable_build,
    "projective_holographic_coresponse_observable_v0": _projective_holographic_coresponse_observable_build,
    "projective_holographic_query_edge_graph_v0": _projective_holographic_query_edge_graph_build,
    "grassmann_stiefel_embedding_v0": _grassmann_stiefel_build,
}


def _parse_method_token(token: str) -> Tuple[str, Dict[str, float]]:
    if ":" not in token:
        return token.strip(), {}
    slug, raw_params = token.split(":", 1)
    params: Dict[str, float] = {}
    for part in raw_params.split(","):
        entry = part.strip()
        if not entry:
            continue
        key, value = entry.split("=", 1)
        params[key.strip()] = float(value.strip())
    return slug.strip(), params


def _load_plan_from_file(path: str) -> List[Tuple[str, Dict[str, float]]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    methods = payload.get("methods", [])
    out: List[Tuple[str, Dict[str, float]]] = []
    for method in methods:
        out.append((method["slug"], method.get("params", {})))
    return out


def _default_plan(dim: int) -> List[Tuple[str, Dict[str, float]]]:
    return [
        ("baseline_vector_state", {}),
        (
            "complex_wavepacket_v1",
            {
                "bands": 5,
                "phase_scale": 0.8,
                "envelope_gain": 0.45,
                "public_ratio": 0.25,
                "public_mask": 0.72,
                "public_chunk": 5,
            },
        ),
        (
            "observer_resonance_v0",
            {"channels": 6, "channel_gain": 0.60, "public_ratio": 0.33},
        ),
        ("tensor_graph_v0", {"slots": 6, "public_ratio": 0.50, "relation_gain": 0.45}),
        (
            "holonomy_loop_embedding_v0",
            {"loops": 8, "public_ratio": 0.24, "public_mask": 0.66, "public_chunk": 4},
        ),
        (
            "codebook_superposition_embedding_v0",
            {
                "codebook_size": 64,
                "top_k": 5,
                "residual_weight": 0.34,
                "public_ratio": 0.20,
                "public_mask": 0.78,
                "public_chunk": 4,
            },
        ),
        (
            "dual_observer_embedding_v0",
            {
                "branches": 4,
                "cross_weight": 0.58,
                "public_ratio": 0.22,
                "public_mask": 0.70,
                "public_chunk": 4,
            },
        ),
    ]


def _flatten_metrics(result: Dict[str, object]) -> Dict[str, float]:
    topo = result["topology"]
    attack = result["attack"]
    wrong_key = result["wrong_key"]
    overhead = result["overhead"]
    relational_graph = result["relational_graph"]
    observable_graph = result.get("observable_graph", {})
    aux_operator = result.get("aux_operator", {})
    observer_query = result.get("observer_query", {})
    semantic_query = result.get("semantic_query", {})
    return {
        "recall_at_k": float(result["recall_at_k"]),
        "public_recall_at_k": float(result["public_probe"]["recall_at_k"]),
        "public_recall_ratio": float(result["public_probe"]["recall_ratio"]),
        "public_top1_match_rate": float(result["public_probe"]["top1_match_rate"]),
        "wrong_key_recall_at_k": float(wrong_key["recall_at_k"]),
        "wrong_key_recall_ratio": float(wrong_key["recall_ratio"]),
        "wrong_key_top1_match_rate": float(wrong_key["top1_match_rate"]),
        "community_ari": float(result["community_ari"]),
        "community_nmi": float(result["community_nmi"]),
        "public_label_ari": float(result["public_label_alignment"]["ari"]),
        "score_label_ari": float(result["score_label_alignment"]["ari"]),
        "public_graph_homophily": float(relational_graph["public_label_homophily"]),
        "score_graph_homophily": float(relational_graph["score_label_homophily"]),
        "score_graph_homophily_gain": float(
            relational_graph["score_label_homophily"]
            - relational_graph["public_label_homophily"]
        ),
        "wrong_key_graph_homophily": float(
            relational_graph["wrong_key_label_homophily"]
        ),
        "wrong_key_graph_ratio": float(relational_graph["wrong_key_homophily_ratio"]),
        "wrong_key_graph_overlap": float(relational_graph["wrong_key_edge_overlap"]),
        "observable_available": float(observable_graph.get("available", 0.0)),
        "observable_label_ari": float(observable_graph.get("label_ari", 0.0)),
        "observable_label_nmi": float(observable_graph.get("label_nmi", 0.0)),
        "observable_graph_homophily": float(
            observable_graph.get("graph_homophily", 0.0)
        ),
        "observable_wrong_key_graph_homophily": float(
            observable_graph.get("wrong_key_graph_homophily", 0.0)
        ),
        "observable_wrong_key_graph_ratio": float(
            observable_graph.get("wrong_key_graph_ratio", 0.0)
        ),
        "observable_wrong_key_graph_overlap": float(
            observable_graph.get("wrong_key_graph_overlap", 0.0)
        ),
        "aux_operator_available": float(aux_operator.get("available", 0.0)),
        "aux_operator_label_ari": float(aux_operator.get("label_ari", 0.0)),
        "aux_operator_graph_homophily": float(aux_operator.get("graph_homophily", 0.0)),
        "aux_operator_wrong_key_graph_homophily": float(
            aux_operator.get("wrong_key_graph_homophily", 0.0)
        ),
        "aux_operator_wrong_key_graph_ratio": float(
            aux_operator.get("wrong_key_graph_ratio", 0.0)
        ),
        "aux_operator_wrong_key_graph_overlap": float(
            aux_operator.get("wrong_key_graph_overlap", 0.0)
        ),
        "aux_operator_carrier_graph_overlap": float(
            aux_operator.get("carrier_graph_overlap", 0.0)
        ),
        "aux_operator_query_rank_corr": float(aux_operator.get("query_rank_corr", 0.0)),
        "aux_operator_query_topk_overlap": float(
            aux_operator.get("query_topk_overlap", 0.0)
        ),
        "observer_query_available": float(observer_query.get("available", 0.0)),
        "observer_ambiguity_mean": float(observer_query.get("ambiguity_mean", 0.0)),
        "observer_ambiguity_std": float(observer_query.get("ambiguity_std", 0.0)),
        "observer_effective_modes_mean": float(
            observer_query.get("effective_modes_mean", 0.0)
        ),
        "semantic_query_available": float(semantic_query.get("available", 0.0)),
        "semantic_focus_mean": float(semantic_query.get("focus_mean", 0.0)),
        "semantic_focus_std": float(semantic_query.get("focus_std", 0.0)),
        "semantic_effective_codes_mean": float(
            semantic_query.get("effective_codes_mean", 0.0)
        ),
        "centroid_corr": float(topo["centroid_distance_corr"]),
        "pair_corr": float(topo["pairwise_similarity_corr"]),
        "attack_p95": float(attack["linear_recon_cosine_p95"]),
        "attack_mean": float(attack["linear_recon_cosine_mean"]),
        "attack_mse": float(attack["linear_recon_mse"]),
        "nn_overlap": float(result["nn_overlap"]),
        "query_ms": float(overhead["query_ms"]),
        "doc_ms": float(overhead["doc_ms"]),
        "public_dim": float(overhead["public_dim"]),
        "entropy_signature": float(overhead["entropy_signature"]),
    }


def _aggregate_metric_rows(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = rows[0].keys()
    mean: Dict[str, float] = {}
    std: Dict[str, float] = {}
    for key in keys:
        values = np.array([row[key] for row in rows], dtype=np.float64)
        mean[key] = float(np.mean(values))
        std[key] = float(np.std(values))
    return {"mean": mean, "std": std}


def _aggregate_query_cohorts(
    seed_results: List[Dict[str, object]],
) -> Dict[str, object]:
    cohort_payloads = [
        result.get("query_cohorts")
        for result in seed_results
        if result.get("status") == "ok" and result.get("query_cohorts")
    ]
    if not cohort_payloads:
        return {"available": 0.0, "cohorts": {}}
    first = cohort_payloads[0]
    cohort_names = list(first["cohorts"].keys())
    aggregated: Dict[str, Dict[str, float]] = {}
    for cohort_name in cohort_names:
        rows = [payload["cohorts"][cohort_name] for payload in cohort_payloads]
        keys = rows[0].keys()
        aggregated[cohort_name] = {
            key: float(np.mean([row[key] for row in rows])) for key in keys
        }
    return {
        "available": 1.0,
        "margin_source": first["margin_source"],
        "margin_threshold_mean": float(
            np.mean([payload["margin_threshold"] for payload in cohort_payloads])
        ),
        "purity_threshold_mean": float(
            np.mean([payload["purity_threshold"] for payload in cohort_payloads])
        ),
        "observer_ambiguity_available": float(
            np.mean(
                [
                    payload.get("observer_ambiguity_available", 0.0)
                    for payload in cohort_payloads
                ]
            )
        ),
        "observer_ambiguity_threshold_mean": float(
            np.mean(
                [
                    payload.get("observer_ambiguity_threshold", 0.0)
                    for payload in cohort_payloads
                ]
            )
        ),
        "semantic_focus_available": float(
            np.mean(
                [
                    payload.get("semantic_focus_available", 0.0)
                    for payload in cohort_payloads
                ]
            )
        ),
        "semantic_focus_threshold_mean": float(
            np.mean(
                [
                    payload.get("semantic_focus_threshold", 0.0)
                    for payload in cohort_payloads
                ]
            )
        ),
        "cohorts": aggregated,
    }


def _run_single_method(
    method_slug: str,
    params: Dict[str, float],
    seed: int,
    dataset: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    top_k: int,
    n_clusters: int,
    secret_key: str,
    wrong_secret_key: str,
    key_bits: int,
) -> Dict[str, object]:
    rng = _set_seed(seed)
    docs, queries, doc_labels, _ = dataset
    dim = docs.shape[1]
    params_with_dim = dict(params)
    params_with_dim["dim"] = dim
    params_with_dim["secret_key"] = secret_key
    builder = METHOD_REGISTRY.get(method_slug)
    if builder is None:
        return {
            "method": method_slug,
            "status": "missing_method",
            "error": f"Method '{method_slug}' not found",
        }

    method = builder(rng, params_with_dim)
    gt_topk = _ground_truth_topk(docs, queries, top_k)
    doc_state = method.encode_docs(docs)
    query_state = method.encode_queries(queries)
    scores = method.score(doc_state, query_state)
    public_docs = _safe_normalize(doc_state["public"].astype(np.float32, copy=False))
    public_queries = _safe_normalize(
        query_state["public"].astype(np.float32, copy=False)
    )
    recall = _recall_at_k(scores, gt_topk, top_k)
    doc_query_state = method.encode_queries(docs)
    wrong_key_params = dict(params_with_dim)
    wrong_key_params["secret_key"] = wrong_secret_key
    wrong_rng = _set_seed(seed)
    wrong_method = builder(wrong_rng, wrong_key_params)
    wrong_query_state = wrong_method.encode_queries(queries)
    wrong_doc_query_state = wrong_method.encode_queries(docs)
    wrong_scores = method.score(doc_state, wrong_query_state)
    carrier_scores = _carrier_similarity_scores(doc_state, query_state)
    public_scores = cosine_similarity(public_queries, public_docs)
    public_recall = _recall_at_k(public_scores, gt_topk, top_k)
    wrong_recall = _recall_at_k(wrong_scores, gt_topk, top_k)
    pred = np.argmax(scores, axis=1)
    public_pred = np.argmax(public_scores, axis=1)
    wrong_pred = np.argmax(wrong_scores, axis=1)
    public_top1_match_rate = float(np.mean(pred == public_pred))
    wrong_top1_match_rate = float(np.mean(pred == wrong_pred))
    community = _community_alignment(docs, public_docs, n_clusters)
    public_label_alignment = _label_alignment(public_docs, doc_labels, n_clusters)
    score_label_alignment = _score_profile_alignment(
        method, doc_state, doc_labels, n_clusters, rng
    )
    relational_graph = _relational_graph_metrics(
        method,
        doc_state,
        doc_query_state,
        wrong_doc_query_state,
        public_docs,
        doc_labels,
        n_clusters,
    )
    observable_graph = _semantic_code_observable_metrics(
        doc_state,
        doc_query_state,
        wrong_doc_query_state,
        doc_labels,
        n_clusters,
    )
    aux_operator = _aux_operator_metrics(
        doc_state,
        query_state,
        doc_query_state,
        wrong_doc_query_state,
        carrier_scores,
        doc_labels,
        n_clusters,
        top_k,
        method.aux_score,
    )
    observer_query = _observer_query_ambiguity(
        doc_state,
        query_state,
        carrier_scores,
        method.params,
    )
    semantic_query = _semantic_query_focus(
        doc_state,
        query_state,
        carrier_scores,
        method.params,
    )
    query_cohorts = _query_cohort_metrics(
        scores,
        wrong_scores,
        gt_topk,
        doc_labels,
        top_k,
        carrier_scores,
        observer_query.get("per_query_ambiguity"),
        semantic_query.get("per_query_focus"),
    )
    observer_query.pop("per_query_ambiguity", None)
    semantic_query.pop("per_query_focus", None)
    leak = _topology_leakage(docs, public_docs, rng)
    attack = _attack_proxies(docs, public_docs)
    nn_overlap = _nn_overlap(docs, public_docs)
    query_ms = _runtime_ms(method.encode_queries, queries, reps=3)
    doc_ms = _runtime_ms(method.encode_docs, docs, reps=3)

    return {
        "method": method_slug,
        "family": method.family,
        "params": method.params,
        "key_bits": key_bits,
        "recall_at_k": recall,
        "public_probe": {
            "recall_at_k": public_recall,
            "recall_ratio": float(public_recall / max(recall, 1e-8)),
            "top1_match_rate": public_top1_match_rate,
        },
        "wrong_key": {
            "recall_at_k": wrong_recall,
            "recall_ratio": float(wrong_recall / max(recall, 1e-8)),
            "top1_match_rate": wrong_top1_match_rate,
        },
        "community_ari": community["ari"],
        "community_nmi": community["nmi"],
        "public_label_alignment": public_label_alignment,
        "score_label_alignment": score_label_alignment,
        "relational_graph": relational_graph,
        "observable_graph": observable_graph,
        "aux_operator": aux_operator,
        "observer_query": observer_query,
        "semantic_query": semantic_query,
        "query_cohorts": query_cohorts,
        "topology": leak,
        "attack": attack,
        "nn_overlap": nn_overlap,
        "overhead": {
            "query_ms": query_ms,
            "doc_ms": doc_ms,
            "public_dim": int(public_docs.shape[1]),
            "entropy_signature": _entropy_of_components(public_docs),
        },
        "status": "ok",
    }


def _readable_summary(results: List[Dict[str, object]]) -> str:
    lines = [
        "method,key_bits,recall_mean,recall_std,public_recall_mean,public_recall_ratio_mean,public_top1_mean,wrong_key_recall_mean,wrong_key_ratio_mean,wrong_key_top1_mean,ari_mean,public_label_ari_mean,score_label_ari_mean,public_graph_h_mean,score_graph_h_mean,score_graph_gain_mean,wrong_graph_h_mean,wrong_graph_ratio_mean,wrong_graph_overlap_mean,observable_available_mean,observable_label_ari_mean,observable_label_nmi_mean,observable_graph_h_mean,observable_wrong_graph_h_mean,observable_wrong_graph_ratio_mean,observable_wrong_graph_overlap_mean,aux_operator_available_mean,aux_operator_label_ari_mean,aux_operator_graph_h_mean,aux_operator_wrong_graph_h_mean,aux_operator_wrong_graph_ratio_mean,aux_operator_wrong_graph_overlap_mean,aux_operator_carrier_graph_overlap_mean,aux_operator_query_rank_corr_mean,aux_operator_query_topk_overlap_mean,observer_query_available_mean,observer_ambiguity_mean,observer_ambiguity_std,observer_effective_modes_mean,semantic_query_available_mean,semantic_focus_mean,semantic_focus_std,semantic_effective_codes_mean,pair_corr_mean,attack_p95_mean,nn_overlap_mean,query_ms_mean,doc_ms_mean"
    ]
    for result in results:
        mean = result["metrics"]["mean"]
        std = result["metrics"]["std"]
        lines.append(
            f"{result['method']},"
            f"{result.get('key_bits', 0)},"
            f"{mean['recall_at_k']:.4f},"
            f"{std['recall_at_k']:.4f},"
            f"{mean['public_recall_at_k']:.4f},"
            f"{mean['public_recall_ratio']:.4f},"
            f"{mean['public_top1_match_rate']:.4f},"
            f"{mean['wrong_key_recall_at_k']:.4f},"
            f"{mean['wrong_key_recall_ratio']:.4f},"
            f"{mean['wrong_key_top1_match_rate']:.4f},"
            f"{mean['community_ari']:.4f},"
            f"{mean['public_label_ari']:.4f},"
            f"{mean['score_label_ari']:.4f},"
            f"{mean['public_graph_homophily']:.4f},"
            f"{mean['score_graph_homophily']:.4f},"
            f"{mean['score_graph_homophily_gain']:.4f},"
            f"{mean['wrong_key_graph_homophily']:.4f},"
            f"{mean['wrong_key_graph_ratio']:.4f},"
            f"{mean['wrong_key_graph_overlap']:.4f},"
            f"{mean['observable_available']:.4f},"
            f"{mean['observable_label_ari']:.4f},"
            f"{mean['observable_label_nmi']:.4f},"
            f"{mean['observable_graph_homophily']:.4f},"
            f"{mean['observable_wrong_key_graph_homophily']:.4f},"
            f"{mean['observable_wrong_key_graph_ratio']:.4f},"
            f"{mean['observable_wrong_key_graph_overlap']:.4f},"
            f"{mean['aux_operator_available']:.4f},"
            f"{mean['aux_operator_label_ari']:.4f},"
            f"{mean['aux_operator_graph_homophily']:.4f},"
            f"{mean['aux_operator_wrong_key_graph_homophily']:.4f},"
            f"{mean['aux_operator_wrong_key_graph_ratio']:.4f},"
            f"{mean['aux_operator_wrong_key_graph_overlap']:.4f},"
            f"{mean['aux_operator_carrier_graph_overlap']:.4f},"
            f"{mean['aux_operator_query_rank_corr']:.4f},"
            f"{mean['aux_operator_query_topk_overlap']:.4f},"
            f"{mean['observer_query_available']:.4f},"
            f"{mean['observer_ambiguity_mean']:.4f},"
            f"{mean['observer_ambiguity_std']:.4f},"
            f"{mean['observer_effective_modes_mean']:.4f},"
            f"{mean['semantic_query_available']:.4f},"
            f"{mean['semantic_focus_mean']:.4f},"
            f"{mean['semantic_focus_std']:.4f},"
            f"{mean['semantic_effective_codes_mean']:.4f},"
            f"{mean['pair_corr']:.4f},"
            f"{mean['attack_p95']:.4f},"
            f"{mean['nn_overlap']:.4f},"
            f"{mean['query_ms']:.3f},"
            f"{mean['doc_ms']:.3f}"
        )
    return "\n".join(lines)


def _readable_query_cohort_summary(results: List[Dict[str, object]]) -> str:
    lines = [
        "method,margin_source,hard_share_mean,hard_recall_mean,hard_delta_vs_carrier_mean,hard_observer_ambiguity_mean,hard_semantic_focus_mean,easy_recall_mean,easy_delta_vs_carrier_mean,coherent_recall_mean,coherent_delta_vs_carrier_mean,coherent_observer_ambiguity_mean,coherent_semantic_focus_mean,diffuse_recall_mean,diffuse_delta_vs_carrier_mean,hard_coherent_share_mean,hard_coherent_recall_mean,hard_coherent_delta_vs_carrier_mean,observer_ambiguity_available_mean,observer_ambiguous_share_mean,observer_ambiguous_recall_mean,observer_ambiguous_delta_vs_carrier_mean,observer_focused_recall_mean,observer_focused_delta_vs_carrier_mean,semantic_focus_available_mean,semantic_focused_share_mean,semantic_focused_recall_mean,semantic_focused_delta_vs_carrier_mean,semantic_unfocused_recall_mean,semantic_unfocused_delta_vs_carrier_mean"
    ]
    for result in results:
        query_cohorts = result.get("query_cohorts", {})
        if query_cohorts.get("available", 0.0) <= 0.0:
            continue
        cohorts = query_cohorts["cohorts"]
        hard = cohorts.get("hard", {})
        easy = cohorts.get("easy", {})
        coherent = cohorts.get("coherent", {})
        diffuse = cohorts.get("diffuse", {})
        hard_coherent = cohorts.get("hard_coherent", {})
        observer_ambiguous = cohorts.get("observer_ambiguous", {})
        observer_focused = cohorts.get("observer_focused", {})
        semantic_focused = cohorts.get("semantic_focused", {})
        semantic_unfocused = cohorts.get("semantic_unfocused", {})
        lines.append(
            f"{result['method']},"
            f"{query_cohorts.get('margin_source', 'unknown')},"
            f"{hard.get('query_share', 0.0):.4f},"
            f"{hard.get('recall_at_k', 0.0):.4f},"
            f"{hard.get('delta_vs_carrier', 0.0):.4f},"
            f"{hard.get('observer_ambiguity_mean', 0.0):.4f},"
            f"{hard.get('semantic_focus_mean', 0.0):.4f},"
            f"{easy.get('recall_at_k', 0.0):.4f},"
            f"{easy.get('delta_vs_carrier', 0.0):.4f},"
            f"{coherent.get('recall_at_k', 0.0):.4f},"
            f"{coherent.get('delta_vs_carrier', 0.0):.4f},"
            f"{coherent.get('observer_ambiguity_mean', 0.0):.4f},"
            f"{coherent.get('semantic_focus_mean', 0.0):.4f},"
            f"{diffuse.get('recall_at_k', 0.0):.4f},"
            f"{diffuse.get('delta_vs_carrier', 0.0):.4f},"
            f"{hard_coherent.get('query_share', 0.0):.4f},"
            f"{hard_coherent.get('recall_at_k', 0.0):.4f},"
            f"{hard_coherent.get('delta_vs_carrier', 0.0):.4f},"
            f"{query_cohorts.get('observer_ambiguity_available', 0.0):.4f},"
            f"{observer_ambiguous.get('query_share', 0.0):.4f},"
            f"{observer_ambiguous.get('recall_at_k', 0.0):.4f},"
            f"{observer_ambiguous.get('delta_vs_carrier', 0.0):.4f},"
            f"{observer_focused.get('recall_at_k', 0.0):.4f},"
            f"{observer_focused.get('delta_vs_carrier', 0.0):.4f},"
            f"{query_cohorts.get('semantic_focus_available', 0.0):.4f},"
            f"{semantic_focused.get('query_share', 0.0):.4f},"
            f"{semantic_focused.get('recall_at_k', 0.0):.4f},"
            f"{semantic_focused.get('delta_vs_carrier', 0.0):.4f},"
            f"{semantic_unfocused.get('recall_at_k', 0.0):.4f},"
            f"{semantic_unfocused.get('delta_vs_carrier', 0.0):.4f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="State-based embedding lab runner")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--seed-runs", type=int, default=3)
    parser.add_argument(
        "--dataset-kind",
        type=str,
        default="latent",
        choices=["latent", "synthetic_text", "hf_text", "sklearn_20ng", "msmarco"],
    )
    parser.add_argument("--docs", type=int, default=6000)
    parser.add_argument("--queries", type=int, default=1200)
    parser.add_argument("--dim", type=int, default=96)
    parser.add_argument("--vectorizer", choices=["tfidf", "sbert"], default="tfidf")
    parser.add_argument("--clusters", type=int, default=12)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument(
        "--method",
        action="append",
        default=[],
        help="Format: slug or slug:key=value,key2=value2",
    )
    parser.add_argument("--plan", type=str, default="", help="JSON file with methods[]")
    parser.add_argument(
        "--secret-key", type=str, default="", help="Master key for keyed state methods"
    )
    parser.add_argument(
        "--out-dir", type=str, default="experiments/privacy_mathlab/results"
    )
    parser.add_argument("--hf-dataset", type=str, default="ag_news")
    parser.add_argument("--hf-config", type=str, default="")
    parser.add_argument("--hf-doc-split", type=str, default="train")
    parser.add_argument("--hf-query-split", type=str, default="test")
    parser.add_argument("--hf-text-field", type=str, default="")
    parser.add_argument("--hf-label-field", type=str, default="")
    parser.add_argument("--hf-doc-pool", type=int, default=24000)
    parser.add_argument("--hf-query-pool", type=int, default=6000)
    parser.add_argument("--key-bits", type=int, default=256)
    args = parser.parse_args()

    global _VECTORIZER_KIND
    _VECTORIZER_KIND = args.vectorizer

    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    os.makedirs(args.out_dir, exist_ok=True)

    method_specs: List[Tuple[str, Dict[str, float]]]
    if args.plan:
        method_specs = _load_plan_from_file(args.plan)
    if args.method:
        method_specs = [_parse_method_token(token) for token in args.method]
    if "method_specs" not in locals():
        method_specs = _default_plan(args.dim)

    true_secret_key = _normalized_key_material(args.secret_key, args.key_bits, "true")
    wrong_secret_key = _normalized_key_material(args.secret_key, args.key_bits, "wrong")

    results: List[Dict[str, object]] = []
    for method_slug, params in method_specs:
        per_seed_rows = []
        per_seed_results = []
        for seed_offset in range(args.seed_runs):
            dataset_seed = args.seed + seed_offset * 17
            seed = dataset_seed
            rng = _set_seed(dataset_seed)
            dataset = _build_dataset(
                kind=args.dataset_kind,
                rng=rng,
                n_docs=args.docs,
                n_queries=args.queries,
                dim=args.dim,
                n_clusters=args.clusters,
                seed=dataset_seed,
                hf_dataset=args.hf_dataset,
                hf_config=args.hf_config,
                hf_doc_split=args.hf_doc_split,
                hf_query_split=args.hf_query_split,
                hf_text_field=args.hf_text_field,
                hf_label_field=args.hf_label_field,
                hf_doc_pool=args.hf_doc_pool,
                hf_query_pool=args.hf_query_pool,
            )
            result = _run_single_method(
                method_slug=method_slug,
                params=params,
                seed=seed,
                dataset=dataset,
                top_k=args.k,
                n_clusters=args.clusters,
                secret_key=true_secret_key,
                wrong_secret_key=wrong_secret_key,
                key_bits=args.key_bits,
            )
            if result.get("status") != "ok":
                per_seed_results.append(result)
                continue
            per_seed_results.append(result)
            per_seed_rows.append(_flatten_metrics(result))
        if not per_seed_rows:
            results.append(
                {
                    "method": method_slug,
                    "status": "error",
                    "error": "No successful seed runs",
                    "per_seed": per_seed_results,
                }
            )
            continue
        builder = METHOD_REGISTRY[method_slug]
        probe_rng = _set_seed(args.seed)
        probe_method = builder(
            probe_rng, {"dim": args.dim, "secret_key": true_secret_key, **params}
        )
        results.append(
            {
                "method": method_slug,
                "family": probe_method.family,
                "params": probe_method.params,
                "key_bits": args.key_bits,
                "status": "ok",
                "seed_runs": args.seed_runs,
                "metrics": _aggregate_metric_rows(per_seed_rows),
                "query_cohorts": _aggregate_query_cohorts(per_seed_results),
                "per_seed": per_seed_results,
            }
        )

    output = {
        "run_id": run_id,
        "seed": args.seed,
        "seed_runs": args.seed_runs,
        "dataset": {
            "kind": args.dataset_kind,
            "docs": args.docs,
            "queries": args.queries,
            "dim": args.dim,
            "clusters": args.clusters,
            "k": args.k,
        },
        "key_eval": {
            "key_bits": args.key_bits,
            "wrong_key_variant": "deterministic-derived-mismatch",
        },
        "results": results,
    }

    out_json = os.path.join(args.out_dir, f"embedding_lab_{run_id}.json")
    with open(out_json, "w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)

    out_md = os.path.join(args.out_dir, f"embedding_lab_{run_id}.md")
    with open(out_md, "w", encoding="utf-8") as handle:
        handle.write(f"# Embedding-Lab Run {run_id}\n\n")
        handle.write("## Method Matrix\n\n")
        handle.write("```text\n")
        handle.write(_readable_summary(results))
        handle.write("\n```\n")
        cohort_summary = _readable_query_cohort_summary(results)
        if cohort_summary:
            handle.write("\n## Query Cohorts\n\n")
            handle.write("```text\n")
            handle.write(cohort_summary)
            handle.write("\n```\n")

    print(f"Run done. JSON: {out_json}")
    print(f"Report:  {out_md}")


if __name__ == "__main__":
    main()
