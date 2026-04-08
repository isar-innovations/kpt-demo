"""Nichtlinearitäts-Ablation: Welche Operationen erzeugen den 50×-Security-Floor?

Baut eine modifizierte Version von keyed_wave_superpose_embedding_v0 mit 8
Boolean-Flags zum Ein-/Ausschalten einzelner Nichtlinearitäten. Misst den
Security-Ratio (correct/wrong key magnitude) für jede Konfiguration über
verschiedene Dimensionen.

Befundlage:
- Voller Carrier: Ratio ~50× (konstant ab d≥64)
- Reiner Phasen-Carrier (nur cos/sin + mode phase shifts): Ratio skaliert super-linear
  (3753× bei d=1024)
- Die Nichtlinearitäten SENKEN den Ratio — diese Ablation findet heraus welche.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from itertools import product
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_lab import (
    EmbeddingStateMethod,
    _safe_normalize,
    _set_seed,
    _method_seed,
    _qr_orthogonal,
    _topk_soft_assign,
    _mask_public_observation,
)

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from datasets import load_dataset
except ImportError as e:
    raise SystemExit(f"Missing dependency: {e}")


@dataclass(frozen=True)
class AblationFlags:
    disable_tanh_amplitude: bool = False
    disable_topk_routing: bool = False
    disable_component_sq: bool = False
    disable_base_sq: bool = False
    disable_sqrt_mode_gate: bool = False
    disable_sqrt_energy_gate: bool = False
    disable_tanh_public: bool = False
    disable_decoy_floor: bool = False

    def label(self) -> str:
        flags = []
        for f in [
            "tanh_amp", "topk", "comp_sq", "base_sq",
            "sqrt_gate", "sqrt_energy", "tanh_pub", "decoy",
        ]:
            flags.append(f)
        active = []
        for name, val in zip(flags, [
            self.disable_tanh_amplitude, self.disable_topk_routing,
            self.disable_component_sq, self.disable_base_sq,
            self.disable_sqrt_mode_gate, self.disable_sqrt_energy_gate,
            self.disable_tanh_public, self.disable_decoy_floor,
        ]):
            if val:
                active.append(name)
        return "+".join(active) if active else "baseline"

    @staticmethod
    def all_disabled() -> "AblationFlags":
        return AblationFlags(True, True, True, True, True, True, True, True)


def build_ablated_carrier(
    dim: int, secret_key: str, flags: AblationFlags, seed: int = 42,
) -> EmbeddingStateMethod:
    """Modifizierter keyed_wave_superpose mit abschaltbaren Nichtlinearitäten."""
    hidden_dim = dim
    modes = 8
    doc_top_k = 3
    query_top_k = 1
    route_temperature = 0.22
    route_scale = 1.25
    collapse_gain = 2.2
    phase_scale = 0.78
    envelope_gain = 0.45
    decoy_floor = 0.0 if flags.disable_decoy_floor else 0.24
    coherence_weight = 0.46
    public_ratio = 0.18
    public_mask = 0.84
    public_chunk = 6
    public_dim = max(8, int(dim * public_ratio))

    local_rng = np.random.default_rng(
        _method_seed("keyed_wave_superpose_embedding_v0", secret_key, dim)
    )
    route_bank = _safe_normalize(local_rng.normal(size=(modes, hidden_dim)).astype(np.float32))
    route_bias = local_rng.uniform(-0.35, 0.35, size=(modes,)).astype(np.float32)
    real_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    imag_proj = _qr_orthogonal(local_rng, dim, hidden_dim)
    phase_bias = local_rng.uniform(0.0, 2.0 * math.pi, size=(hidden_dim,)).astype(np.float32)
    mode_phase_shift = local_rng.uniform(-math.pi, math.pi, size=(modes, hidden_dim)).astype(np.float32)
    public_mix = local_rng.normal(size=(modes, public_dim)).astype(np.float32)
    mode_cos = np.cos(mode_phase_shift).astype(np.float32)
    mode_sin = np.sin(mode_phase_shift).astype(np.float32)

    def _encode(x: np.ndarray, top_k: int, scale: float):
        y = _safe_normalize(np.array(x, dtype=np.float32, copy=True))
        carrier_real = y @ real_proj
        carrier_imag = y @ imag_proj
        carrier_phase = phase_scale * carrier_real + phase_bias[None, :]

        if flags.disable_tanh_amplitude:
            carrier_amp = np.sqrt(np.maximum(1e-6, 1.0 + envelope_gain * np.clip(carrier_imag, -1, 1))).astype(np.float32)
        else:
            carrier_amp = np.sqrt(np.maximum(1e-6, 1.0 + envelope_gain * np.tanh(carrier_imag))).astype(np.float32)

        base_wave_real = carrier_amp * np.cos(carrier_phase)
        base_wave_imag = carrier_amp * np.sin(carrier_phase)
        base_norm = np.sqrt(np.sum(base_wave_real**2 + base_wave_imag**2, axis=1, keepdims=True))
        base_norm = np.where(base_norm == 0.0, 1.0, base_norm)
        base_wave_real = base_wave_real / base_norm
        base_wave_imag = base_wave_imag / base_norm

        carrier_energy = (base_wave_real**2 + base_wave_imag**2).astype(np.float32)
        route_logits = (carrier_energy @ route_bank.T + route_bias[None, :]) / math.sqrt(max(1.0, float(hidden_dim)))

        if flags.disable_topk_routing:
            shifted = route_logits - np.max(route_logits, axis=1, keepdims=True)
            sparse_weight = np.exp(shifted / max(1e-4, route_temperature)).astype(np.float32)
            sparse_weight = sparse_weight / np.sum(sparse_weight, axis=1, keepdims=True)
        else:
            sparse_weight = _topk_soft_assign(route_logits * scale, top_k, route_temperature)

        mode_weight = decoy_floor / float(modes) + (1.0 - decoy_floor) * sparse_weight
        mode_total = np.sum(mode_weight, axis=1, keepdims=True)
        mode_total = np.where(mode_total == 0.0, 1.0, mode_total)
        mode_weight = mode_weight / mode_total

        mode_scale = np.sqrt(np.maximum(mode_weight, 1e-6)).astype(np.float32)[:, :, None]
        rotated_real = base_wave_real[:, None, :] * mode_cos[None] - base_wave_imag[:, None, :] * mode_sin[None]
        rotated_imag = base_wave_real[:, None, :] * mode_sin[None] + base_wave_imag[:, None, :] * mode_cos[None]
        wave_real = mode_scale * rotated_real
        wave_imag = mode_scale * rotated_imag
        wave_norm = np.sqrt(np.sum(wave_real**2 + wave_imag**2, axis=(1, 2), keepdims=True))
        wave_norm = np.where(wave_norm == 0.0, 1.0, wave_norm)
        wave_real = wave_real / wave_norm
        wave_imag = wave_imag / wave_norm

        mode_energy = mode_weight * np.maximum(route_logits, 0.0)

        if flags.disable_tanh_public:
            pub_input = np.clip(mode_weight + 0.35 * mode_energy, -3, 3)
        else:
            pub_input = np.tanh(mode_weight + 0.35 * mode_energy)
        public = _safe_normalize(pub_input @ public_mix)
        public = _mask_public_observation(
            "keyed_wave_superpose_embedding_v0", secret_key, public, public_mask, public_chunk,
        )
        return {
            "public": public.astype(np.float32),
            "base_wave_real": base_wave_real.astype(np.float32),
            "base_wave_imag": base_wave_imag.astype(np.float32),
            "wave_real": wave_real.astype(np.float32),
            "wave_imag": wave_imag.astype(np.float32),
            "mode_weight": mode_weight.astype(np.float32),
            "mode_energy": mode_energy.astype(np.float32),
        }

    def _score(doc_state, query_state):
        base_overlap = query_state["base_wave_real"] @ doc_state["base_wave_real"].T
        base_overlap = base_overlap + query_state["base_wave_imag"] @ doc_state["base_wave_imag"].T

        component_overlap = np.einsum("qmh,dmh->qdm", query_state["wave_real"], doc_state["wave_real"], dtype=np.float32)
        component_overlap = component_overlap + np.einsum("qmh,dmh->qdm", query_state["wave_imag"], doc_state["wave_imag"], dtype=np.float32)

        joint_weight = np.einsum("qm,dm->qdm", query_state["mode_weight"], doc_state["mode_weight"], dtype=np.float32)
        if flags.disable_sqrt_mode_gate:
            mode_gate = np.maximum(joint_weight, 0.0)
        else:
            mode_gate = np.sqrt(np.maximum(joint_weight, 0.0))

        joint_energy = np.einsum("qm,dm->qdm", query_state["mode_energy"], doc_state["mode_energy"], dtype=np.float32)
        if flags.disable_sqrt_energy_gate:
            energy_gate = np.maximum(joint_energy, 0.0)
        else:
            energy_gate = np.sqrt(np.maximum(joint_energy, 0.0))

        if flags.disable_component_sq:
            coherence = np.abs(component_overlap) * (0.30 + 0.70 * mode_gate)
        else:
            coherence = (component_overlap**2) * (0.30 + 0.70 * mode_gate)

        coherence_score = np.mean(coherence, axis=2, dtype=np.float32)
        energy_score = np.mean(energy_gate, axis=2, dtype=np.float32)
        mode_support = query_state["mode_weight"] @ doc_state["mode_weight"].T
        superposed_score = 0.80 * coherence_score + 0.10 * energy_score + 0.10 * mode_support

        if flags.disable_base_sq:
            return coherence_weight * superposed_score + (1.0 - coherence_weight) * np.abs(base_overlap)
        return coherence_weight * superposed_score + (1.0 - coherence_weight) * (base_overlap**2)

    return EmbeddingStateMethod(
        method_name=f"ablated_{flags.label()}",
        family="ablation",
        params={"dim": dim},
        encode_docs=lambda x: _encode(x, doc_top_k, route_scale),
        encode_queries=lambda x: _encode(x, query_top_k, route_scale * collapse_gain),
        score=_score,
    )


def measure_security_ratio(
    method_ok: EmbeddingStateMethod,
    method_wrong: EmbeddingStateMethod,
    plaintext: np.ndarray,
    n_queries: int = 20,
    k: int = 10,
) -> Dict[str, float]:
    doc_state = method_ok.encode_docs(plaintext)
    rng = np.random.default_rng(123)
    qi = rng.choice(plaintext.shape[0], size=min(n_queries, plaintext.shape[0]), replace=False)
    q_ok = method_ok.encode_queries(plaintext[qi])
    q_wrong = method_wrong.encode_queries(plaintext[qi])

    scores_ok = method_ok.score(doc_state, q_ok)
    scores_wrong = method_ok.score(doc_state, q_wrong)

    mag_ok = float(np.mean(np.sort(scores_ok, axis=1)[:, -k:]))
    mag_wrong = float(np.mean(np.sort(np.abs(scores_wrong), axis=1)[:, -k:]))
    ratio = mag_ok / max(1e-12, mag_wrong)

    plain_sim = cosine_similarity(plaintext[qi], plaintext)
    recall_ok, recall_wrong = [], []
    for i in range(len(qi)):
        true_nbrs = set(np.argpartition(-plain_sim[i], k)[:k].tolist()) - {int(qi[i])}
        if not true_nbrs:
            continue
        ok_nbrs = set(np.argpartition(-scores_ok[i], k)[:k].tolist()) - {int(qi[i])}
        wr_nbrs = set(np.argpartition(-scores_wrong[i], k)[:k].tolist()) - {int(qi[i])}
        recall_ok.append(len(true_nbrs & ok_nbrs) / len(true_nbrs))
        recall_wrong.append(len(true_nbrs & wr_nbrs) / len(true_nbrs))

    return {
        "mag_ok": round(mag_ok, 6),
        "mag_wrong": round(mag_wrong, 6),
        "ratio": round(ratio, 2),
        "recall_ok": round(float(np.mean(recall_ok)), 4) if recall_ok else 0.0,
        "recall_wrong": round(float(np.mean(recall_wrong)), 4) if recall_wrong else 0.0,
    }


def generate_docs(n_docs: int, dim: int, seed: int = 42):
    ds = load_dataset("ag_news", split="train")
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(ds), size=min(n_docs, len(ds)), replace=False)
    texts = [ds[int(i)]["text"] for i in indices]
    vec = TfidfVectorizer(max_features=dim, stop_words="english")
    return _safe_normalize(vec.fit_transform(texts).toarray().astype(np.float32))


def run_ablation(n_docs: int = 200, dims: List[int] = None, seed: int = 42):
    if dims is None:
        dims = [64, 128, 256, 384, 512, 768, 1024]

    # Einzelablationen: baseline + 8 singles + alle-aus
    configs: List[Tuple[str, AblationFlags]] = [("baseline", AblationFlags())]
    flag_names = [
        "tanh_amp", "topk", "comp_sq", "base_sq",
        "sqrt_gate", "sqrt_energy", "tanh_pub", "decoy",
    ]
    field_names = [
        "disable_tanh_amplitude", "disable_topk_routing",
        "disable_component_sq", "disable_base_sq",
        "disable_sqrt_mode_gate", "disable_sqrt_energy_gate",
        "disable_tanh_public", "disable_decoy_floor",
    ]
    for name, field in zip(flag_names, field_names):
        configs.append((f"only_{name}", AblationFlags(**{field: True})))
    configs.append(("all_disabled", AblationFlags.all_disabled()))

    results = []
    for dim in dims:
        print(f"\n--- dim={dim} ---")
        plaintext = generate_docs(n_docs, dim, seed)

        for label, flags in configs:
            t0 = time.perf_counter()
            method_ok = build_ablated_carrier(dim, "correct-key", flags, seed)
            method_wrong = build_ablated_carrier(dim, "wrong-key", flags, seed)
            metrics = measure_security_ratio(method_ok, method_wrong, plaintext)
            dt = time.perf_counter() - t0

            row = {"dim": dim, "ablation": label, **metrics, "elapsed_s": round(dt, 2)}
            results.append(row)
            print(f"  {label:<20} ratio={metrics['ratio']:>8.1f}  recall_ok={metrics['recall_ok']:.3f}  recall_wr={metrics['recall_wrong']:.3f}  {dt:.1f}s")

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nonlinearity Ablation")
    parser.add_argument("--n-docs", type=int, default=200)
    parser.add_argument("--dims", type=str, default="64,128,256,384,512,768,1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str,
                        default="experiments/privacy_mathlab/results/nonlinearity_ablation.json")
    args = parser.parse_args()
    dims = [int(d) for d in args.dims.split(",")]

    print("=" * 72)
    print("NONLINEARITY ABLATION")
    print("Frage: Welche Nichtlinearität erzeugt den 50×-Floor?")
    print("=" * 72)

    results = run_ablation(n_docs=args.n_docs, dims=dims, seed=args.seed)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"experiment": "nonlinearity_ablation", "results": results}, f, indent=2)
    print(f"\nGespeichert: {args.output}")

    # Zusammenfassung: Ranking der Ablationen bei d=384
    print(f"\n{'='*72}")
    print("RANKING bei d=384 (welche Nichtlinearität senkt den Ratio am meisten):")
    d384 = [r for r in results if r["dim"] == 384]
    baseline_ratio = next((r["ratio"] for r in d384 if r["ablation"] == "baseline"), 50)
    for r in sorted(d384, key=lambda x: x["ratio"], reverse=True):
        delta = r["ratio"] - baseline_ratio
        print(f"  {r['ablation']:<20} ratio={r['ratio']:>8.1f}  (Δ={delta:>+8.1f})  recall={r['recall_ok']:.3f}")


if __name__ == "__main__":
    main()
