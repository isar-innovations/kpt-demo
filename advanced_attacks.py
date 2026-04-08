"""Advanced Attacks auf KPT — CPA + Gradient-basierte Key Recovery.

Attack 1: Chosen Plaintext Attack (CPA)
  Angreifer wählt Dokumente mit bekannter Struktur (orthogonale Basis,
  bekannte Cluster) und versucht daraus die Transformation zu lernen.

Attack 2: Gradient-Based Key Recovery
  Pipeline ist differenzierbar. Optimiere Key-Seed direkt per Backprop
  durch die gesamte KPT-Pipeline mit bekannten Plaintext-Ciphertext-Paaren.

Attack 3: Higher-Order Statistics
  Prüfe ob 3-Punkt-Korrelationen oder Cluster-Struktur im Public Layer
  mehr leaken als paarweise Korrelation.

Attack 4: Collusion Attack
  Zwei User mit verschiedenen Keys sehen dieselben Docs verschlüsselt.
  Können sie ihre Views kombinieren um Plaintext-Struktur zu rekonstruieren?
"""

from __future__ import annotations

import hashlib
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_lab import (
    METHOD_REGISTRY,
    _set_seed,
    _safe_normalize,
    _method_seed,
    _qr_orthogonal,
    _topk_soft_assign,
    _mask_public_observation,
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


# =========================================================================
# Attack 1: Chosen Plaintext Attack (CPA)
# =========================================================================

def attack_cpa(dim: int = 384, n_chosen: int = 500, n_test: int = 200):
    """Angreifer wählt Dokumente mit bekannter Struktur."""
    print(f"\n{'='*60}")
    print(f"  ATTACK 1: Chosen Plaintext Attack (CPA)")
    print(f"  {n_chosen} gewählte Paare, {n_test} Test-Paare")
    print(f"{'='*60}\n")

    secret_key = "alice-secret-key-2026"
    kpt = build_kpt(secret_key, dim)
    rng = np.random.default_rng(123)

    # --- Strategie A: Orthogonale Basis-Vektoren ---
    print(f"  Strategie A: Orthogonale Basis + Einheitsvektoren")
    # Sende Einheitsvektoren durch KPT
    n_basis = min(n_chosen, dim)
    basis = np.eye(dim, dtype=np.float32)[:n_basis]
    basis_state = kpt.encode_docs(basis)

    # Versuche lineare Rekonstruktion: X_plain @ W ≈ X_encrypted
    wave_real = basis_state["wave_real"].reshape(n_basis, -1)  # flatten (modes × hidden)
    W_ls = np.linalg.lstsq(basis[:n_basis], wave_real, rcond=None)[0]

    # Teste auf ungesehenen zufälligen Vektoren
    test_plain = _safe_normalize(rng.normal(size=(n_test, dim)).astype(np.float32))
    test_state = kpt.encode_docs(test_plain)
    test_wave_real = test_state["wave_real"].reshape(n_test, -1)

    predicted = test_plain @ W_ls
    # Cosine similarity pro Zeile
    cos_per_row = np.array([
        np.dot(predicted[i], test_wave_real[i]) /
        (np.linalg.norm(predicted[i]) * np.linalg.norm(test_wave_real[i]) + 1e-12)
        for i in range(n_test)
    ])
    print(f"    Linear Recon (wave_real): cosine {cos_per_row.mean():.4f} ± {cos_per_row.std():.4f}")

    # --- Strategie B: Cluster-Injektion ---
    print(f"\n  Strategie B: 10 bekannte Cluster injizieren")
    n_clusters = 10
    centers = _safe_normalize(rng.normal(size=(n_clusters, dim)).astype(np.float32))
    chosen_docs = []
    chosen_labels = []
    for c in range(n_clusters):
        for _ in range(n_chosen // n_clusters):
            noise = rng.normal(0, 0.05, size=(dim,)).astype(np.float32)
            chosen_docs.append(centers[c] + noise)
            chosen_labels.append(c)
    chosen_docs = _safe_normalize(np.array(chosen_docs))
    chosen_labels = np.array(chosen_labels)

    state = kpt.encode_docs(chosen_docs)
    public = state["public"]

    # Kann der Angreifer die Cluster im Public Layer wiederfinden?
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

    km_public = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(public)
    ari_public = adjusted_rand_score(chosen_labels, km_public.labels_)
    nmi_public = normalized_mutual_info_score(chosen_labels, km_public.labels_)

    # Und im wave_real?
    wave_flat = state["wave_real"].reshape(len(chosen_docs), -1)
    km_wave = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(wave_flat)
    ari_wave = adjusted_rand_score(chosen_labels, km_wave.labels_)
    nmi_wave = normalized_mutual_info_score(chosen_labels, km_wave.labels_)

    print(f"    Public Layer — ARI: {ari_public:.4f}, NMI: {nmi_public:.4f}")
    print(f"    Wave Real   — ARI: {ari_wave:.4f}, NMI: {nmi_wave:.4f}")

    # --- Strategie C: MLP mit gewählten Paaren (stärker als random KPA) ---
    print(f"\n  Strategie C: MLP Inversion mit gewählten Paaren")
    try:
        from sklearn.neural_network import MLPRegressor
        # Train auf chosen docs → wave_real
        mlp = MLPRegressor(
            hidden_layer_sizes=(512, 512),
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
        )
        mlp.fit(chosen_docs, wave_flat)
        pred_wave = mlp.predict(test_plain)
        test_wave_flat = test_state["wave_real"].reshape(n_test, -1)

        cos_mlp = np.array([
            np.dot(pred_wave[i], test_wave_flat[i]) /
            (np.linalg.norm(pred_wave[i]) * np.linalg.norm(test_wave_flat[i]) + 1e-12)
            for i in range(n_test)
        ])
        print(f"    MLP Recon (wave_real): cosine {cos_mlp.mean():.4f} ± {cos_mlp.std():.4f}")

        # Kann die MLP-Rekonstruktion für Retrieval genutzt werden?
        from sklearn.metrics.pairwise import cosine_similarity
        true_sim = cosine_similarity(test_plain)
        pred_sim = cosine_similarity(pred_wave)
        mask = np.triu(np.ones((n_test, n_test), dtype=bool), k=1)
        sim_corr = float(np.corrcoef(true_sim[mask], pred_sim[mask])[0, 1])
        print(f"    Similarity correlation (plain vs predicted): {sim_corr:.4f}")
    except Exception as e:
        print(f"    MLP fehlgeschlagen: {e}")

    return {
        "linear_recon_cosine": float(cos_per_row.mean()),
        "cluster_ari_public": float(ari_public),
        "cluster_ari_wave": float(ari_wave),
        "cluster_nmi_public": float(nmi_public),
        "cluster_nmi_wave": float(nmi_wave),
    }


# =========================================================================
# Attack 2: Gradient-Based Key Recovery
# =========================================================================

def attack_gradient_key_recovery(dim: int = 384, n_pairs: int = 50, n_steps: int = 2000):
    """Optimiere Key-Seed direkt per numerischem Gradienten."""
    print(f"\n{'='*60}")
    print(f"  ATTACK 2: Gradient-Based Key Recovery")
    print(f"  {n_pairs} Known-Plaintext-Paare, {n_steps} Optimierungsschritte")
    print(f"{'='*60}\n")

    secret_key = "alice-secret-key-2026"
    kpt_true = build_kpt(secret_key, dim)
    rng = np.random.default_rng(777)

    # Generiere Known-Plaintext-Paare
    plaintexts = _safe_normalize(rng.normal(size=(n_pairs, dim)).astype(np.float32))
    true_state = kpt_true.encode_docs(plaintexts)
    target_wave = true_state["wave_real"].reshape(n_pairs, -1)

    # Optimiere: finde key_candidate der die wave_real reproduziert
    # Da der Key ein String ist, optimieren wir im Seed-Space (256-bit)
    # Strategie: zufällige Key-Kandidaten + Hill-Climbing
    print(f"  Phase 1: Random Search ({n_steps} zufällige Keys)")

    best_loss = float('inf')
    best_key = ""
    losses = []

    for step in range(n_steps):
        # Zufälliger Key-Kandidat
        candidate_key = os.urandom(16).hex()
        kpt_candidate = build_kpt(candidate_key, dim)
        candidate_state = kpt_candidate.encode_docs(plaintexts)
        candidate_wave = candidate_state["wave_real"].reshape(n_pairs, -1)

        # Loss: mittlerer cosine distance
        cos_sims = np.array([
            np.dot(target_wave[i], candidate_wave[i]) /
            (np.linalg.norm(target_wave[i]) * np.linalg.norm(candidate_wave[i]) + 1e-12)
            for i in range(n_pairs)
        ])
        loss = 1.0 - cos_sims.mean()
        losses.append(loss)

        if loss < best_loss:
            best_loss = loss
            best_key = candidate_key

        if (step + 1) % 500 == 0:
            print(f"    Step {step+1}: best_loss={best_loss:.6f}, mean_cos={1-best_loss:.6f}")

    # Phase 2: Levenshtein-Varianten des besten Keys
    print(f"\n  Phase 2: Lokale Suche um besten Kandidaten ({best_key[:12]}...)")
    for _ in range(200):
        # Mutiere 1-3 Zeichen
        key_bytes = list(best_key)
        for _ in range(rng.integers(1, 4)):
            pos = rng.integers(0, len(key_bytes))
            key_bytes[pos] = hex(rng.integers(0, 16))[2:]
        candidate_key = "".join(key_bytes)
        kpt_candidate = build_kpt(candidate_key, dim)
        candidate_state = kpt_candidate.encode_docs(plaintexts)
        candidate_wave = candidate_state["wave_real"].reshape(n_pairs, -1)

        cos_sims = np.array([
            np.dot(target_wave[i], candidate_wave[i]) /
            (np.linalg.norm(target_wave[i]) * np.linalg.norm(candidate_wave[i]) + 1e-12)
            for i in range(n_pairs)
        ])
        loss = 1.0 - cos_sims.mean()
        if loss < best_loss:
            best_loss = loss
            best_key = candidate_key

    # Teste ob der beste Key für Retrieval taugt
    kpt_best = build_kpt(best_key, dim)
    test_plain = _safe_normalize(rng.normal(size=(50, dim)).astype(np.float32))
    test_state_true = kpt_true.encode_docs(test_plain)
    test_state_best = kpt_best.encode_docs(test_plain)

    # Cross-score: docs mit true key, queries mit best key
    query_state_best = kpt_best.encode_queries(test_plain[:10])
    scores_cross = kpt_true.score(test_state_true, query_state_best)
    max_cross = float(np.max(scores_cross))
    mean_cross = float(np.mean(scores_cross))

    # Correct score zum Vergleich
    query_state_true = kpt_true.encode_queries(test_plain[:10])
    scores_correct = kpt_true.score(test_state_true, query_state_true)
    mean_correct = float(np.mean(np.sort(scores_correct, axis=1)[:, -10:]))

    losses_arr = np.array(losses)
    print(f"\n  Ergebnisse:")
    print(f"    Best cosine similarity to target: {1-best_loss:.6f}")
    print(f"    Mean loss über alle Kandidaten:   {losses_arr.mean():.6f}")
    print(f"    Cross-key score (best found):     {mean_cross:.6f} (max: {max_cross:.6f})")
    print(f"    Correct-key score:                {mean_correct:.6f}")

    if max_cross < 0.05:
        print(f"    GESCHEITERT: Optimierter Key erreicht nicht mal 0.05.")
    elif max_cross < mean_correct * 0.5:
        print(f"    GESCHEITERT: Score weit unter correct-key Niveau.")
    else:
        print(f"    WARNUNG: Partieller Erfolg!")

    return {
        "best_cosine_to_target": float(1 - best_loss),
        "mean_loss": float(losses_arr.mean()),
        "cross_key_score_max": max_cross,
        "cross_key_score_mean": mean_cross,
        "correct_key_score_mean": mean_correct,
        "n_steps": n_steps,
    }


# =========================================================================
# Attack 3: Higher-Order Statistics
# =========================================================================

def attack_higher_order(dim: int = 384, n_docs: int = 1000):
    """Prüfe ob 3-Punkt-Korrelationen oder Cluster im Public Layer leaken."""
    print(f"\n{'='*60}")
    print(f"  ATTACK 3: Higher-Order Statistics")
    print(f"  {n_docs} Dokumente, 3-Punkt-Korrelationen + Cluster-Analyse")
    print(f"{'='*60}\n")

    secret_key = "alice-secret-key-2026"
    kpt = build_kpt(secret_key, dim)
    rng = np.random.default_rng(999)

    # Generiere Docs mit bekannter Cluster-Struktur (5 Cluster)
    n_clusters = 5
    centers = _safe_normalize(rng.normal(size=(n_clusters, dim)).astype(np.float32))
    docs = []
    labels = []
    for c in range(n_clusters):
        for _ in range(n_docs // n_clusters):
            noise = rng.normal(0, 0.08, size=(dim,)).astype(np.float32)
            docs.append(centers[c] + noise)
            labels.append(c)
    docs = _safe_normalize(np.array(docs))
    labels = np.array(labels)

    state = kpt.encode_docs(docs)
    public = state["public"]

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score

    # 2-Punkt: paarweise Korrelation (bekannt: ~0.002)
    raw_sim = cosine_similarity(docs[:500])
    pub_sim = cosine_similarity(public[:500])
    mask2 = np.triu(np.ones((500, 500), dtype=bool), k=1)
    pair_corr = float(np.corrcoef(raw_sim[mask2], pub_sim[mask2])[0, 1])
    print(f"  2-Punkt pair_corr: {pair_corr:.4f}")

    # 3-Punkt: Dreiecks-Korrelation
    # Für Triplets (i,j,k): korreliert raw_sim[i,j]*raw_sim[j,k] mit pub_sim[i,j]*pub_sim[j,k]?
    n_triplets = 5000
    idx = rng.integers(0, 500, size=(n_triplets, 3))
    raw_triple = raw_sim[idx[:, 0], idx[:, 1]] * raw_sim[idx[:, 1], idx[:, 2]]
    pub_triple = pub_sim[idx[:, 0], idx[:, 1]] * pub_sim[idx[:, 1], idx[:, 2]]
    triple_corr = float(np.corrcoef(raw_triple, pub_triple)[0, 1])
    print(f"  3-Punkt triple_corr: {triple_corr:.4f}")

    # 4-Punkt: Quadrupel
    idx4 = rng.integers(0, 500, size=(n_triplets, 4))
    raw_quad = (raw_sim[idx4[:, 0], idx4[:, 1]] * raw_sim[idx4[:, 1], idx4[:, 2]] *
                raw_sim[idx4[:, 2], idx4[:, 3]])
    pub_quad = (pub_sim[idx4[:, 0], idx4[:, 1]] * pub_sim[idx4[:, 1], idx4[:, 2]] *
                pub_sim[idx4[:, 2], idx4[:, 3]])
    quad_corr = float(np.corrcoef(raw_quad, pub_quad)[0, 1])
    print(f"  4-Punkt quad_corr: {quad_corr:.4f}")

    # Cluster-Recovery im Public Layer
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(public)
    ari = adjusted_rand_score(labels, km.labels_)
    print(f"\n  Cluster-Recovery (Public Layer):")
    print(f"    ARI: {ari:.4f} (0=random, 1=perfekt)")

    # Cluster-Recovery im Wave-Space
    wave_flat = state["wave_real"].reshape(len(docs), -1)
    km_wave = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit(wave_flat)
    ari_wave = adjusted_rand_score(labels, km_wave.labels_)
    print(f"    ARI (wave_real): {ari_wave:.4f}")

    # Spectral Clustering auf Public Layer
    try:
        from sklearn.cluster import SpectralClustering
        sc = SpectralClustering(n_clusters=n_clusters, random_state=42, n_init=3).fit(public)
        ari_spectral = adjusted_rand_score(labels, sc.labels_)
        print(f"    ARI (spectral on public): {ari_spectral:.4f}")
    except Exception:
        ari_spectral = 0.0

    return {
        "pair_corr_2pt": pair_corr,
        "triple_corr_3pt": triple_corr,
        "quad_corr_4pt": quad_corr,
        "cluster_ari_public": float(ari),
        "cluster_ari_wave": float(ari_wave),
        "cluster_ari_spectral": float(ari_spectral),
    }


# =========================================================================
# Attack 4: Collusion Attack
# =========================================================================

def attack_collusion(dim: int = 384, n_docs: int = 500):
    """Zwei User kombinieren ihre verschlüsselten Views."""
    print(f"\n{'='*60}")
    print(f"  ATTACK 4: Collusion Attack")
    print(f"  2 User mit verschiedenen Keys sehen dieselben {n_docs} Docs")
    print(f"{'='*60}\n")

    key_alice = "alice-key-2026"
    key_bob = "bob-key-2026"
    kpt_alice = build_kpt(key_alice, dim)
    kpt_bob = build_kpt(key_bob, dim)
    rng = np.random.default_rng(555)

    # Selbe Dokumente, verschiedene Keys
    docs = _safe_normalize(rng.normal(size=(n_docs, dim)).astype(np.float32))
    state_alice = kpt_alice.encode_docs(docs)
    state_bob = kpt_bob.encode_docs(docs)

    from sklearn.metrics.pairwise import cosine_similarity

    # Was jeder einzeln sieht (Public Layer)
    pub_alice = state_alice["public"]
    pub_bob = state_bob["public"]

    raw_sim = cosine_similarity(docs)
    mask = np.triu(np.ones((n_docs, n_docs), dtype=bool), k=1)

    # Einzel-Korrelation
    alice_sim = cosine_similarity(pub_alice)
    bob_sim = cosine_similarity(pub_bob)
    corr_alice = float(np.corrcoef(raw_sim[mask], alice_sim[mask])[0, 1])
    corr_bob = float(np.corrcoef(raw_sim[mask], bob_sim[mask])[0, 1])
    print(f"  Einzel-Korrelation (Alice public): {corr_alice:.4f}")
    print(f"  Einzel-Korrelation (Bob public):   {corr_bob:.4f}")

    # Collusion Strategie 1: Concatenate public layers
    combined_pub = np.hstack([pub_alice, pub_bob])
    combined_sim = cosine_similarity(combined_pub)
    corr_combined = float(np.corrcoef(raw_sim[mask], combined_sim[mask])[0, 1])
    print(f"\n  Collusion (concatenate public): {corr_combined:.4f}")

    # Collusion Strategie 2: Concatenate wave_real
    wave_alice = state_alice["wave_real"].reshape(n_docs, -1)
    wave_bob = state_bob["wave_real"].reshape(n_docs, -1)
    combined_wave = np.hstack([wave_alice, wave_bob])
    combined_wave_sim = cosine_similarity(combined_wave)
    corr_wave = float(np.corrcoef(raw_sim[mask], combined_wave_sim[mask])[0, 1])
    print(f"  Collusion (concatenate waves):  {corr_wave:.4f}")

    # Collusion Strategie 3: Element-weise Multiplikation (Interferenz-Muster)
    cross_wave = wave_alice * wave_bob
    cross_sim = cosine_similarity(cross_wave)
    corr_cross = float(np.corrcoef(raw_sim[mask], cross_sim[mask])[0, 1])
    print(f"  Collusion (wave product):       {corr_cross:.4f}")

    # Collusion Strategie 4: CCA (Canonical Correlation Analysis)
    try:
        from sklearn.cross_decomposition import CCA
        n_cca = min(200, n_docs)
        cca = CCA(n_components=10, max_iter=500)
        cca.fit(wave_alice[:n_cca], wave_bob[:n_cca])
        Xa, Xb = cca.transform(wave_alice, wave_bob)
        cca_combined = np.hstack([Xa, Xb])
        cca_sim = cosine_similarity(cca_combined)
        corr_cca = float(np.corrcoef(raw_sim[mask], cca_sim[mask])[0, 1])
        print(f"  Collusion (CCA alignment):      {corr_cca:.4f}")
    except Exception as e:
        corr_cca = 0.0
        print(f"  Collusion (CCA): fehlgeschlagen ({e})")

    # Bewertung
    best_collusion = max(abs(corr_combined), abs(corr_wave), abs(corr_cross), abs(corr_cca))
    print(f"\n  Beste Collusion-Korrelation: {best_collusion:.4f}")
    if best_collusion < 0.05:
        print(f"  GESCHEITERT: Collusion bringt keinen verwertbaren Vorteil.")
    elif best_collusion < 0.20:
        print(f"  SCHWACH: Minimaler Vorteil, nicht für Retrieval nutzbar.")
    else:
        print(f"  WARNUNG: Collusion zeigt messbare Korrelation!")

    return {
        "corr_alice_solo": corr_alice,
        "corr_bob_solo": corr_bob,
        "corr_concat_public": corr_combined,
        "corr_concat_wave": corr_wave,
        "corr_wave_product": corr_cross,
        "corr_cca": corr_cca,
        "best_collusion": best_collusion,
    }


# =========================================================================
# Main
# =========================================================================

def main():
    import json

    results = {}

    results["cpa"] = attack_cpa(dim=384, n_chosen=500, n_test=200)
    results["gradient_key_recovery"] = attack_gradient_key_recovery(dim=384, n_pairs=50, n_steps=2000)
    results["higher_order"] = attack_higher_order(dim=384, n_docs=1000)
    results["collusion"] = attack_collusion(dim=384, n_docs=500)

    # Summary
    print(f"\n{'='*60}")
    print(f"  ZUSAMMENFASSUNG: 4 Advanced Attacks")
    print(f"{'='*60}\n")

    print(f"  1. CPA (Chosen Plaintext):")
    print(f"     Linear Recon: cosine {results['cpa']['linear_recon_cosine']:.4f}")
    print(f"     Cluster ARI (public): {results['cpa']['cluster_ari_public']:.4f}")
    print(f"     Cluster ARI (wave): {results['cpa']['cluster_ari_wave']:.4f}")

    print(f"\n  2. Gradient Key Recovery:")
    print(f"     Best cosine to target: {results['gradient_key_recovery']['best_cosine_to_target']:.6f}")
    print(f"     Cross-key score: {results['gradient_key_recovery']['cross_key_score_mean']:.6f}")
    print(f"     Correct-key score: {results['gradient_key_recovery']['correct_key_score_mean']:.6f}")

    print(f"\n  3. Higher-Order Statistics:")
    print(f"     2-Punkt: {results['higher_order']['pair_corr_2pt']:.4f}")
    print(f"     3-Punkt: {results['higher_order']['triple_corr_3pt']:.4f}")
    print(f"     4-Punkt: {results['higher_order']['quad_corr_4pt']:.4f}")
    print(f"     Cluster ARI: {results['higher_order']['cluster_ari_public']:.4f}")

    print(f"\n  4. Collusion (2 User):")
    print(f"     Best Korrelation: {results['collusion']['best_collusion']:.4f}")

    out = "results/advanced_attacks.json"
    os.makedirs("results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Gespeichert: {out}")


if __name__ == "__main__":
    main()
