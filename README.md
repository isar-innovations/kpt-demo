# Keyed Phase Transform (KPT)

Access control on meaning for vector databases.

KPT makes semantic search key-dependent. With the correct key, retrieval works normally. With any other key, the score space collapses to noise. No decryption at query time.

Repository for the paper "Keyed Phase Transform for Private Vector Retrieval."

## Results

| Metric | Value |
|---|---|
| Recall@10 (3 datasets, 6k docs) | >= 0.906 |
| Recall@10 (AG News, 100k docs) | 0.993 |
| Public-layer pair correlation | <= 0.029 |
| Scrambled wave pair correlation | < 0.01 |
| Encode overhead | 0.05ms/doc |
| Geometric key depth | 623 bits |
| Adversarial attacks defeated | 5/5 |
| Cross-key collision tests | 0/9,900 |
| Key-gated clustering ARI | 1.0 correct, 0.04 wrong |
| Embedding models tested | MiniLM (384d), mxbai (1024d), bge-m3 (1024d) |
| Quantization | int16 works (half storage), int8 breaks |

## Quick Start

```bash
pip install -r requirements.txt

# Interactive demo: store and search with passwords
python kpt_vault.py

# Multi-user demo: Alice vs Bob vs Attacker on AG News
python demo_multi_user.py

# Collision test: 100 keys, same documents
python collision_test.py --keys 100 --docs 500
```

## Experiments

Each script reproduces one section of the paper.

### Core Method and Ablation (Section 3 + 4.2)

```bash
python nonlinearity_ablation.py
```

8-component ablation across 7 dimensions. Identifies phase coherence as sole isolation mechanism. decoy_floor is the primary limiter (15,905x at d=1024 without). Pure phase rotation: 3,753x at d=1024.

### DCPE/SAP Comparison (Section 4.3)

```bash
python embedding_lab.py \
  --plan embedding_lab_dcpe_sap_agnews_plan.json \
  --dataset-kind hf_text --hf-dataset ag_news \
  --vectorizer sbert --docs 6000 --queries 1200 --seed-runs 3
```

KPT recall 0.991, pair_corr 0.029 vs DCPE recall 0.393, pair_corr 0.763 across 3 datasets.

### Key Sensitivity / Avalanche (Section 4.4)

```bash
python phase_transition_attack.py
```

1 character difference produces complete score collapse. No correlation between edit distance and attack success.

### Red-Team Attacks (Section 5.1-5.3)

```bash
python attack_evaluation.py
```

- Score-oracle: Gram correlation 0.024
- Known-plaintext: cosine 0.228 (predicted 0.4-0.55)
- Statistical leakage: pair_corr 0.00 at all N on public layer

### Advanced Attacks (Section 5.4-5.5)

```bash
python advanced_attacks.py
```

- CPA: MLP cosine 0.005 on scrambled waves
- Gradient key recovery: cross-key score 0.012
- Higher-order statistics: 3-point correlation -0.002
- Collusion (2 users): correlation 0.002 (was 0.944 before scrambling)

### Production Scale (Section 4.7)

```bash
python benchmark_scale.py --n-docs 100000 --n-queries 10000
```

Recall 0.993 at 100k docs, pair_corr 0.002 under controlled sequential sampling. GPU recommended for sbert encoding.

### Collision Test

```bash
python collision_test.py --keys 100 --docs 500 --queries 50
```

0 collisions above 0.05 across 9,900 cross-key tests. Worst case 0.025, gap 119 sigma.

### Formal Proofs

```bash
python proof_scramble_security.py
python proof_wrong_key_concentration.py
```

Proof 1: Per-document scramble indistinguishability. E=0, Var=1/d, Monte Carlo verified across 5 dimensions.
Proof 2: Wrong-key score concentration. 10k keys, McDiarmid bound P(>0.05) <= 10^-18.

### Interactive Vault

```bash
python kpt_vault.py
```

Store documents with a password, search with a password. Text encrypted with AES-256-GCM, vectors encrypted with KPT. Wrong password: text unreadable, search returns noise.

Commands: `add`, `search`, `inspect` (attacker view), `attack` (automated attacks), `dump`, `stats`.

## Architecture

```
Input: x (any embedding, any dimension) + Key K (any string, RSA key, password)
  |
  +-- Carrier Wave: orthogonal projection, tanh amplitude, phase from carrier
  +-- Mode Decomposition: 8 modes, key-dependent sparse top-k softmax routing
  +-- Superposition: key-dependent phase shifts, normalize
  +-- Per-Doc Scramble: HKDF(key, mode_weight) derived permutation + sign flip
  +-- Public Layer: destructured projection (intentionally lossy)
  +-- Score: unscramble, then coherence + base overlap (see paper for full formula)

Correct key:  constructive interference, Score ~0.557
Wrong key:    destructive interference, Score ~0.012
At rest:      scrambled waves pair_corr < 0.01, routing components ~0.07
```

Key derivation: HKDF (RFC 5869) with SHA-512. 512-bit key material per context. Supports arbitrary key lengths. 256-bit post-quantum security.

Vectorizer-agnostic: tested on all-MiniLM-L6-v2 (384d), mxbai-embed-large (1024d), bge-m3 (1024d).

## What KPT Is

A key-gated semantic layer for vector databases. The same stored representation supports authorized search with the correct key and returns noise without it. Beyond search: clustering, deduplication, recommendation, and anomaly detection are all key-gated through the same mechanism.

Not yet formally proven. Two analytical theorems (sinc decay, geometric key depth), two verified proofs (scramble indistinguishability, wrong-key concentration), five defeated attacks. The formal security reduction to a known hard problem is open work.

## Citation

```
@article{wagensonner2026kpt,
  title={Keyed Phase Transform for Private Vector Retrieval},
  author={Wagensonner, Nino},
  year={2026}
}
```

## License

Apache-2.0
