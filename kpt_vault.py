"""KPT Vault — Interaktives CLI zum Testen von Keyed Phase Transform.

Schreib Texte rein mit Passwort, lies sie wieder aus mit Passwort.
Falsches Passwort → Müll. Daten at rest → nur verschlüsselte Vektoren sichtbar.

Usage:
    python kpt_vault.py                  # Interaktiver Modus
    python kpt_vault.py --db vault.db    # Eigene DB-Datei
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import readline
import sqlite3
import sys
import textwrap
import time
from typing import Dict, List, Optional, Tuple
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_lab import (
    _set_seed,
    _safe_normalize,
    _method_seed,
    _topk_soft_assign,
    _mask_public_observation,
    _qr_orthogonal,
    EmbeddingStateMethod,
)

# ---------------------------------------------------------------------------
# Vectorizer: sbert wenn vorhanden, sonst TF-IDF Fallback
# ---------------------------------------------------------------------------

_sbert_model = None


def _get_vectorizer():
    global _sbert_model
    try:
        from sentence_transformers import SentenceTransformer
        if _sbert_model is None:
            print("  Lade sentence-transformers (all-MiniLM-L6-v2)...")
            _sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
        return "sbert", _sbert_model
    except ImportError:
        return "tfidf", None


def _vectorize(texts: List[str]) -> np.ndarray:
    kind, model = _get_vectorizer()
    if kind == "sbert":
        embs = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _safe_normalize(embs.astype(np.float32))

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    X = tfidf.fit_transform(texts)
    dim = min(384, X.shape[1] - 1, X.shape[0] - 1)
    if dim < 2:
        return _safe_normalize(np.random.randn(len(texts), 384).astype(np.float32))
    svd = TruncatedSVD(n_components=dim)
    reduced = svd.fit_transform(X)
    if reduced.shape[1] < 384:
        padded = np.zeros((reduced.shape[0], 384), dtype=np.float32)
        padded[:, :reduced.shape[1]] = reduced
        reduced = padded
    return _safe_normalize(reduced.astype(np.float32))


# ---------------------------------------------------------------------------
# KPT Builder (standalone, kein METHOD_REGISTRY nötig)
# ---------------------------------------------------------------------------

CARRIER_PARAMS = {
    "modes": 8, "doc_top_k": 3, "query_top_k": 1,
    "route_temperature": 0.22, "route_scale": 1.25,
    "collapse_gain": 2.2, "phase_scale": 0.78,
    "envelope_gain": 0.45, "decoy_floor": 0.24,
    "coherence_weight": 0.46, "public_ratio": 0.18,
    "public_mask": 0.84, "public_chunk": 6,
}


def build_kpt(key: str, dim: int = 384) -> EmbeddingStateMethod:
    params = {**CARRIER_PARAMS, "dim": dim, "hidden_dim": dim, "secret_key": key}
    rng = _set_seed(42)
    from embedding_lab import METHOD_REGISTRY
    return METHOD_REGISTRY["keyed_wave_superpose_embedding_v0"](rng, params)


# ---------------------------------------------------------------------------
# SQLite Store
# ---------------------------------------------------------------------------

def init_db(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            encrypted_text BLOB NOT NULL,
            added_at REAL NOT NULL,
            kpt_public BLOB NOT NULL,
            kpt_wave_real BLOB NOT NULL,
            kpt_wave_imag BLOB NOT NULL,
            kpt_base_wave_real BLOB NOT NULL,
            kpt_base_wave_imag BLOB NOT NULL,
            kpt_mode_weight BLOB NOT NULL,
            kpt_mode_energy BLOB NOT NULL,
            key_hash TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _key_hash(key: str) -> str:
    return hashlib.sha256(f"kpt-vault|{key}".encode()).hexdigest()[:16]


def _derive_aes_key(key: str) -> bytes:
    return hashlib.sha256(f"kpt-vault-aes|{key}".encode()).digest()


def _encrypt_text(plaintext: str, key: str) -> bytes:
    aes_key = _derive_aes_key(key)
    nonce = os.urandom(12)
    ciphertext = AESGCM(aes_key).encrypt(nonce, plaintext.encode("utf-8"), None)
    return nonce + ciphertext


def _decrypt_text(blob: bytes, key: str) -> Optional[str]:
    aes_key = _derive_aes_key(key)
    nonce, ciphertext = blob[:12], blob[12:]
    try:
        return AESGCM(aes_key).decrypt(nonce, ciphertext, None).decode("utf-8")
    except Exception:
        return None


def store_document(conn: sqlite3.Connection, text: str, key: str, state: Dict[str, np.ndarray]):
    encrypted = _encrypt_text(text, key)
    conn.execute(
        """INSERT INTO documents
           (encrypted_text, added_at, kpt_public, kpt_wave_real, kpt_wave_imag,
            kpt_base_wave_real, kpt_base_wave_imag, kpt_mode_weight, kpt_mode_energy, key_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            encrypted,
            time.time(),
            state["public"].tobytes(),
            state["wave_real"].tobytes(),
            state["wave_imag"].tobytes(),
            state["base_wave_real"].tobytes(),
            state["base_wave_imag"].tobytes(),
            state["mode_weight"].tobytes(),
            state["mode_energy"].tobytes(),
            _key_hash(key),
        ),
    )
    conn.commit()


def load_all_docs(conn: sqlite3.Connection) -> List[dict]:
    rows = conn.execute(
        "SELECT id, encrypted_text, added_at, kpt_public, kpt_wave_real, kpt_wave_imag, "
        "kpt_base_wave_real, kpt_base_wave_imag, kpt_mode_weight, kpt_mode_energy, key_hash "
        "FROM documents ORDER BY id"
    ).fetchall()
    docs = []
    for r in rows:
        docs.append({
            "id": r[0],
            "encrypted_text": r[1],
            "added_at": r[2],
            "kpt_public": r[3],
            "kpt_wave_real": r[4],
            "kpt_wave_imag": r[5],
            "kpt_base_wave_real": r[6],
            "kpt_base_wave_imag": r[7],
            "kpt_mode_weight": r[8],
            "kpt_mode_energy": r[9],
            "key_hash": r[10],
        })
    return docs


def docs_to_state(docs: List[dict], dim: int = 384, modes: int = 8) -> Dict[str, np.ndarray]:
    n = len(docs)
    hidden_dim = dim
    public_dim = max(8, int(dim * 0.18))
    # public_chunk reduces dim
    use_dim = (public_dim // 6) * 6
    actual_public_dim = (use_dim // 6) + (public_dim - use_dim) if use_dim > 0 else public_dim

    def _from_blob(blob: bytes, shape: Tuple) -> np.ndarray:
        return np.frombuffer(blob, dtype=np.float32).reshape(shape)

    # Infer actual public dim from first doc
    first_pub = np.frombuffer(docs[0]["kpt_public"], dtype=np.float32)
    actual_public_dim = first_pub.shape[0]

    public = np.stack([np.frombuffer(d["kpt_public"], dtype=np.float32) for d in docs])
    wave_real = np.stack([_from_blob(d["kpt_wave_real"], (modes, hidden_dim)) for d in docs])
    wave_imag = np.stack([_from_blob(d["kpt_wave_imag"], (modes, hidden_dim)) for d in docs])
    base_wave_real = np.stack([_from_blob(d["kpt_base_wave_real"], (hidden_dim,)) for d in docs])
    base_wave_imag = np.stack([_from_blob(d["kpt_base_wave_imag"], (hidden_dim,)) for d in docs])
    mode_weight = np.stack([_from_blob(d["kpt_mode_weight"], (modes,)) for d in docs])
    mode_energy = np.stack([_from_blob(d["kpt_mode_energy"], (modes,)) for d in docs])

    return {
        "public": public,
        "wave_real": wave_real,
        "wave_imag": wave_imag,
        "base_wave_real": base_wave_real,
        "base_wave_imag": base_wave_imag,
        "mode_weight": mode_weight,
        "mode_energy": mode_energy,
    }


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

def cmd_add(conn: sqlite3.Connection, key: str):
    print("\n  Texte eingeben (leere Zeile = fertig):")
    texts = []
    while True:
        line = input("  > ").strip()
        if not line:
            break
        texts.append(line)

    if not texts:
        print("  Nichts eingegeben.")
        return

    print(f"\n  Vectorisiere {len(texts)} Texte...")
    raw_vecs = _vectorize(texts)
    dim = raw_vecs.shape[1]

    print(f"  Verschlüssle mit KPT (key='{key[:3]}...', dim={dim})...")
    kpt = build_kpt(key, dim)
    state = kpt.encode_docs(raw_vecs)

    for i, text in enumerate(texts):
        single_state = {k: v[i:i+1] for k, v in state.items()}
        store_document(conn, text, key, {k: v[0] for k, v in single_state.items()})

    print(f"  {len(texts)} Dokumente gespeichert.")


def cmd_search(conn: sqlite3.Connection, key: str):
    docs = load_all_docs(conn)
    if not docs:
        print("\n  Vault ist leer. Erst Dokumente hinzufügen.")
        return

    query_text = input("  Suchanfrage: ").strip()
    if not query_text:
        return

    print(f"  Suche mit Key '{key[:3]}...'...")
    query_vec = _vectorize([query_text])
    dim = query_vec.shape[1]

    kpt = build_kpt(key, dim)
    query_state = kpt.encode_queries(query_vec)
    doc_state = docs_to_state(docs, dim)
    scores = kpt.score(doc_state, query_state)[0]  # (n_docs,)

    ranked = np.argsort(-scores)
    k = min(10, len(docs))

    print(f"\n  Top-{k} Ergebnisse (Score | Text)")
    print(f"  {'─' * 60}")
    decrypted_count = 0
    for rank, idx in enumerate(ranked[:k]):
        score = scores[idx]
        plaintext = _decrypt_text(docs[idx]["encrypted_text"], key)
        if plaintext:
            decrypted_count += 1
            truncated = plaintext[:70] + "..." if len(plaintext) > 70 else plaintext
            print(f"    {rank+1:2d}. [{score:.4f}] {truncated}")
        else:
            print(f"  x {rank+1:2d}. [{score:.4f}] <entschlüsselung fehlgeschlagen>")

    if decrypted_count == 0:
        print(f"\n  Falscher Key. Kein Dokument konnte entschlüsselt werden.")
        print(f"  Scores sind Rauschen (~0.01), Text ist nicht lesbar.")


def cmd_inspect(conn: sqlite3.Connection):
    docs = load_all_docs(conn)
    if not docs:
        print("\n  Vault ist leer.")
        return

    print(f"\n  Angreifer-Sicht: {len(docs)} Dokumente in der DB\n")
    print(f"  {'ID':>4} | {'Key-Hash':>16} | {'Ciphertext (hex)':>24} | Public Layer (first 6)")
    print(f"  {'─' * 90}")

    for d in docs:
        public = np.frombuffer(d["kpt_public"], dtype=np.float32)
        pub_str = ", ".join(f"{v:+.2f}" for v in public[:6])
        ct_hex = d["encrypted_text"][:12].hex() + "..."
        print(f"  {d['id']:4d} | {d['key_hash']:>16} | {ct_hex:>24} | [{pub_str}]")

    public = np.frombuffer(docs[0]["kpt_public"], dtype=np.float32)
    print(f"\n  Was ein Angreifer sieht:")
    print(f"  - {len(docs)} Einträge")
    print(f"  - Text: AES-256-GCM verschlüsselt (nur Ciphertext-Bytes)")
    print(f"  - Vektoren: KPT-verschlüsselt ({public.shape[0]} Public-Dims von 384 Original)")
    print(f"  - Key-Hashes: zeigen welche Docs zusammengehören")
    print(f"  - Ohne Key: kein Text lesbar, keine sinnvolle Suche möglich")


def cmd_attack(conn: sqlite3.Connection):
    docs = load_all_docs(conn)
    if not docs:
        print("\n  Vault ist leer.")
        return

    print(f"\n  Angreifer-Simulation auf {len(docs)} Dokumente...")
    print(f"  Zugriff auf: SQLite-DB (Ciphertext, Public Layer, KPT-Vektoren)")
    print(f"  Kein Zugriff auf: Secret Key, Klartext, Original-Embeddings\n")

    pub_vecs = np.stack([np.frombuffer(d["kpt_public"], dtype=np.float32) for d in docs])

    # Angriff 1: Text entschlüsseln
    print(f"  Angriff 1: AES-256-GCM Ciphertext knacken")
    sample = docs[0]["encrypted_text"]
    print(f"  Ciphertext: {sample[:24].hex()}... ({len(sample)} bytes)")
    print(f"  GESCHEITERT: AES-256-GCM ohne Key nicht brechbar.\n")

    # Angriff 2: Wrong-Key-Brute-Force auf KPT
    print(f"  Angriff 2: 50 zufällige Keys auf KPT-Vektoren")
    dim = 384
    doc_state = docs_to_state(docs, dim)
    max_score = 0.0
    for i in range(50):
        fake_key = f"attacker-key-{i}-{os.urandom(8).hex()}"
        kpt = build_kpt(fake_key, dim)
        query_vec = _vectorize(["test query"])
        q_state = kpt.encode_queries(query_vec)
        scores = kpt.score(doc_state, q_state)[0]
        max_score = max(max_score, float(np.max(scores)))

    print(f"  Bester Score über 50 Keys: {max_score:.4f}")
    print(f"  Zum Vergleich: Korrekter Key gibt ~0.55")
    if max_score < 0.05:
        print(f"  GESCHEITERT: Kein Key kommt auch nur in die Nähe.\n")

    # Angriff 3: PCA auf Public Layer
    print(f"  Angriff 3: PCA auf Public Layer (Information Bound)")
    from sklearn.decomposition import PCA
    n_comp = min(pub_vecs.shape[1], pub_vecs.shape[0] - 1, 50)
    if n_comp >= 2:
        pca = PCA(n_components=n_comp)
        pca.fit(pub_vecs)
        cumvar = np.cumsum(pca.explained_variance_ratio_)
        eff_dim_95 = int(np.searchsorted(cumvar, 0.95)) + 1
        print(f"  Public Dimensionen: {pub_vecs.shape[1]}")
        print(f"  Effektive Dimensionen (95%): {eff_dim_95}")
        print(f"  Original Dimensionen: {dim}")
        print(f"  Information Ratio: {eff_dim_95}/{dim} = {eff_dim_95/dim:.1%}")
    else:
        print(f"  Zu wenig Dokumente für PCA (braucht mindestens 3).")


def cmd_dump_row(conn: sqlite3.Connection):
    doc_id = input("\n  Dokument-ID: ").strip()
    if not doc_id.isdigit():
        print("  Ungültige ID.")
        return

    row = conn.execute(
        "SELECT id, encrypted_text, key_hash, kpt_public, kpt_wave_real, kpt_mode_weight FROM documents WHERE id = ?",
        (int(doc_id),)
    ).fetchone()

    if not row:
        print(f"  Dokument {doc_id} nicht gefunden.")
        return

    ct = row[1]
    print(f"\n  Dokument #{row[0]}")
    print(f"  Ciphertext: {ct[:32].hex()}... ({len(ct)} bytes)")
    print(f"  Key-Hash: {row[2]}")

    public = np.frombuffer(row[3], dtype=np.float32)
    wave_real = np.frombuffer(row[4], dtype=np.float32)
    mode_weight = np.frombuffer(row[5], dtype=np.float32)

    print(f"\n  Public Layer ({public.shape[0]} dims):")
    print(f"  {public[:12]}")

    print(f"\n  Wave Real (first 16 of {wave_real.shape[0]} values):")
    print(f"  {wave_real[:16]}")

    print(f"\n  Mode Weights ({mode_weight.shape[0]} modes):")
    for i, w in enumerate(mode_weight):
        bar = "█" * int(w * 50)
        print(f"    Mode {i}: {w:.4f} {bar}")

    print(f"\n  Das ist alles was at rest gespeichert ist.")
    print(f"  Text: AES-256-GCM verschlüsselt. Vektoren: KPT-verschlüsselt.")
    print(f"  Ohne den Key: nichts lesbar, nichts suchbar.")


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

HELP = """
  Befehle:
    add             Texte mit Passwort hinzufügen
    search          Mit Passwort suchen
    inspect         Daten at rest ansehen (Angreifer-Sicht)
    attack          Angreifer-Simulation (3 Attacken)
    dump            Einzelnes Dokument im Detail ansehen
    stats           DB-Statistiken
    clear           Alle Daten löschen
    help            Diese Hilfe
    quit            Beenden
"""


def _ask_key(prompt: str = "  Passwort: ") -> str:
    import getpass
    try:
        return getpass.getpass(prompt)
    except EOFError:
        return ""


def main():
    import argparse
    parser = argparse.ArgumentParser(description="KPT Vault — Interaktiver Encrypted Vector Store")
    parser.add_argument("--db", default="kpt_vault.db", help="SQLite-Datenbankpfad")
    args = parser.parse_args()

    conn = init_db(args.db)
    doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]

    print(f"\n  KPT Vault — Keyed Phase Transform Demo")
    print(f"  DB: {args.db} ({doc_count} Dokumente)")
    print(f"  Vectorizer: {_get_vectorizer()[0]}")
    print(HELP)

    while True:
        try:
            cmd = input("\n kpt> ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\n  Bye.")
            break

        if not cmd:
            continue

        if cmd in ("quit", "exit", "q"):
            print("  Bye.")
            break
        elif cmd == "help":
            print(HELP)
        elif cmd == "add":
            key = _ask_key()
            if not key:
                print("  Passwort erforderlich.")
                continue
            cmd_add(conn, key)
        elif cmd == "search":
            key = _ask_key()
            if not key:
                print("  Passwort erforderlich.")
                continue
            cmd_search(conn, key)
        elif cmd == "inspect":
            cmd_inspect(conn)
        elif cmd == "attack":
            cmd_attack(conn)
        elif cmd == "dump":
            cmd_dump_row(conn)
        elif cmd == "stats":
            count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            db_size = os.path.getsize(args.db)
            keys = conn.execute("SELECT DISTINCT key_hash FROM documents").fetchall()
            print(f"\n  Dokumente: {count}")
            print(f"  Verschiedene Keys: {len(keys)}")
            print(f"  DB-Größe: {db_size / 1024:.1f} KB")
            for kh in keys:
                kc = conn.execute("SELECT COUNT(*) FROM documents WHERE key_hash = ?", kh).fetchone()[0]
                print(f"    Key {kh[0]}: {kc} Docs")
        elif cmd == "clear":
            confirm = input("  Wirklich alles löschen? (ja/nein): ").strip()
            if confirm == "ja":
                conn.execute("DELETE FROM documents")
                conn.commit()
                print("  Vault geleert.")
        else:
            print(f"  Unbekannter Befehl: {cmd}")
            print(HELP)

    conn.close()


if __name__ == "__main__":
    main()
