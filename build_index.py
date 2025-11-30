#!/usr/bin/env python3
# build_index.py — build embeddings/faiss index + mapping + tour_entities.json (compatible with app.py & entities.py)
# This version reuses common_utils.flatten_json to ensure indices/path stability.
#
# Usage:
#   pip install -r requirements.txt
#   export OPENAI_API_KEY="sk-..."
#   python build_index.py

import os
import sys
import json
import time
import datetime
from typing import Any, List
import numpy as np

# try imports with helpful fallbacks
try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

try:
    from openai import OpenAI
    from openai import OpenAIError
except Exception:
    OpenAI = None
    OpenAIError = Exception

# import shared flatten
try:
    from common_utils import flatten_json
except Exception:
    def flatten_json(path: str) -> List[dict]:
        # fallback: simple flatten (shouldn't occur if common_utils present)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        mapping = []
        def scan(obj, prefix="root"):
            if isinstance(obj, dict):
                for k,v in obj.items():
                    scan(v, f"{prefix}.{k}")
            elif isinstance(obj, list):
                for i,v in enumerate(obj):
                    scan(v, f"{prefix}[{i}]")
            elif isinstance(obj, str):
                t = obj.strip()
                if t:
                    mapping.append({"path": prefix, "text": t})
            else:
                try:
                    s = str(obj).strip()
                    if s:
                        mapping.append({"path": prefix, "text": s})
                except Exception:
                    pass
        scan(data, "root")
        return mapping

# Try to import entities.build_entity_index to persist tour_entities.json
try:
    from entities import build_entity_index, ENTITY_PATH_DEFAULT
except Exception:
    build_entity_index = None
    ENTITY_PATH_DEFAULT = os.environ.get("TOUR_ENTITIES_PATH", "tour_entities.json")

# ---------- Config ----------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

KNOW_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
META_PATH = os.environ.get("FAISS_META_PATH", "faiss_index_meta.json")
TOUR_ENTITIES_PATH = os.environ.get("TOUR_ENTITIES_PATH", ENTITY_PATH_DEFAULT)

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.environ.get("BUILD_BATCH_SIZE", "8"))  # small batch for low RAM
RETRY_LIMIT = int(os.environ.get("RETRY_LIMIT", "5"))
RETRY_BASE = float(os.environ.get("RETRY_BASE_DELAY", "1.0"))

TMP_EMB_FILE = "emb_tmp.bin"  # binary store for concatenated float32 rows

# ---------- Utilities ----------
def synthetic_embedding(text: str, dim: int = 1536):
    """Deterministic synthetic embedding (fallback when no API)."""
    h = abs(hash(text)) % (10 ** 12)
    vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]
    return vec

def call_embeddings_with_retry(inputs: List[str], model: str):
    """Call OpenAI embeddings with retries; returns list of vectors (lists)."""
    if not OPENAI_KEY or OpenAI is None:
        # No API available -> synthetic
        dim = 1536 if "3-small" in model else 3072
        return [synthetic_embedding(text, dim) for text in inputs]

    client = OpenAI(api_key=OPENAI_KEY)
    attempt = 0
    while attempt <= RETRY_LIMIT:
        try:
            resp = client.embeddings.create(model=model, input=inputs)
            if resp.data and len(resp.data) > 0:
                out = [r.embedding for r in resp.data]
                print(f"✅ Generated {len(out)} embeddings (model={model})", flush=True)
                return out
            else:
                raise ValueError("Empty response from OpenAI embeddings API")
        except Exception as e:
            attempt += 1
            if attempt > RETRY_LIMIT:
                print(f"❌ Embedding API failed after {RETRY_LIMIT} attempts: {e}", file=sys.stderr)
                # fallback to synthetic embeddings
                dim = 1536 if "3-small" in model else 3072
                return [synthetic_embedding(text, dim) for text in inputs]
            delay = RETRY_BASE * (2 ** (attempt - 1))
            print(f"⚠️ Embedding API error (attempt {attempt}/{RETRY_LIMIT}): {e}. Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)
    # final fallback
    dim = 1536 if "3-small" in model else 3072
    return [synthetic_embedding(text, dim) for text in inputs]

# ---------- Main build flow ----------
def build_index():
    print("Flattening knowledge.json via common_utils...")
    mapping = flatten_json(KNOW_PATH)
    texts = [m["text"] for m in mapping]
    n = len(texts)
    print(f"Found {n} passages.")
    if n == 0:
        print("No passages to index -> exit", file=sys.stderr)
        sys.exit(1)

    # prepare temporary binary file
    if os.path.exists(TMP_EMB_FILE):
        try:
            os.remove(TMP_EMB_FILE)
        except Exception:
            pass

    dim = None
    total_rows = 0

    # compute embeddings batch by batch, write to binary file
    batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i:i+BATCH_SIZE]
        inputs = [t if (t and str(t).strip()) else " " for t in batch]
        print(f"Embedding batch {i//BATCH_SIZE + 1}/{batches} ...", flush=True)

        vecs = call_embeddings_with_retry(inputs, EMBEDDING_MODEL)
        if not vecs or any(v is None for v in vecs):
            # ensure no None entries
            for j, v in enumerate(vecs):
                if v is None:
                    vecs[j] = synthetic_embedding(inputs[j])
        if dim is None and vecs:
            dim = len(vecs[0])

        arr = np.array(vecs, dtype="float32")
        # normalize rows (unit norm)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / (norms + 1e-12)

        # append to binary file
        with open(TMP_EMB_FILE, "ab") as f:
            f.write(arr.tobytes())

        total_rows += arr.shape[0]

    if total_rows == 0:
        print("No embeddings created -> exit", file=sys.stderr)
        sys.exit(1)

    print("Loading embeddings via memmap...")
    emb = np.memmap(TMP_EMB_FILE, dtype="float32", mode="r", shape=(total_rows, dim))

    # build FAISS index (IndexFlatIP uses inner product on normalized vectors -> cosine)
    print("Building index...")
    if HAS_FAISS:
        index = faiss.IndexFlatIP(dim)
        try:
            index.add(np.asarray(emb))
        except Exception:
            # fallback to loading in-memory chunk if memmap add fails
            index.add(np.array(emb, dtype="float32"))
        try:
            faiss.write_index(index, FAISS_INDEX_PATH)
            print(f"Saved FAISS index to {FAISS_INDEX_PATH}")
        except Exception:
            print("Failed to persist FAISS index (continuing).", file=sys.stderr)
    else:
        # fallback: save vectors to npz and rely on NumpyIndex in app.py
        np.savez_compressed(FALLBACK_VECTORS_PATH, mat=np.asarray(emb))
        print(f"FAISS not available: saved fallback vectors to {FALLBACK_VECTORS_PATH}")

    # Save mapping (list of {"path","text"}) expected by app.py
    print(f"Saving mapping to {FAISS_MAPPING_PATH} ...")
    try:
        with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
            json.dump(mapping, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("Failed to save mapping:", e, file=sys.stderr)

    # Save fallback vectors even when FAISS present (cheap)
    try:
        if HAS_FAISS:
            np.savez_compressed(FALLBACK_VECTORS_PATH, mat=np.asarray(emb))
    except Exception:
        pass

    # Build tour_entities.json using entities.build_entity_index if available
    try:
        if build_entity_index is not None:
            print(f"Building entity index -> {TOUR_ENTITIES_PATH} ...")
            idx_map = build_entity_index(mapping, out_path=TOUR_ENTITIES_PATH)
            print(f"Saved entity index keys: {len(idx_map)}")
        else:
            print("entities.build_entity_index not available; skipping entity index build.")
    except Exception as e:
        print("Failed to build entity index:", e, file=sys.stderr)

    # write meta
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "num_passages": total_rows,
        "embedding_model": EMBEDDING_MODEL,
        "dimension": dim,
        "faiss_available": HAS_FAISS,
        "notes": "Built with batch memmap flow; tour_entities.json produced if entities module present"
    }
    try:
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    # cleanup
    try:
        os.remove(TMP_EMB_FILE)
    except Exception:
        pass

    print("DONE. Index ready.")
    print(f"- index: {FAISS_INDEX_PATH if HAS_FAISS else '(none, fallback used)'}")
    print(f"- mapping: {FAISS_MAPPING_PATH}")
    print(f"- vectors (npz): {FALLBACK_VECTORS_PATH}")
    print(f"- entities: {TOUR_ENTITIES_PATH} (if produced)")
    print(f"- meta: {META_PATH}")

if __name__ == "__main__":
    try:
        build_index()
    except Exception as e:
        print("ERROR building index:", e, file=sys.stderr)
        raise