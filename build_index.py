#!/usr/bin/env python3
# build_index.py — Optimized for app.py compatibility
# Outputs: faiss_index.bin, faiss_mapping.json, vectors.npz, faiss_index_meta.json

import os
import sys
import json
import time
import datetime
from typing import Any, List
import numpy as np

# Try imports
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# ---------- Config (Must match app.py) ----------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

KNOW_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
META_PATH = os.environ.get("FAISS_META_PATH", "faiss_index_meta.json")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH_SIZE = int(os.environ.get("BUILD_BATCH_SIZE", "20"))
RETRY_LIMIT = 3

# ---------- Utilities ----------
def flatten_json(path: str) -> List[dict]:
    """Flatten knowledge.json into list of {'path':..., 'text':...}."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping = []

    def scan(obj: Any, prefix: str = "root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            t = obj.strip()
            if t:
                # Clean up text slightly
                mapping.append({"path": prefix, "text": t})
        elif isinstance(obj, (int, float, bool)):
             mapping.append({"path": prefix, "text": str(obj)})

    scan(data, "root")
    return mapping

def synthetic_embedding(text: str, dim: int = 1536):
    """Deterministic fallback embedding."""
    h = abs(hash(text)) % (10 ** 12)
    vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]
    return vec

def get_embeddings_batch(texts: List[str], model: str):
    """Call OpenAI API with retry."""
    if not OPENAI_KEY or OpenAI is None:
        print("⚠️ OpenAI not available. Using synthetic embeddings.", file=sys.stderr)
        dim = 1536 if "small" in model else 3072
        return [synthetic_embedding(t, dim) for t in texts]

    client = OpenAI(api_key=OPENAI_KEY)
    
    for attempt in range(RETRY_LIMIT):
        try:
            # Replace newlines for better embedding quality
            clean_texts = [t.replace("\n", " ") for t in texts]
            
            resp = client.embeddings.create(input=clean_texts, model=model)
            return [d.embedding for d in resp.data]
        except Exception as e:
            print(f"⚠️ Error batch (attempt {attempt+1}/{RETRY_LIMIT}): {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
            
    print("❌ Failed to get embeddings. Using synthetic fallback.", file=sys.stderr)
    dim = 1536 if "small" in model else 3072
    return [synthetic_embedding(t, dim) for t in texts]

# ---------- Main Build Process ----------
def build_index():
    print(f"📂 Loading data from {KNOW_PATH}...")
    mapping = flatten_json(KNOW_PATH)
    texts = [m["text"] for m in mapping]
    n = len(texts)
    print(f"Found {n} text segments to index.")

    if n == 0:
        print("Nothing to index. Exiting.")
        sys.exit(0)

    # Generate Embeddings
    print(f"🚀 Generating embeddings (Model: {EMBEDDING_MODEL})...")
    all_vecs = []
    
    for i in range(0, n, BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"   Processing batch {i//BATCH_SIZE + 1}/{(n-1)//BATCH_SIZE + 1}...", end="\r")
        vecs = get_embeddings_batch(batch, EMBEDDING_MODEL)
        all_vecs.extend(vecs)
    
    print("\n✅ Embedding generation complete.")

    # Convert to Numpy
    mat = np.array(all_vecs, dtype="float32")
    dim = mat.shape[1]
    
    # Normalize for Cosine Similarity
    print("📐 Normalizing vectors...")
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / (norms + 1e-12)

    # 1. Save FAISS Index (if available)
    if HAS_FAISS:
        print(f"💾 Building FAISS index (dim={dim})...")
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"   -> Saved to {FAISS_INDEX_PATH}")
    else:
        print("⚠️ FAISS not installed. Skipping .bin generation.")

    # 2. Save Numpy Fallback (Always save this for compatibility)
    print(f"💾 Saving Numpy fallback vectors...")
    np.savez_compressed(FALLBACK_VECTORS_PATH, mat=mat)
    print(f"   -> Saved to {FALLBACK_VECTORS_PATH}")

    # 3. Save Mapping
    print(f"💾 Saving text mapping...")
    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    print(f"   -> Saved to {FAISS_MAPPING_PATH}")

    # 4. Save Metadata
    meta = {
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "count": n,
        "model": EMBEDDING_MODEL,
        "dimension": dim,
        "faiss_available": HAS_FAISS
    }
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print("🎉 DONE! Index ready for app.py")

if __name__ == "__main__":
    build_index()
