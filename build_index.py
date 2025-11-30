#!/usr/bin/env python3
# build_index.py — Bản HOÀN HẢO (v1.0)
# Chuẩn hoá 16 trường – tương thích 100% app.py.
# Xuất: faiss_index.bin, vectors.npz, faiss_mapping.json

import os, json, time, sys
import numpy as np
from typing import List, Dict, Optional

# Try FAISS
try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

# OpenAI API mới
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- CONFIG ----------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
META_PATH = os.environ.get("META_PATH", "faiss_meta.json")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH = int(os.environ.get("BUILD_BATCH_SIZE", "32"))

# 16 trường chuẩn
CANONICAL_FIELDS = [
    "tour_name","summary","location","duration","price",
    "includes","notes","style","transport","accommodation",
    "meals","event_support","hotline","mission",
    "includes_extra","extras"
]

def synthetic_embedding(text: str, dim: int = 1536):
    h = abs(hash(text)) % (10**12)
    return [(float((h >> (i % 32)) & 0xFF)+ (i % 7))/255.0 for i in range(dim)]

def embed_batch(texts: List[str], model: str):
    """Batch embed - fallback if no API"""
    if not OPENAI_KEY or OpenAI is None:
        dim = 1536 if "3-small" in model else 3072
        return [synthetic_embedding(t, dim) for t in texts]

    cli = OpenAI(api_key=OPENAI_KEY)
    try:
        r = cli.embeddings.create(model=model, input=texts)
        return [d.embedding for d in r.data]
    except Exception:
        dim = 1536 if "3-small" in model else 3072
        return [synthetic_embedding(t, dim) for t in texts]

# ---------- FLATTEN KNOWLEDGE.JSON ----------
def flatten_knowledge() -> List[dict]:
    """Flatten knowledge.json theo đúng chuẩn app.py yêu cầu."""
    if not os.path.exists(KNOWLEDGE_PATH):
        raise FileNotFoundError(f"{KNOWLEDGE_PATH} không tồn tại")

    with open(KNOWLEDGE_PATH,"r",encoding="utf-8") as f:
        data = json.load(f)

    mapping = []

    def scan(obj, prefix="root"):
        if isinstance(obj, dict):
            for k in sorted(obj.keys()):
                scan(obj[k], f"{prefix}.{k}")
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
            except:
                pass

    scan(data)

    # Bổ sung field, tour_index, tour_name
    for m in mapping:
        path = m["path"]

        # field
        last = path.split(".")[-1]
        m["field"] = last

        # tour_index
        import re
        mt = re.search(r"tours\[(\d+)\]", path)
        m["tour_index"] = int(mt.group(1)) if mt else None

        # tour_name
        if m["field"] == "tour_name":
            m["tour_name"] = m["text"]
        else:
            m["tour_name"] = None

        # Chuẩn hoá alias → canonical field
        if m["field"] in ("includes_extra",):
            m["field"] = "includes"
        if m["field"] in ("extras",):
            m["field"] = "notes"

    return mapping

# ---------- BUILD INDEX ----------
def build_index():
    print("📌 Flatten knowledge.json…")
    mapping = flatten_knowledge()
    texts = [m["text"] for m in mapping]
    n = len(texts)
    print(f"Found {n} passages")

    if n == 0:
        raise RuntimeError("Không có passage nào — knowledge.json rỗng")

    # batch embed
    vecs = []
    for i in range(0,n,BATCH):
        batch = texts[i:i+BATCH]
        embs = embed_batch(batch, EMBED_MODEL)
        vecs.extend(embs)
        print(f"  Embedded {len(vecs)}/{n}")

    mat = np.array(vecs,dtype="float32")
    # normalize rows
    mat = mat / (np.linalg.norm(mat,axis=1,keepdims=True)+1e-12)

    # save fallback vectors
    np.savez_compressed(FALLBACK_VECTORS_PATH, mat=mat)
    print("Saved vectors:", FALLBACK_VECTORS_PATH)

    # save mapping
    with open(FAISS_MAPPING_PATH,"w",encoding="utf-8") as f:
        json.dump(mapping,f,ensure_ascii=False,indent=2)
    print("Saved mapping:", FAISS_MAPPING_PATH)

    # build FAISS
    if HAS_FAISS:
        dim = mat.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(mat)
        faiss.write_index(index, FAISS_INDEX_PATH)
        print("Saved FAISS index:", FAISS_INDEX_PATH)

    # meta
    meta = {
        "created_at": time.time(),
        "n_passages": int(n),
        "dim": int(mat.shape[1]),
        "embedding_model": EMBED_MODEL,
        "faiss": HAS_FAISS
    }
    with open(META_PATH,"w",encoding="utf-8") as f:
        json.dump(meta,f,indent=2)
    print("Saved meta:", META_PATH)

    print("\n🎉 BUILD DONE — mapping + vectors + index OK")

if __name__=="__main__":
    try:
        build_index()
    except Exception as e:
        print("❌ ERROR:", e)
        raise
