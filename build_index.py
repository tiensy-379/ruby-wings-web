#!/usr/bin/env python3
"""
Replacement build_index.py

- Fully self-contained replacement tailored to run on Windows or Linux without creating new directories.
- Uses environment variables when provided, otherwise defaults to local files in working directory.
- Compatible with both old and new OpenAI Python SDKs.
- Deterministic fallback embeddings when OPENAI_API_KEY is absent.
- Produces: knowledge_fixed.json, faiss_mapping.json (index metadata), vectors.npz (fallback), faiss_index.bin (if FAISS available and enabled).
- Creates .bak backups before overwriting files.

Usage:
    python build_index.py [--force] [--dry-run]

Notes:
- Do NOT expect the script to create /mnt/data on Windows. Defaults are local files.
- To target a specific path, set env vars: KNOWLEDGE_IN, KNOWLEDGE_OUT, FAISS_MAPPING_PATH, FAISS_INDEX_PATH, FALLBACK_VECTORS_PATH
"""

import os
import sys
import json
import hashlib
import logging
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Optional libs
try:
    import openai
except Exception:
    openai = None

try:
    import numpy as np
except Exception:
    print("ERROR: numpy is required. Install via pip install numpy", file=sys.stderr)
    raise

# faiss optional
try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

# requests optional (used for unit checks)
try:
    import requests
except Exception:
    requests = None

# ----------------- Logging -----------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOG = logging.getLogger("build_index")

# ----------------- Config ------------------
# Default base: use /mnt/data on Linux only if present, otherwise local directory
if os.name == "nt":
    BASE = "."
else:
    # if /mnt/data exists use it, else local
    BASE = "/mnt/data" if Path("/mnt/data").exists() else "."

OPENAI_KEY = os.getenv("OPENAI_API_KEY", "").strip()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
FAISS_ENABLED = os.getenv("FAISS_ENABLED", "false").lower() in ("1","true","yes")

KNOWLEDGE_IN = Path(os.getenv("KNOWLEDGE_IN", str(Path(BASE) / "knowledge.json")))
KNOWLEDGE_OUT = Path(os.getenv("KNOWLEDGE_OUT", str(Path(BASE) / "knowledge_fixed.json")))
FAISS_MAPPING_PATH = Path(os.getenv("FAISS_MAPPING_PATH", str(Path(BASE) / "faiss_mapping.json")))
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", str(Path(BASE) / "faiss_index.bin")))
FALLBACK_VECTORS_PATH = Path(os.getenv("FALLBACK_VECTORS_PATH", str(Path(BASE) / "vectors.npz")))
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "1536"))

APP_HOME = os.getenv("RBW_APP_HOME", "http://localhost:10000/home")
APP_REINDEX = os.getenv("RBW_APP_REINDEX", "http://localhost:10000/reindex")
APP_CHAT = os.getenv("RBW_APP_CHAT", "http://localhost:10000/chat")
RBW_ALLOW_REINDEX = os.getenv("RBW_ALLOW_REINDEX", "0")
RBW_ADMIN_TOKEN = os.getenv("RBW_ADMIN_TOKEN")

MAX_FIELD_CHARS = 2000
MIN_PASSAGE = 20
MAX_PASSAGE = 300

# ----------------- Helpers -----------------

def backup_if_exists(p: Path):
    try:
        if p.exists():
            bak = p.with_suffix(p.suffix + ".bak")
            LOG.info("Backing up %s -> %s", p, bak)
            shutil.copy2(p, bak)
    except Exception:
        LOG.exception("Backup failed for %s", p)


def write_json_atomic(path: Path, data: Any):
    backup_if_exists(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp.replace(path)
    LOG.info("Wrote %s", path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def truncate_or_split_text(text: str, limit: int = MAX_FIELD_CHARS) -> List[str]:
    text = text.strip()
    if len(text) <= limit:
        return [text]
    parts = []
    cur = ""
    for para in text.split("\n"):
        para = para.strip()
        if not para:
            continue
        if len(cur) + 1 + len(para) <= limit:
            cur = (cur + "\n" + para).strip() if cur else para
        else:
            if cur:
                parts.append(cur)
            if len(para) <= limit:
                cur = para
            else:
                for i in range(0, len(para), limit):
                    parts.append(para[i:i+limit])
                cur = ""
    if cur:
        parts.append(cur)
    return parts

# ----------------- Normalization -----------------

def normalize_item(item: Dict[str, Any], idx: int) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out['id'] = item.get('id') or item.get('slug') or f'item_{idx}'
    out['name'] = item.get('name') or item.get('title') or ''
    out['short_description'] = item.get('description') or item.get('short_description') or ''
    out['vision'] = item.get('vision','')
    out['mission'] = item.get('mission','')

    core_values = item.get('core_values') or item.get('values') or []
    if isinstance(core_values, str):
        core_values = [v.strip() for v in core_values.split('\n') if v.strip()]
    out['core_values'] = []
    for v in core_values:
        out['core_values'].extend(truncate_or_split_text(v)) if isinstance(v, str) else None

    highlights = item.get('highlights') or item.get('features') or []
    if isinstance(highlights, str):
        highlights = [h.strip() for h in highlights.split('\n') if h.strip()]
    out['highlights'] = []
    for h in highlights:
        out['highlights'].extend(truncate_or_split_text(h)) if isinstance(h, str) else None

    out['itinerary'] = item.get('itinerary') or {}
    out['price_info'] = item.get('price_info') or item.get('price_table') or item.get('price_range') or {}
    out['target_audience'] = item.get('target_audience') or item.get('audience') or ''
    out['contact'] = item.get('contact') or item.get('contacts') or ''
    out['tags'] = item.get('tags') or []
    out['faqs'] = item.get('faqs') or []

    # ensure fields not exceed max
    for fld in ['short_description','vision','mission','target_audience','contact']:
        val = out.get(fld,'')
        if val and isinstance(val,str) and len(val) > MAX_FIELD_CHARS:
            parts = truncate_or_split_text(val)
            out[fld] = parts[0]
            for i,p in enumerate(parts[1:], start=1):
                out[f"{fld}_part_{i}"] = p

    return out

# ----------------- QA generation -----------------

def generate_faqs_for_item(item: Dict[str,Any]) -> List[Dict[str,Any]]:
    name = item.get('name') or item.get('id')
    qas = []
    def add(q,a,kw=None):
        qas.append({'question': q, 'answer': a, 'keywords': kw or []})
    if item.get('mission'):
        add(f"Sứ mệnh của {name} là gì?", item['mission'][:500], ["mission", name])
    if item.get('vision'):
        add(f"Tầm nhìn của {name} là gì?", item['vision'][:500], ["vision", name])
    if item.get('price_info'):
        add(f"Giá tham khảo của {name} như thế nào?", str(item['price_info'])[:300], ["giá", "price"])
    if item.get('highlights'):
        add(f"Điểm nổi bật của {name}?", ", ".join(item.get('highlights')[:5]), ["highlights", name])
    if item.get('contact'):
        add(f"Làm sao để liên hệ {name}?", item['contact'] if isinstance(item['contact'],str) else json.dumps(item['contact'], ensure_ascii=False), ["contact", name])
    if not qas:
        add(f"Thông tin chính về {name} là gì?", item.get('short_description',''), [name])
    return qas[:3]

# ----------------- Passage extraction -----------------

def extract_passages(normalized_items: List[Dict[str,Any]]) -> List[Dict[str,str]]:
    passages = []
    for idx, it in enumerate(normalized_items):
        base = f"root.items[{idx}]"
        def add(path, text):
            if not text: return
            s = ' '.join(str(text).split())
            if len(s) < MIN_PASSAGE: return
            for i in range(0, len(s), MAX_PASSAGE):
                chunk = s[i:i+MAX_PASSAGE].strip()
                if len(chunk) >= MIN_PASSAGE:
                    passages.append({'path': f"{base}.{path}", 'text': chunk})
        for fld in ['short_description','vision','mission','target_audience','contact']:
            add(fld, it.get(fld))
        for arr in ['core_values','highlights']:
            for j, v in enumerate(it.get(arr, [])):
                add(f"{arr}[{j}]", v)
        itin = it.get('itinerary')
        if isinstance(itin, str):
            add('itinerary', itin)
        elif isinstance(itin, dict):
            for k,v in itin.items():
                add(f"itinerary.{k}", v)
        elif isinstance(itin, list):
            for j,v in enumerate(itin):
                add(f"itinerary[{j}]", v)
    return passages

# ----------------- OpenAI embedding wrapper -----------------
def openai_embed_texts(texts: List[str], model: str) -> List[List[float]]:
    if not texts:
        return []
    if openai is None or not OPENAI_KEY:
        raise RuntimeError("OpenAI not available or OPENAI_API_KEY not set")
    try:
        # try modern then fallback
        try:
            resp = openai.Embeddings.create(model=model, input=texts)
        except Exception:
            resp = openai.Embedding.create(model=model, input=texts)
        if isinstance(resp, dict):
            data = resp.get('data', [])
        else:
            data = getattr(resp, 'data', []) or []
        vectors = []
        for item in data:
            if isinstance(item, dict):
                vec = item.get('embedding') or item.get('vector')
            else:
                vec = getattr(item, 'embedding', None) or getattr(item, 'vector', None)
            if not vec:
                raise RuntimeError('Missing embedding in OpenAI response')
            vectors.append(vec)
        return vectors
    except Exception as e:
        raise RuntimeError(f"OpenAI embedding failed: {e}")

# ----------------- Deterministic fallback embedding -----------------
def deterministic_embed(text: str, dim: int = EMBEDDING_DIM) -> List[float]:
    h = hashlib.sha256(text.encode('utf-8')).digest()
    needed = dim * 4
    rep = (h * ((needed // len(h)) + 1))[:needed]
    arr = np.frombuffer(rep, dtype=np.uint8).astype(np.float32)
    arr = arr.reshape(-1,4)
    ints = (arr[:,0]*256**3 + arr[:,1]*256**2 + arr[:,2]*256 + arr[:,3]).astype(np.float64)
    floats = (ints % 1000000) / 1000000.0
    vec = np.resize(floats, dim).astype(np.float32)
    norm = np.linalg.norm(vec)
    if norm == 0:
        return vec.tolist()
    return (vec / norm).tolist()

# ----------------- Generate embeddings for passages -----------------
def generate_embeddings_for_passages(passages: List[Dict[str,str]]) -> np.ndarray:
    texts = [p['text'] for p in passages]
    if OPENAI_KEY and openai is not None:
        try:
            vectors = openai_embed_texts(texts, model=EMBEDDING_MODEL)
            arr = np.array(vectors, dtype='float32')
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            arr = arr / norms
            return arr
        except Exception:
            LOG.exception("OpenAI embeddings failed; falling back to deterministic embeddings")
    # fallback deterministic
    arr = np.vstack([deterministic_embed(t) for t in texts]).astype('float32')
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms[norms==0] = 1.0
    arr = arr / norms
    return arr

# ----------------- Save vectors -----------------
def save_vectors_npz(path: Path, vectors: np.ndarray):
    backup_if_exists(path)
    np.savez_compressed(path, vectors=vectors)
    LOG.info("Saved vectors to %s", path)

# ----------------- Build FAISS or fallback index -----------------
def build_faiss_index(vectors: np.ndarray, path: Path):
    if not HAS_FAISS:
        raise RuntimeError('faiss not installed')
    d = vectors.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(vectors)
    backup_if_exists(path)
    faiss.write_index(index, str(path))
    LOG.info("Wrote FAISS index to %s", path)

# ----------------- Main pipeline -----------------
def run_pipeline(force_rebuild: bool = False, dry_run: bool = False) -> Tuple[bool, Dict[str,Any]]:
    details: Dict[str,Any] = {"errors": []}
    try:
        if not KNOWLEDGE_IN.exists():
            raise FileNotFoundError(f"Input knowledge file not found: {KNOWLEDGE_IN}")
        LOG.info("Reading knowledge from %s", KNOWLEDGE_IN)
        raw = read_json(KNOWLEDGE_IN)
        if isinstance(raw, list):
            items = raw
        elif isinstance(raw, dict):
            items = raw.get('tours') or raw.get('items') or raw.get('data') or [raw]
        else:
            items = [raw]

        normalized = [normalize_item(it, i) for i,it in enumerate(items)]
        # generate faqs if missing
        for it in normalized:
            if not it.get('faqs'):
                it['faqs'] = generate_faqs_for_item(it)

        if not dry_run:
            write_json_atomic(KNOWLEDGE_OUT, normalized)

        passages = extract_passages(normalized)
        LOG.info("Extracted %d passages", len(passages))
        if not dry_run:
            write_json_atomic(FAISS_MAPPING_PATH, passages)

        vectors = generate_embeddings_for_passages(passages)
        LOG.info("Embeddings shape: %s", vectors.shape)
        if not dry_run:
            save_vectors_npz(FALLBACK_VECTORS_PATH, vectors)

        # build faiss if enabled and available
        if FAISS_ENABLED and HAS_FAISS:
            try:
                build_faiss_index(vectors, FAISS_INDEX_PATH)
            except Exception:
                LOG.exception("Failed to build FAISS index; will keep fallback vectors")
                details['errors'].append('faiss_build_failed')
        else:
            LOG.info("FAISS not used; stored vectors for fallback at %s", FALLBACK_VECTORS_PATH)

        # final checks
        if FAISS_MAPPING_PATH.exists():
            mapping = read_json(FAISS_MAPPING_PATH)
            if len(mapping) != len(passages):
                details['errors'].append('mapping_length_mismatch')
        else:
            details['errors'].append('mapping_missing')

        # success if no critical errors
        success = len(details['errors']) == 0
        details.update({'passages': len(passages), 'created': {
            'knowledge_out': str(KNOWLEDGE_OUT) if KNOWLEDGE_OUT.exists() else None,
            'mapping': str(FAISS_MAPPING_PATH) if FAISS_MAPPING_PATH.exists() else None,
            'vectors': str(FALLBACK_VECTORS_PATH) if FALLBACK_VECTORS_PATH.exists() else None,
            'faiss': str(FAISS_INDEX_PATH) if FAISS_INDEX_PATH.exists() else None
        }})
        return success, details
    except Exception as e:
        LOG.exception("Pipeline failed: %s", e)
        details['errors'].append(str(e))
        # rollback created files (attempt restore from .bak)
        for p in (KNOWLEDGE_OUT, FAISS_MAPPING_PATH, FALLBACK_VECTORS_PATH, FAISS_INDEX_PATH):
            try:
                bak = Path(str(p) + '.bak')
                if bak.exists():
                    shutil.copy2(bak, p)
                    LOG.info('Restored backup %s -> %s', bak, p)
                else:
                    if p.exists():
                        p.unlink()
                        LOG.info('Removed created file %s', p)
            except Exception:
                LOG.exception('Rollback step failed for %s', p)
        return False, details

# ----------------- CLI -----------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', help='force rebuild')
    parser.add_argument('--dry-run', action='store_true', help='do not write files')
    args = parser.parse_args()

    ok, info = run_pipeline(force_rebuild=args.force, dry_run=args.dry_run)
    if ok:
        LOG.info('Pipeline completed OK')
        print(json.dumps(info, ensure_ascii=False, indent=2))
        sys.exit(0)
    else:
        LOG.error('Pipeline failed: %s', info.get('errors'))
        print(json.dumps(info, ensure_ascii=False, indent=2))
        sys.exit(2)
