#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust app.py for Ruby Wings — final version.
Fixes: OpenAI SDK variations, safe atomic writes, dir creation, flexible embedding dims,
robust vectors.npz loading, FAISS optional, safer reindex, UTF-8 everywhere.
Run: python app.py  (or gunicorn -w1 -b 0.0.0.0:10000 app:app)
"""
import os
import json
import logging
import threading
import hashlib
from functools import lru_cache
from typing import List, Tuple, Dict, Any

# numpy required
try:
    import numpy as np
except Exception as e:
    raise RuntimeError("numpy is required") from e

# optional openai
try:
    import openai
except Exception:
    openai = None

# optional faiss
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception as e:
    faiss = None
    HAS_FAISS = False
    FAISS_IMPORT_ERROR = str(e)

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS

# ---------- logging ----------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rbw.app")

# ---------- config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
# Defaults are relative to repo root — easy to copy/restore
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge_fixed.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-large")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED_ENV = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")
RBW_ALLOW_REINDEX = os.environ.get("RBW_ALLOW_REINDEX", "0")
RBW_ADMIN_TOKEN = os.environ.get("RBW_ADMIN_TOKEN", "")

if OPENAI_API_KEY and openai is not None:
    try:
        openai.api_key = OPENAI_API_KEY
        openai.api_base = OPENAI_BASE_URL
    except Exception:
        pass

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY not set — deterministic embedding fallback will be used.")
if FAISS_ENABLED_ENV and not HAS_FAISS:
    logger.warning("FAISS requested but not available: %s", FAISS_IMPORT_ERROR if 'FAISS_IMPORT_ERROR' in globals() else 'unknown')

# ---------- flask ----------
app = Flask(__name__)
CORS(app)

# ---------- global state ----------
KNOWLEDGE: Any = {}
MAPPING: List[Dict[str, str]] = []
FLATTENED_TEXTS: List[str] = []
INDEX_LOCK = threading.Lock()
INDEX = None  # faiss index or NumpyFallbackIndex

# ---------- fallback index ----------
class NumpyFallbackIndex:
    def __init__(self, mat: np.ndarray = None):
        if mat is None or getattr(mat, "size", 0) == 0:
            self.mat = np.empty((0,0), dtype="float32")
            self.dim = None
        else:
            self.mat = mat.astype("float32")
            self.dim = self.mat.shape[1]

    def add(self, mat: np.ndarray):
        if mat is None or getattr(mat, "size", 0) == 0:
            return
        mat = mat.astype("float32")
        if getattr(self.mat, "size", 0) == 0:
            self.mat = mat.copy()
            self.dim = mat.shape[1]
        else:
            if mat.shape[1] != self.dim:
                raise ValueError("Dimension mismatch")
            self.mat = np.vstack([self.mat, mat])

    def search(self, qvec: np.ndarray, k: int):
        if getattr(self.mat, "size", 0) == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        q = qvec.astype("float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        mat_norm = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        sims = np.dot(q, mat_norm.T)
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype("float32"), idx.astype("int64")

    @property
    def ntotal(self):
        return 0 if getattr(self.mat, "size", 0) == 0 else self.mat.shape[0]

    def save(self, path: str):
        try:
            # ensure directory exists
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            np.savez_compressed(path, mat=self.mat)
            logger.info("Saved fallback vectors to %s", path)
        except Exception:
            logger.exception("Failed saving fallback vectors")

    @classmethod
    def load(cls, path: str):
        try:
            arr = np.load(path, allow_pickle=False)
            key = 'mat' if 'mat' in arr.files else (arr.files[0] if len(arr.files)>0 else None)
            mat = arr[key] if key else None
            return cls(mat=mat)
        except Exception:
            logger.exception("Failed loading fallback vectors from %s", path)
            return cls(None)

# ---------- helpers ----------
def ensure_parent(path: str):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)

def backup_if_exists(path: str):
    try:
        if os.path.exists(path):
            bak = f"{path}.bak"
            ensure_parent(bak)
            with open(path, "rb") as r, open(bak, "wb") as w:
                w.write(r.read())
            logger.info("Backed up %s -> %s", path, bak)
    except Exception:
        logger.exception("Backup failed for %s", path)

def write_json_atomic(path: str, data: Any):
    try:
        ensure_parent(path)
        backup_if_exists(path)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
        logger.info("Wrote JSON to %s", path)
    except Exception:
        logger.exception("Atomic write failed for %s", path)
        raise

def load_mapping(path: str = FAISS_MAPPING_PATH):
    global MAPPING, FLATTENED_TEXTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            MAPPING = json.load(f)
        FLATTENED_TEXTS = [m.get("text","") for m in MAPPING]
        logger.info("Loaded mapping %d entries", len(MAPPING))
    except Exception:
        logger.exception("Failed to load mapping; resetting")
        MAPPING = []
        FLATTENED_TEXTS = []

def load_knowledge(path: str = KNOWLEDGE_PATH) -> int:
    global KNOWLEDGE, MAPPING, FLATTENED_TEXTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOWLEDGE = json.load(f)
    except Exception:
        logger.exception("Could not open knowledge file: %s", path)
        KNOWLEDGE = {}
    # if mapping exists, prefer it (so pipeline can be incremental)
    if os.path.exists(FAISS_MAPPING_PATH):
        load_mapping(FAISS_MAPPING_PATH)
        return len(FLATTENED_TEXTS)
    # flatten
    MAPPING = []
    FLATTENED_TEXTS = []
    def scan(obj, prefix="root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            t = obj.strip()
            if t:
                MAPPING.append({"path": prefix, "text": t})
                FLATTENED_TEXTS.append(t)
    scan(KNOWLEDGE, "root")
    logger.info("Flattened %d passages from knowledge", len(FLATTENED_TEXTS))
    return len(FLATTENED_TEXTS)

# ---------- embeddings ----------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    # try openai: support both Embeddings and Embedding client shapes
    if OPENAI_API_KEY and openai is not None:
        try:
            if hasattr(openai, "Embeddings"):
                resp = openai.Embeddings.create(model=EMBEDDING_MODEL, input=short)
            else:
                resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=short)
            data = resp.get("data", None) if isinstance(resp, dict) else getattr(resp, "data", None)
            if data and len(data) > 0:
                first = data[0]
                emb = first.get("embedding") if isinstance(first, dict) else getattr(first, "embedding", None)
                if emb:
                    return emb, len(emb)
        except Exception:
            logger.exception("OpenAI embedding error; fallback will be used.")
    # deterministic fallback: sha256 -> bytes -> floats -> normalized vector
    try:
        h = hashlib.sha256(short.encode("utf-8")).digest()
        needed = 1536 * 4
        rep = (h * ((needed // len(h)) + 1))[:needed]
        arr = np.frombuffer(rep, dtype=np.uint8).astype(np.float32)
        arr = arr.reshape(-1, 4)
        ints = (arr[:,0]*256**3 + arr[:,1]*256**2 + arr[:,2]*256 + arr[:,3]).astype(np.float64)
        floats = (ints % 1000000) / 1000000.0
        vec = np.resize(floats, 1536).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec.tolist(), len(vec)
        vec = (vec / norm).astype(np.float32)
        return vec.tolist(), len(vec)
    except Exception:
        logger.exception("Fallback embedding generation failed")
        return [], 0

# ---------- index build / management ----------
def build_index(force_rebuild: bool = False) -> bool:
    global INDEX, MAPPING, FLATTENED_TEXTS
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED_ENV and HAS_FAISS
        # try load persisted
        if not force_rebuild:
            try:
                if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                    idx = faiss.read_index(FAISS_INDEX_PATH)
                    load_mapping(FAISS_MAPPING_PATH)
                    INDEX = idx
                    logger.info("Loaded FAISS index from disk")
                    # sanity check mapping vs index.ntotal
                    try:
                        if idx.ntotal != len(MAPPING):
                            logger.warning("FAISS ntotal (%d) != mapping length (%d)", idx.ntotal, len(MAPPING))
                    except Exception:
                        pass
                    return True
                if (not use_faiss) and os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                    idx = NumpyFallbackIndex.load(FALLBACK_VECTORS_PATH)
                    load_mapping(FAISS_MAPPING_PATH)
                    INDEX = idx
                    logger.info("Loaded fallback vectors from disk")
                    return True
            except Exception:
                logger.exception("Failed to load persisted index; will rebuild")
        if not FLATTENED_TEXTS:
            logger.warning("No passages to build index")
            INDEX = None
            return False
        vectors = []
        dims = None
        for t in FLATTENED_TEXTS:
            emb, d = embed_text(t)
            if not emb:
                continue
            if dims is None:
                dims = d
            vectors.append(np.array(emb, dtype="float32"))
        if not vectors or dims is None:
            logger.warning("No vectors produced")
            INDEX = None
            return False
        mat = np.vstack(vectors).astype("float32")
        # normalize rows
        row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
        mat = mat / (row_norms + 1e-12)
        try:
            if use_faiss:
                index = faiss.IndexFlatIP(dims)
                index.add(mat)
                INDEX = index
                try:
                    ensure_parent(FAISS_INDEX_PATH)
                    faiss.write_index(INDEX, FAISS_INDEX_PATH)
                    write_json_atomic(FAISS_MAPPING_PATH, MAPPING)
                except Exception:
                    logger.exception("Failed to persist FAISS index/mapping")
                logger.info("Built FAISS index dims=%d n=%d", dims, index.ntotal)
                return True
            else:
                idx = NumpyFallbackIndex(mat)
                INDEX = idx
                try:
                    ensure_parent(FALLBACK_VECTORS_PATH)
                    idx.save(FALLBACK_VECTORS_PATH)
                    write_json_atomic(FAISS_MAPPING_PATH, MAPPING)
                except Exception:
                    logger.exception("Failed to persist fallback vectors/mapping")
                logger.info("Built fallback index dims=%d n=%d", dims, idx.ntotal)
                return True
        except Exception:
            logger.exception("Index build failed")
            INDEX = None
            return False

# ---------- querying ----------
def query_index(query: str, top_k: int = TOP_K) -> List[Tuple[float, Dict[str,Any]]]:
    global INDEX, MAPPING
    if not query:
        return []
    if INDEX is None:
        built = build_index(force_rebuild=False)
        if not built:
            logger.warning("Index not ready")
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    try:
        if HAS_FAISS and FAISS_ENABLED_ENV and isinstance(INDEX, type(faiss.IndexFlatIP(1))):
            D, I = INDEX.search(vec, top_k)
        else:
            D, I = INDEX.search(vec, top_k)
    except Exception:
        logger.exception("Index query failed")
        return []
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(MAPPING):
            continue
        results.append((float(score), MAPPING[idx]))
    return results

# ---------- prompt ----------
def compose_system_prompt(top_passages: List[Tuple[float, dict]]) -> str:
    header = ("Bạn là trợ lý Ruby Wings — chuyên tư vấn du lịch trải nghiệm, retreat. Trả lời ngắn gọn, lịch sự, tone Ruby Wings.\n\n")
    if not top_passages:
        return header + "Không tìm thấy dữ liệu nội bộ phù hợp."
    content = header + "Dữ liệu nội bộ (theo độ liên quan):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"
    content += "\n---\nƯu tiên trích dẫn dữ liệu nội bộ ở trên. Không bịa đặt. Trả lời ngắn, lịch sự."
    return content

# ---------- endpoints ----------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Ruby Wings index backend running",
        "knowledge_count": len(FLATTENED_TEXTS),
        "index_exists": INDEX is not None,
        "faiss_available": HAS_FAISS,
        "faiss_enabled_env": FAISS_ENABLED_ENV
    })

@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        secret = request.headers.get("X-RBW-ADMIN", "")
        allow_env = os.environ.get("RBW_ALLOW_REINDEX", RBW_ALLOW_REINDEX)
        if not secret and allow_env != "1":
            return jsonify({"error": "reindex not allowed (set RBW_ALLOW_REINDEX=1 or provide X-RBW-ADMIN)"}), 403
        if secret and RBW_ADMIN_TOKEN and secret != RBW_ADMIN_TOKEN:
            return jsonify({"error": "invalid admin token"}), 403
        load_knowledge(KNOWLEDGE_PATH)
        ok = build_index(force_rebuild=True)
        return jsonify({"ok": ok, "count": len(FLATTENED_TEXTS)})
    except Exception as e:
        logger.exception("Unhandled error in /reindex")
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True) or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Bạn chưa nhập câu hỏi."}), 400
        top_k = int(data.get("top_k", TOP_K))
        top = query_index(user_message, top_k)
        system_prompt = compose_system_prompt(top)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        logger.info("CHAT: hits=%d model=%s", len(top), CHAT_MODEL)
        resp_text = ""
        if OPENAI_API_KEY and openai is not None:
            try:
                if hasattr(openai, "ChatCompletion"):
                    res = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages, temperature=0.2, max_tokens=int(data.get("max_tokens",700)))
                else:
                    res = openai.chat.completions.create(model=CHAT_MODEL, messages=messages)
                if isinstance(res, dict):
                    choices = res.get("choices", [])
                    if choices:
                        first = choices[0]
                        if isinstance(first.get("message"), dict):
                            resp_text = first["message"].get("content","") or ""
                        else:
                            resp_text = first.get("text","") or ""
                else:
                    choices = getattr(res, "choices", None)
                    if choices and len(choices) > 0:
                        first = choices[0]
                        msg = getattr(first, "message", None)
                        if isinstance(msg, dict):
                            resp_text = msg.get("content","") or ""
                        else:
                            resp_text = str(first)
            except Exception:
                logger.exception("OpenAI chat failed; falling back to snippet reply")
        if not resp_text:
            if top:
                snippets = "\n\n".join([f"- ({m.get('path')}) {m.get('text')[:300]}" for _, m in top[:5]])
                resp_text = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}\n\nNếu cần trích dẫn cụ thể, hãy yêu cầu phần nào."
            else:
                resp_text = "Xin lỗi — hiện không có dữ liệu nội bộ phù hợp và API OpenAI chưa sẵn sàng."
        sources = [m for _, m in top]
        return jsonify({"reply": resp_text, "sources": sources})
    except Exception as e:
        logger.exception("Unhandled error in /chat")
        return jsonify({"error": str(e)}), 500

@app.route("/stream", methods=["POST"])
def stream():
    try:
        data = request.get_json(force=True) or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "empty message"}), 400
        top_k = int(data.get("top_k", TOP_K))
        top = query_index(user_message, top_k)
        system_prompt = compose_system_prompt(top)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        def gen():
            try:
                if not OPENAI_API_KEY or openai is None:
                    yield f"data: {json.dumps({'error':'openai_key_missing'})}\n\n"
                    return
                try:
                    stream_iter = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages, stream=True)
                except Exception:
                    try:
                        stream_iter = openai.chat.completions.create(model=CHAT_MODEL, messages=messages, stream=True)
                    except Exception:
                        yield f"data: {json.dumps({'error':'openai_stream_failed'})}\n\n"
                        return
                for chunk in stream_iter:
                    try:
                        if isinstance(chunk, dict):
                            choices = chunk.get("choices", [])
                            if choices and len(choices)>0:
                                delta = choices[0].get("delta", {})
                                content = delta.get("content","")
                                if content:
                                    yield f"data: {json.dumps({'delta': content})}\n\n"
                        else:
                            choices = getattr(chunk, "choices", None)
                            if choices and len(choices)>0:
                                delta = getattr(choices[0], "delta", None)
                                if delta and isinstance(delta, dict):
                                    content = delta.get("content","")
                                    if content:
                                        yield f"data: {json.dumps({'delta': content})}\n\n"
                    except Exception:
                        logger.exception("stream chunk error")
                        continue
                yield f"data: {json.dumps({'done': True, 'sources': [m for _, m in top]})}\n\n"
            except Exception as ex:
                logger.exception("stream generator error")
                yield f"data: {json.dumps({'error': str(ex)})}\n\n"
        return Response(stream_with_context(gen()), mimetype="text/event-stream")
    except Exception as e:
        logger.exception("Unhandled error in /stream")
        return jsonify({"error": str(e)}), 500

# ---------- init ----------
try:
    load_knowledge(KNOWLEDGE_PATH)
    if os.path.exists(FAISS_MAPPING_PATH):
        load_mapping(FAISS_MAPPING_PATH)
    # try persistent index load
    if FAISS_ENABLED_ENV and HAS_FAISS and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
        try:
            INDEX = faiss.read_index(FAISS_INDEX_PATH)
            logger.info("Loaded FAISS index at startup")
        except Exception:
            logger.exception("Failed to load FAISS index at startup; scheduling rebuild")
            threading.Thread(target=build_index, kwargs={"force_rebuild": True}, daemon=True).start()
    elif (not FAISS_ENABLED_ENV or not HAS_FAISS) and os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
        try:
            INDEX = NumpyFallbackIndex.load(FALLBACK_VECTORS_PATH)
            logger.info("Loaded fallback vectors at startup")
        except Exception:
            logger.exception("Failed to load fallback vectors; scheduling rebuild")
            threading.Thread(target=build_index, kwargs={"force_rebuild": True}, daemon=True).start()
    else:
        threading.Thread(target=build_index, kwargs={"force_rebuild": False}, daemon=True).start()
except Exception:
    logger.exception("Initialization error")

# ---------- run ----------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", "10000"))
    logger.info("Starting app on port %d", port)
    app.run(host="0.0.0.0", port=port)
