#!/usr/bin/env python3
# app.py — Integrated version with intent classifier, coreference, clarifier, session store & metrics.
# This file is intended to be a drop-in replacement (chép đè) for your existing app.py.
#
# Requirements:
#   - The repository should contain: knowledge.json, entities.py, build_index.py
#   - Optional modules (if present) will be used: intent_classifier.py, coref.py, clarifier.py, session_store.py, metrics.py
#   - Install requirements: pip install -r requirements.txt
#
# Start:
#   python app.py    # dev
#   or via gunicorn in render.yaml

import os
import json
import threading
import logging
import re
import unicodedata
import uuid
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Tuple, Dict, Optional, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np

# Optional heavy deps
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

# OpenAI new client (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# Optional helper modules (stubs or real)
try:
    from entities import build_entity_index, load_entity_index, find_tours_by_place, ENTITY_PATH_DEFAULT
except Exception:
    build_entity_index = None
    load_entity_index = None
    find_tours_by_place = None
    ENTITY_PATH_DEFAULT = os.environ.get("TOUR_ENTITIES_PATH", "tour_entities.json")

try:
    from intent_classifier import detect_field
except Exception:
    detect_field = None

try:
    from coref import resolve_pronoun
except Exception:
    resolve_pronoun = None

try:
    from clarifier import need_clarify, build_clarify_question
except Exception:
    need_clarify = None
    build_clarify_question = None

try:
    from session_store import load_session as redis_load_session, save_session as redis_save_session, get_redis
except Exception:
    redis_load_session = None
    redis_save_session = None
    get_redis = None

try:
    from metrics import incr as metrics_incr, get_metrics
except Exception:
    metrics_incr = None
    get_metrics = None

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw")

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = None
if OPENAI_API_KEY and OpenAI is not None:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    if not OPENAI_API_KEY:
        logger.info("OPENAI_API_KEY not set - LLM features disabled.")
    else:
        logger.warning("OpenAI client not available - LLM features disabled.")

KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

# Session and timeout
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "300"))  # seconds

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Global state ----------
KNOW: Dict = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []
INDEX = None
INDEX_LOCK = threading.Lock()

TOUR_NAME_TO_INDEX: Dict[str, int] = {}
ENTITY_INDEX: Dict[str, dict] = {}

# In-memory fallback session store when Redis not configured
USER_SESSIONS: Dict[str, Dict[str, Any]] = {}

# Keyword -> field mapping (fallback)
KEYWORD_FIELD_MAP: Dict[str, Dict] = {
    "tour_list": {
        "keywords": [
            "tên tour", "tour gì", "danh sách tour", "có những tour nào", "liệt kê tour",
            "show tour", "tour hiện có", "tour available", "liệt kê các tour đang có",
            "list tour", "tour đang bán", "tour hiện hành", "tour nào", "tours", "liệt kê các tour",
            "liệt kê các hành trình", "list tours", "show tours", "các tour hiện tại"
        ],
        "field": "tour_name"
    },
    "mission": {"keywords": ["tầm nhìn", "sứ mệnh", "giá trị cốt lõi", "triết lý", "vision", "mission"], "field": "mission"},
    "summary": {"keywords": ["tóm tắt chương trình tour", "tóm tắt", "overview", "brief", "mô tả ngắn"], "field": "summary"},
    "style": {"keywords": ["phong cách hành trình", "tính chất hành trình", "concept tour", "vibe tour", "style"], "field": "style"},
    "transport": {"keywords": ["vận chuyển", "phương tiện", "di chuyển", "xe gì", "transportation"], "field": "transport"},
    "includes": {"keywords": ["lịch trình chi tiết", "chương trình chi tiết", "chi tiết hành trình", "itinerary", "schedule", "includes"], "field": "includes"},
    "location": {"keywords": ["ở đâu", "đi đâu", "địa phương nào", "nơi nào", "điểm đến", "destination", "location"], "field": "location"},
    "duration": {"keywords": ["thời gian tour", "kéo dài", "mấy ngày", "bao lâu", "ngày đêm", "duration", "tour dài bao lâu", "tour bao nhiêu ngày", "2 ngày 1 đêm", "3 ngày 2 đêm"], "field": "duration"},
    "price": {"keywords": ["giá tour", "chi phí", "bao nhiêu tiền", "price", "cost"], "field": "price"},
    "notes": {"keywords": ["lưu ý", "ghi chú", "notes", "cần chú ý"], "field": "notes"},
    "accommodation": {"keywords": ["chỗ ở", "nơi lưu trú", "khách sạn", "homestay", "accommodation"], "field": "accommodation"},
    "meals": {"keywords": ["ăn uống", "ẩm thực", "meals", "thực đơn", "bữa"], "field": "meals"},
    "event_support": {"keywords": ["hỗ trợ", "dịch vụ hỗ trợ", "event support", "dịch vụ tăng cường"], "field": "event_support"},
    "cancellation_policy": {"keywords": ["phí huỷ", "chính sách huỷ", "cancellation", "refund policy"], "field": "cancellation_policy"},
    "booking_method": {"keywords": ["đặt chỗ", "đặt tour", "booking", "cách đặt"], "field": "booking_method"},
    "who_can_join": {"keywords": ["phù hợp đối tượng", "ai tham gia", "who should join"], "field": "who_can_join"},
    "hotline": {"keywords": ["hotline", "số điện thoại", "liên hệ", "contact number"], "field": "hotline"},
}

# ---------- Utilities ----------
def normalize_text_simple(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ---------- Index-tour-name helpers ----------
def index_tour_names():
    global TOUR_NAME_TO_INDEX
    TOUR_NAME_TO_INDEX = {}
    for m in MAPPING:
        path = m.get("path", "")
        if path.endswith(".tour_name"):
            txt = m.get("text", "") or ""
            norm = normalize_text_simple(txt)
            if not norm:
                continue
            match = re.search(r"\[(\d+)\]", path)
            if match:
                idx = int(match.group(1))
                prev = TOUR_NAME_TO_INDEX.get(norm)
                if prev is None:
                    TOUR_NAME_TO_INDEX[norm] = idx
                else:
                    # prefer longer name
                    try:
                        prev_text = MAPPING[next(i for i,m2 in enumerate(MAPPING) if re.search(rf"\[{prev}\]", m2.get('path','')) )].get("text","")
                        if len(txt) > len(prev_text):
                            TOUR_NAME_TO_INDEX[norm] = idx
                    except Exception:
                        TOUR_NAME_TO_INDEX[norm] = idx

def find_tour_indices_from_message(message: str) -> List[int]:
    if not message:
        return []
    msg_n = normalize_text_simple(message)
    if not msg_n:
        return []
    matches = []
    for norm_name, idx in TOUR_NAME_TO_INDEX.items():
        tour_words = set(norm_name.split())
        msg_words = set(msg_n.split())
        common_words = tour_words & msg_words
        if len(common_words) >= 1:
            matches.append((len(common_words), norm_name))
    if matches:
        matches.sort(reverse=True)
        best_score = matches[0][0]
        selected = [TOUR_NAME_TO_INDEX[nm] for sc, nm in matches if sc == best_score]
        return sorted(set(selected))
    return []

# ---------- MAPPING helpers ----------
def get_passages_by_field(field_name: str, limit: int = 50, tour_indices: Optional[List[int]] = None) -> List[Tuple[float, dict]]:
    exact_matches: List[Tuple[float, dict]] = []
    global_matches: List[Tuple[float, dict]] = []
    for m in MAPPING:
        path = m.get("path", "")
        if path.endswith(f".{field_name}") or f".{field_name}" in path:
            is_exact = False
            if tour_indices:
                for ti in tour_indices:
                    if f"[{ti}]" in path:
                        is_exact = True
                        break
            if is_exact:
                exact_matches.append((2.0, m))
            elif not tour_indices:
                global_matches.append((1.0, m))
    all_results = exact_matches + global_matches
    all_results.sort(key=lambda x: x[0], reverse=True)
    return all_results[:limit]

def find_hotline_for_tour(ti: int) -> Optional[str]:
    for m in MAPPING:
        p = m.get("path","")
        if ti >= 0 and p.endswith(f"tours[{ti}].hotline"):
            return m.get("text")
    # global hotline
    for m in MAPPING:
        p = m.get("path","")
        if p.endswith(".hotline") or p.endswith(".contact"):
            return m.get("text")
    return None

# ---------- Embeddings & Index ----------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    if client is not None:
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=short)
            if resp.data and len(resp.data) > 0:
                emb = resp.data[0].embedding
                return emb, len(emb)
        except Exception:
            logger.exception("OpenAI embedding failed; falling back.")
    # deterministic fallback
    h = abs(hash(short)) % (10 ** 12)
    fallback_dim = 1536
    vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(fallback_dim)]
    return vec, fallback_dim

class NumpyIndex:
    def __init__(self, mat: Optional[np.ndarray] = None):
        if mat is None or getattr(mat, "size", 0) == 0:
            self.mat = np.empty((0, 0), dtype="float32")
            self.dim = None
        else:
            self.mat = mat.astype("float32")
            self.dim = self.mat.shape[1]
    def add(self, mat: np.ndarray):
        if getattr(mat, "size", 0) == 0:
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
        if self.mat is None or getattr(self.mat, "size", 0) == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        q = qvec.astype("float32")
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        m = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        sims = np.dot(q, m.T)
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype("float32"), idx.astype("int64")
    @property
    def ntotal(self):
        return 0 if getattr(self.mat, "size", 0) == 0 else self.mat.shape[0]
    def save(self, path):
        try:
            np.savez_compressed(path, mat=self.mat)
        except Exception:
            logger.exception("Failed to save fallback vectors")
    @classmethod
    def load(cls, path):
        try:
            arr = np.load(path)
            mat = arr["mat"]
            return cls(mat=mat)
        except Exception:
            logger.exception("Failed to load fallback vectors")
            return cls(None)

def _index_dim(idx) -> Optional[int]:
    try:
        d = getattr(idx, "d", None)
        if isinstance(d, int) and d > 0:
            return d
    except Exception:
        pass
    try:
        d = getattr(idx, "dim", None)
        if isinstance(d, int) and d > 0:
            return d
    except Exception:
        pass
    try:
        if HAS_FAISS and isinstance(idx, faiss.Index):
            return int(idx.d)
    except Exception:
        pass
    return None

def choose_embedding_model_for_dim(dim: int) -> str:
    if dim == 1536:
        return "text-embedding-3-small"
    if dim == 3072:
        return "text-embedding-3-large"
    return os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)

def build_index(force_rebuild: bool = False) -> bool:
    global INDEX, MAPPING, FLAT_TEXTS, EMBEDDING_MODEL
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED and HAS_FAISS
        if not FLAT_TEXTS:
            logger.warning("No texts to index")
            INDEX = None
            return False
        vectors = []
        dims = None
        for text in FLAT_TEXTS:
            emb, d = embed_text(text)
            if not emb:
                continue
            if dims is None:
                dims = d
            vectors.append(np.array(emb, dtype="float32"))
        if not vectors or dims is None:
            logger.warning("No vectors")
            INDEX = None
            return False
        try:
            mat = np.vstack(vectors).astype("float32")
            row_norms = np.linalg.norm(mat, axis=1, keepdims=True)
            mat = mat / (row_norms + 1e-12)
            if use_faiss:
                index = faiss.IndexFlatIP(dims)
                index.add(mat)
                INDEX = index
                try:
                    faiss.write_index(INDEX, FAISS_INDEX_PATH)
                    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                        json.dump(MAPPING, f, ensure_ascii=False, indent=2)
                except Exception:
                    logger.exception("Persist failed")
                index_tour_names()
                logger.info("FAISS index built")
                return True
            else:
                idx = NumpyIndex(mat)
                INDEX = idx
                try:
                    idx.save(FALLBACK_VECTORS_PATH)
                    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                        json.dump(MAPPING, f, ensure_ascii=False, indent=2)
                except Exception:
                    logger.exception("Save mapping failed")
                index_tour_names()
                logger.info("Numpy index built")
                return True
        except Exception:
            logger.exception("Error building index")
            INDEX = None
            return False

def query_index(query: str, top_k: int = TOP_K) -> List[Tuple[float, dict]]:
    global INDEX
    if not query:
        return []
    if INDEX is None:
        built = build_index(force_rebuild=False)
        if not built:
            logger.warning("Index not available")
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    idx_dim = _index_dim(INDEX)
    if idx_dim and vec.shape[1] != idx_dim:
        logger.info("Index dim mismatch - trying rebuild")
        desired_model = choose_embedding_model_for_dim(idx_dim)
        if OPENAI_API_KEY:
            global EMBEDDING_MODEL
            EMBEDDING_MODEL = desired_model
            rebuilt = build_index(force_rebuild=True)
            if not rebuilt:
                logger.error("Rebuild failed")
                return []
            emb2, d2 = embed_text(query)
            if not emb2:
                return []
            vec = np.array(emb2, dtype="float32").reshape(1, -1)
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        else:
            logger.error("No OPENAI_API_KEY, cannot rebuild")
            return []
    try:
        D, I = INDEX.search(vec, top_k)
    except Exception:
        logger.exception("Index search error")
        return []
    results = []
    try:
        scores = D[0].tolist() if getattr(D, "shape", None) else []
        idxs = I[0].tolist() if getattr(I, "shape", None) else []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(MAPPING):
                continue
            results.append((float(score), MAPPING[idx]))
    except Exception:
        logger.exception("Failed to parse results")
    return results

# ---------- Knowledge loader ----------
def load_knowledge(path: str = KNOWLEDGE_PATH):
    global KNOW, FLAT_TEXTS, MAPPING
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOW = json.load(f)
    except Exception:
        logger.exception("Could not open knowledge.json; continuing with empty knowledge.")
        KNOW = {}
    FLAT_TEXTS = []
    MAPPING = []
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
                FLAT_TEXTS.append(t)
                MAPPING.append({"path": prefix, "text": t})
        else:
            try:
                s = str(obj).strip()
                if s:
                    FLAT_TEXTS.append(s)
                    MAPPING.append({"path": prefix, "text": s})
            except Exception:
                pass
    scan(KNOW)
    index_tour_names()
    logger.info("Knowledge loaded: %d passages", len(FLAT_TEXTS))

# ---------- Session management (Redis optional) ----------
def get_or_create_session():
    # check cookie ID
    session_id = request.cookies.get('session_id')
    sess = None
    if session_id and redis_load_session is not None:
        try:
            sess = redis_load_session(session_id)
        except Exception:
            sess = None
    if not session_id or (sess is None and session_id not in USER_SESSIONS):
        session_id = str(uuid.uuid4())
        sess = {
            'created_at': datetime.now().isoformat(),
            'last_activity': datetime.now().isoformat(),
            'last_tour_index': None,
            'last_tour_name': None,
            'recent_entities': [],
            'conversation_count': 0
        }
        USER_SESSIONS[session_id] = sess
        logger.info(f"🆕 Created new session: {session_id}")
    # If Redis in use, prefer Redis save/load
    if redis_save_session is not None:
        try:
            redis_save_session(session_id, sess, ttl=SESSION_TIMEOUT)
        except Exception:
            pass
    else:
        USER_SESSIONS[session_id] = sess
    # update last_activity
    sess['last_activity'] = datetime.now().isoformat()
    cleanup_expired_sessions()
    return session_id, sess

def cleanup_expired_sessions():
    expired = []
    now = datetime.utcnow()
    for sid, data in list(USER_SESSIONS.items()):
        try:
            last = datetime.fromisoformat(data.get('last_activity'))
        except Exception:
            last = now
        if (now - last).total_seconds() > SESSION_TIMEOUT:
            expired.append(sid)
    for sid in expired:
        USER_SESSIONS.pop(sid, None)
        logger.info(f"🗑️ Cleaned expired session: {sid}")

def update_session_context(session_data: Dict[str, Any], tour_indices: List[int], new_entity: Optional[int], user_message: str):
    # if new explicit tours, update last_tour
    if tour_indices:
        session_data['last_tour_index'] = tour_indices[0]
        # find tour name
        session_data['last_tour_name'] = None
        for m in MAPPING:
            if f"[{tour_indices[0]}]" in m.get('path','') and m.get('path','').endswith("tour_name"):
                session_data['last_tour_name'] = m.get('text')
                break
        session_data['conversation_count'] = 1
        if 'recent_entities' not in session_data:
            session_data['recent_entities'] = []
        session_data['recent_entities'].append({'type':'tour','id':tour_indices[0], 'ts': datetime.now().isoformat()})
        # trim recent
        session_data['recent_entities'] = session_data['recent_entities'][-10:]
        logger.info(f"🎯 Updated session context to tour: {session_data['last_tour_name']}")
    elif new_entity is not None:
        # resolved from pronoun mapping
        session_data['recent_entities'] = session_data.get('recent_entities', [])
        session_data['recent_entities'].append({'type':'tour','id': new_entity, 'ts': datetime.now().isoformat()})
        session_data['recent_entities'] = session_data['recent_entities'][-10:]
        logger.info(f"🔁 Added resolved entity to session: {new_entity}")
    elif session_data.get('last_tour_index') is not None:
        session_data['conversation_count'] = session_data.get('conversation_count',0) + 1

# ---------- Routes ----------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "knowledge_count": len(FLAT_TEXTS),
        "index_exists": INDEX is not None,
        "index_dim": _index_dim(INDEX),
        "embedding_model": EMBEDDING_MODEL,
        "faiss_available": HAS_FAISS,
        "faiss_enabled": FAISS_ENABLED,
        "active_sessions": len(USER_SESSIONS)
    })

@app.route("/reindex", methods=["POST"])
def reindex():
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
        return jsonify({"error": "reindex not allowed (set RBW_ALLOW_REINDEX=1 or provide X-RBW-ADMIN)"}), 403
    load_knowledge()
    ok = build_index(force_rebuild=True)
    # rebuild entity index as well if available
    if build_entity_index is not None:
        try:
            global ENTITY_INDEX
            ENTITY_INDEX = build_entity_index(MAPPING)
            logger.info("Rebuilt entity index (%d keys)", len(ENTITY_INDEX))
        except Exception:
            logger.exception("Failed to rebuild entity index")
    return jsonify({"ok": ok, "count": len(FLAT_TEXTS)})

@app.route("/metrics", methods=["GET"])
def metrics_endpoint():
    if get_metrics is not None:
        return jsonify(get_metrics())
    return jsonify({"error":"metrics not enabled"}), 404

@app.route("/chat", methods=["POST"])
def chat():
    # metrics
    if metrics_incr is not None:
        try:
            metrics_incr("requests", 1)
        except Exception:
            pass

    session_id, session_data = get_or_create_session()
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply":"Bạn chưa nhập câu hỏi."})

    # 1) Preprocess and optional coref resolution
    resolved_entity = None
    if resolve_pronoun is not None:
        try:
            r = resolve_pronoun(user_message, session_data)
            if r and r.get("resolved_entity") is not None:
                resolved_entity = r.get("resolved_entity")
                # resolved_entity is tour index
                logger.info("Resolved pronoun to entity: %s", resolved_entity)
        except Exception:
            logger.exception("coref resolution failed")

    # 2) Intent/field detection (prefer intent_classifier if present)
    detected_field = None
    detected_conf = 0.0
    if detect_field is not None:
        try:
            f, c = detect_field(user_message)
            detected_field = f
            detected_conf = float(c or 0.0)
        except Exception:
            logger.exception("intent classifier failed")
            detected_field, detected_conf = None, 0.0

    # fallback to keyword map if classifier not available or no result
    if not detected_field:
        text_l = user_message.lower()
        for k, v in KEYWORD_FIELD_MAP.items():
            for kw in v["keywords"]:
                if kw in text_l:
                    detected_field = v["field"]
                    detected_conf = max(detected_conf, 0.6)
                    break
            if detected_field:
                break

    logger.info("User: %s | detected_field: %s (conf=%.2f)", user_message, detected_field, detected_conf)

    # 3) Tour detection: by name, by place (entity index), or resolved_entity
    tour_indices: List[int] = []
    # a) by name match
    try:
        tour_indices = find_tour_indices_from_message(user_message)
    except Exception:
        logger.exception("tour name detection failed")

    # b) if none and resolved_entity exists, use it
    if not tour_indices and resolved_entity is not None:
        tour_indices = [resolved_entity]

    # c) if still none, try place-based entity lookup
    place_used = False
    place_matches = []
    if not tour_indices and find_tours_by_place is not None and ENTITY_INDEX:
        try:
            place_matches = find_tours_by_place(user_message, ENTITY_INDEX, top_k=5, fuzzy_threshold=75, semantic_fallback_fn=query_index)
            if place_matches:
                tour_indices = [pm[0] for pm in place_matches]
                place_used = True
                logger.info("Detected place-based tours: %s", tour_indices)
                if metrics_incr is not None:
                    try:
                        metrics_incr("entity_matches", 1)
                    except Exception:
                        pass
        except Exception:
            logger.exception("Place lookup failed")

    # If still none and session context exists, use it
    if not tour_indices and session_data.get('last_tour_index') is not None:
        tour_indices = [session_data['last_tour_index']]
        logger.info("Using session context tour: %s", session_data.get('last_tour_name'))

    # 4) Clarify if ambiguous
    clarify = False
    if need_clarify is not None:
        try:
            clarify = need_clarify(detected_field, tour_indices, detected_conf)
        except Exception:
            clarify = False
    else:
        # fallback: if no field detected and multiple tours -> clarify
        if not detected_field and len(tour_indices) > 1:
            clarify = True
        # if field detected but confidence low and multiple tours -> clarify
        if detected_field and detected_conf < 0.6 and len(tour_indices) > 1:
            clarify = True

    if clarify:
        # build candidate list for question
        candidates = []
        for ti in tour_indices[:5]:
            name = None
            for m in MAPPING:
                if m.get("path","").endswith(f"tours[{ti}].tour_name"):
                    name = m.get("text")
                    break
            candidates.append({"tour_index": ti, "tour_name": name})
        question = None
        if build_clarify_question is not None:
            try:
                question = build_clarify_question(candidates)
            except Exception:
                question = None
        if not question:
            pieces = [f"{i+1}) {c['tour_name'] or 'Tour#'+str(c['tour_index'])}" for i,c in enumerate(candidates)]
            question = "Bạn đang nói đến tour nào? " + " / ".join(pieces)
        if metrics_incr is not None:
            try:
                metrics_incr("clarify_count", 1)
            except Exception:
                pass
        return jsonify({"reply": question, "clarify": True, "candidates": candidates})

    # 5) Update session context with tour_indices or resolved entity
    update_session_context(session_data, tour_indices, resolved_entity, user_message)
    # persist session if redis available
    if redis_save_session is not None:
        try:
            redis_save_session(session_id, session_data, ttl=SESSION_TIMEOUT)
        except Exception:
            pass

    # 6) Deterministic field-first flow (if detected_field present)
    top_results: List[Tuple[float, dict]] = []
    structured = None
    reply = ""
    used_deterministic = False

    if detected_field:
        # If user asked for list of tours
        if detected_field == "tour_name":
            top_results = get_passages_by_field("tour_name", limit=1000, tour_indices=None)
            names = [m.get("text","") for _, m in top_results]
            seen = set()
            names_u = [x for x in names if x and not (x in seen or seen.add(x))]
            reply = "Các tour hiện có:\n" + "\n".join(f"- {n}" for n in names_u)
            structured = {"type":"tour_list", "tour_names": names_u}
            used_deterministic = True
            if metrics_incr is not None:
                try:
                    metrics_incr("deterministic_hits", 1)
                except Exception:
                    pass
        else:
            # If tour specified -> get field per tour
            if tour_indices:
                parts = []
                missing_tours = []
                for ti in tour_indices:
                    passages = get_passages_by_field(detected_field, limit=TOP_K, tour_indices=[ti])
                    if passages:
                        vals = [m.get("text","") for _, m in passages]
                        tour_name = None
                        for m in MAPPING:
                            if m.get("path","").endswith(f"tours[{ti}].tour_name"):
                                tour_name = m.get("text")
                                break
                        parts.append({"tour_index":ti, "tour_name": tour_name, "field": detected_field, "values": vals, "sources":[m.get("path") for _,m in passages]})
                    else:
                        missing_tours.append(ti)
                if parts:
                    lines = []
                    for p in parts:
                        label = f'Tour "{p["tour_name"]}"' if p["tour_name"] else f"Tour #{p['tour_index']}"
                        vals_text = "\n".join(f"- {v}" for v in p["values"])
                        lines.append(f"{label} — {detected_field}:\n{vals_text}")
                    reply = "\n\n".join(lines)
                    structured = {"type":"field_values_by_tour", "field":detected_field, "data": parts}
                    used_deterministic = True
                    if metrics_incr is not None:
                        try:
                            metrics_incr("deterministic_hits", 1)
                        except Exception:
                            pass
                if missing_tours and not parts:
                    # fallback to global or contact
                    global_matches = get_passages_by_field(detected_field, limit=TOP_K, tour_indices=None)
                    if global_matches:
                        vals = [m.get("text","") for _,m in global_matches]
                        reply = f"Tôi không tìm thấy '{detected_field}' cho tour bạn hỏi, nhưng có thông tin chung:\n" + "\n".join(f"- {v}" for v in vals)
                        structured = {"type":"field_global_fallback", "field":detected_field, "values": vals, "sources":[m.get("path") for _,m in global_matches]}
                        used_deterministic = True
                    else:
                        hotline = None
                        if len(missing_tours)==1:
                            hotline = find_hotline_for_tour(missing_tours[0])
                        if not hotline:
                            hotline = find_hotline_for_tour(-1)
                        contact_txt = f" Liên hệ nhân viên: {hotline}" if hotline else " Liên hệ nhân viên để biết thêm chi tiết."
                        reply = f"Xin lỗi — hồ sơ tour không có thông tin '{detected_field}'.{contact_txt}"
                        structured = {"type":"field_missing", "field":detected_field, "tour_indices": missing_tours, "contact": hotline}
                        used_deterministic = True
            else:
                # No specific tour: global matches
                global_matches = get_passages_by_field(detected_field, limit=TOP_K, tour_indices=None)
                if global_matches:
                    vals = [m.get("text","") for _,m in global_matches]
                    reply = f"Thông tin '{detected_field}' (tổng quát):\n" + "\n".join(f"- {v}" for v in vals)
                    structured = {"type":"field_global", "field":detected_field, "values": vals, "sources":[m.get("path") for _,m in global_matches]}
                    used_deterministic = True
                    if metrics_incr is not None:
                        try:
                            metrics_incr("deterministic_hits", 1)
                        except Exception:
                            pass
                else:
                    # semantic fallback
                    top_results = query_index(user_message, TOP_K)
                    if top_results:
                        snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                        reply = f"Tôi tìm thấy thông tin liên quan (semantic fallback):\n\n{snippets}"
                        structured = {"type":"semantic_fallback", "field":detected_field, "snippets":[m.get("text") for _,m in top_results[:5]], "sources":[m.get("path") for _,m in top_results[:5]]}
                        used_deterministic = True
                    else:
                        hotline = find_hotline_for_tour(-1)
                        contact_txt = f" Liên hệ nhân viên: {hotline}" if hotline else " Liên hệ nhân viên để biết thêm."
                        reply = f"Xin lỗi — không có thông tin '{detected_field}' trong hệ thống hiện tại.{contact_txt}"
                        structured = {"type":"field_missing_global", "field":detected_field, "contact": hotline}
                        used_deterministic = True

    # 7) If not deterministic, fallback to semantic search + LLM
    if not used_deterministic:
        top_k = int(data.get("top_k", TOP_K))
        top_results = query_index(user_message, top_k)
        if client is not None and top_results:
            system_prompt = "Bạn là trợ lý, CHỈ dùng DỮ LIỆU NỘI BỘ bên dưới để trả lời. Nếu không có trong dữ liệu, nói 'Không có thông tin'. Trả ngắn gọn và kèm nguồn (path).\n\nDỮ LIỆU NỘI BỘ:\n"
            for i, (score, m) in enumerate(top_results, start=1):
                system_prompt += f"[{i}] (score={score:.3f}) {m.get('path')}\n{m.get('text')}\n\n"
            messages = [{"role":"system","content":system_prompt},{"role":"user","content":user_message}]
            try:
                resp = client.chat.completions.create(
                    model=CHAT_MODEL,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=int(data.get("max_tokens", 400)),
                )
                if resp.choices and len(resp.choices)>0:
                    reply = resp.choices[0].message.content or ""
                    if metrics_incr is not None:
                        try:
                            metrics_incr("llm_hits", 1)
                        except Exception:
                            pass
            except Exception:
                logger.exception("OpenAI chat failed")
        if not reply:
            if top_results:
                snippets = "\n\n".join([f"- ({m.get('path')}) {m.get('text')}" for _, m in top_results[:5]])
                reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}"
            else:
                reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan. Bạn muốn mình chuyển sang nhân viên hỗ trợ?"

    # 8) Prepare response
    response_payload = {
        "reply": reply,
        "structured": structured,
        "sources": [m for _, m in (top_results if 'top_results' in locals() and top_results else [])] if ('top_results' in locals() and top_results) else (structured.get("data") if structured and structured.get("data") else []),
        "context_tour": session_data.get('last_tour_name'),
        "session_active": session_data.get('last_tour_name') is not None
    }

    # Save session
    if redis_save_session is not None:
        try:
            redis_save_session(session_id, session_data, ttl=SESSION_TIMEOUT)
        except Exception:
            pass
    else:
        USER_SESSIONS[session_id] = session_data

    resp = jsonify(response_payload)
    resp.set_cookie('session_id', session_id, max_age=SESSION_TIMEOUT, httponly=True)
    logger.info("Response prepared | deterministic=%s | field=%s | tours=%s | place_used=%s", used_deterministic, detected_field, tour_indices, place_used)
    return resp

# ---------- Initialization ----------
try:
    load_knowledge()
    # try loading existing mapping file into MAPPING if present
    if os.path.exists(FAISS_MAPPING_PATH):
        try:
            with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                file_map = json.load(f)
            if file_map and (len(file_map) == len(MAPPING) or len(MAPPING) == 0):
                MAPPING[:] = file_map
                FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                index_tour_names()
                logger.info("Mapping loaded from %s", FAISS_MAPPING_PATH)
        except Exception:
            logger.exception("Could not load FAISS mapping at startup")
    # Build/load entity index
    try:
        if load_entity_index is not None and os.path.exists(ENTITY_PATH_DEFAULT):
            ENTITY_INDEX = load_entity_index(ENTITY_PATH_DEFAULT)
            logger.info("Loaded entity index from %s (%d keys)", ENTITY_PATH_DEFAULT, len(ENTITY_INDEX))
        elif build_entity_index is not None:
            ENTITY_INDEX = build_entity_index(MAPPING)
            logger.info("Built entity index from mapping (%d keys)", len(ENTITY_INDEX))
        else:
            logger.info("entities module not available; skipping entity index")
    except Exception:
        logger.exception("Failed to build/load entity index")
    # start index build in background
    t = threading.Thread(target=build_index, kwargs={"force_rebuild": False}, daemon=True)
    t.start()
except Exception:
    logger.exception("Initialization error")

if __name__ == "__main__":
    # ensure mapping saved
    if MAPPING and not os.path.exists(FAISS_MAPPING_PATH):
        try:
            with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                json.dump(MAPPING, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    built = build_index(force_rebuild=False)
    if not built:
        logger.warning("Index not ready at startup; endpoint will attempt on-demand build.")
    # dev server
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))