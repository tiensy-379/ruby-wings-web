#!/usr/bin/env python3
"""
app.py — BẢN "HOÀN HẢO NHẤT" cho Ruby Wings chatbot
- Tích hợp: session current_tour, fuzzy tour NER, two-stage retrieval (deterministic tour-field first),
  two-tier reranking (semantic -> lexical), deterministic fallbacks, logging trace.
- Tương thích với build_index.py (mapping: list of {"path","text","field","tour_index","tour_name"})
- ENV vars:
    OPENAI_API_KEY, EMBEDDING_MODEL, CHAT_MODEL, FAISS_INDEX_PATH, FAISS_MAPPING_PATH,
    FALLBACK_VECTORS_PATH, FAISS_ENABLED, SESSION_STORE ("memory" or "redis"), REDIS_URL

Lưu ý: file này bao gồm các giải pháp fallback (không phụ thuộc bắt buộc vào OpenAI hay faiss).
"""

import os
import json
import re
import unicodedata
import threading
import logging
import uuid
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Optional, Tuple
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import difflib

# Optional: redis session store
try:
    import redis
except Exception:
    redis = None

# Optional FAISS
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# OpenAI new SDK
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw_app_perfect")

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")
TOP_K = int(os.environ.get("TOP_K", "5"))
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", str(60 * 5)))  # seconds
SESSION_STORE = os.environ.get("SESSION_STORE", "memory")  # or 'redis'
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

# Initialize OpenAI client if possible
client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized")
    except Exception:
        logger.exception("OpenAI client init failed")
else:
    logger.info("OpenAI client not available or OPENAI_API_KEY missing; falling back to deterministic behavior")

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app)

# ---------- Global state ----------
KNOW: Dict = {}
MAPPING: List[dict] = []  # expected stable-ordered mapping produced by build_index.py
FLAT_TEXTS: List[str] = []
INDEX = None
INDEX_LOCK = threading.Lock()

# Tour name normalized -> index map
TOUR_NAME_TO_INDEX: Dict[str, int] = {}

# Session backend (in-memory fallback)
USER_SESSIONS: Dict[str, dict] = {}
if SESSION_STORE == "redis" and redis is not None:
    try:
        REDIS_CLIENT = redis.from_url(REDIS_URL)
        logger.info("Using redis session store %s", REDIS_URL)
    except Exception:
        logger.exception("Redis init failed; falling back to memory store")
        REDIS_CLIENT = None
else:
    REDIS_CLIENT = None

# ---------- Field keywords (expanded to 16 fields) ----------
# Each entry: 'field': set([...keywords...])
KEYWORD_FIELD_MAP = {
    "tour_name": ["tên tour", "tour gì", "danh sách tour", "liệt kê tour", "tour nào", "list tour", "tours", "show tour"],
    "summary": ["tóm tắt", "tóm tắt chương trình", "overview", "mô tả ngắn", "brief"],
    "location": ["đi đâu", "địa điểm", "điểm đến", "location", "đi đâu", "ở đâu"],
    "duration": ["thời gian", "kéo dài", "mấy ngày", "bao lâu", "ngày đêm", "duration"],
    "price": ["giá", "giá tour", "giá vé", "bao nhiêu tiền", "chi phí", "cost", "price"],
    "includes": ["lịch trình chi tiết", "chương trình chi tiết", "includes", "itinerary", "schedule"],
    "notes": ["lưu ý", "ghi chú", "notes", "lưu ý"],
    "style": ["phong cách", "style", "tính chất hành trình", "vibe"],
    "transport": ["vận chuyển", "phương tiện", "xe", "di chuyển", "transportation"],
    "accommodation": ["chỗ ở", "khách sạn", "homestay", "accommodation"],
    "meals": ["ăn uống", "ăn gì", "thực đơn", "bữa", "meals"],
    "event_support": ["hỗ trợ sự kiện", "hỗ trợ đoàn", "event support", "hỗ trợ"],
    "hotline": ["hotline", "số điện thoại", "liên hệ", "contact"],
    "mission": ["sứ mệnh", "mission", "tầm nhìn"],
    "includes_extra": ["bao gồm", "include", "gồm có"],
    "extras": ["dịch vụ thêm", "option", "tùy chọn", "extra"]
}
# normalize keywords to lowercase plain
for k, lst in list(KEYWORD_FIELD_MAP.items()):
    KEYWORD_FIELD_MAP[k] = list({s.lower(): None for s in lst}.keys())

# Map synonyms to canonical 16-field names (final fields expected in knowledge.json)
CANONICAL_FIELD_MAP = {
    "tour_name": "tour_name",
    "summary": "summary",
    "location": "location",
    "duration": "duration",
    "price": "price",
    "includes": "includes",
    "includes_extra": "includes",
    "notes": "notes",
    "style": "style",
    "transport": "transport",
    "accommodation": "accommodation",
    "meals": "meals",
    "event_support": "event_support",
    "hotline": "hotline",
    "mission": "mission",
    "extras": "notes"
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


def token_set(s: str) -> set:
    return set(normalize_text_simple(s).split()) if s else set()


def sequence_similarity(a: str, b: str) -> float:
    # use difflib ratio as fallback for fuzzy string similarity
    try:
        return difflib.SequenceMatcher(None, a, b).ratio()
    except Exception:
        return 0.0


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = a & b
    uni = a | b
    return len(inter) / max(1, len(uni))


def levenshtein_ratio(a: str, b: str) -> float:
    # lightweight approximation using SequenceMatcher
    return sequence_similarity(a, b)

# ---------- Tour detection (Fuzzy NER) ----------

def index_tour_names():
    """Populate TOUR_NAME_TO_INDEX from MAPPING entries that end with .tour_name.
    Expects mapping entries include optional fields: 'field', 'tour_index', 'tour_name'"""
    global TOUR_NAME_TO_INDEX
    TOUR_NAME_TO_INDEX = {}
    for i, m in enumerate(MAPPING):
        path = m.get("path", "")
        text = m.get("text", "")
        field = m.get("field") or ("tour_name" if path.endswith(".tour_name") or ".tour_name" in path else None)
        if field == "tour_name" or path.endswith(".tour_name"):
            norm = normalize_text_simple(text)
            if norm:
                # keep first occurrence (mapping is stable ordered)
                if norm not in TOUR_NAME_TO_INDEX:
                    # attempt to extract explicit tour_index from path
                    idx = None
                    match = re.search(r"tours\[(\d+)\]", path)
                    if match:
                        try:
                            idx = int(match.group(1))
                        except Exception:
                            idx = None
                    TOUR_NAME_TO_INDEX[norm] = idx if idx is not None else i
    logger.info("Indexed %d tour names", len(TOUR_NAME_TO_INDEX))


def find_tour_indices_from_message(message: str, top_n: int = 3) -> List[int]:
    """Return list of candidate tour indices (ordered by score desc)."""
    msg = normalize_text_simple(message)
    if not msg:
        return []

    msg_tokens = token_set(msg)
    candidates: List[Tuple[float, int]] = []

    # exact substring match first
    for norm_name, idx in TOUR_NAME_TO_INDEX.items():
        if norm_name in msg:
            candidates.append((1.0 + len(token_set(norm_name & msg_tokens)), idx))

    # token overlap + sequence similarity
    if not candidates:
        for norm_name, idx in TOUR_NAME_TO_INDEX.items():
            tset = token_set(norm_name)
            jac = jaccard(tset, msg_tokens)
            seq = sequence_similarity(norm_name, msg)
            score = max(jac, seq * 0.9) + (0.01 * len(tset & msg_tokens))
            # threshold low to capture synonyms/abbrev
            if score > 0.12:
                candidates.append((score, idx))

    # if still empty, consider partial token matches
    if not candidates:
        for norm_name, idx in TOUR_NAME_TO_INDEX.items():
            tset = token_set(norm_name)
            common = tset & msg_tokens
            if common:
                score = len(common) / max(1, len(tset))
                candidates.append((score, idx))

    if not candidates:
        return []

    candidates.sort(key=lambda x: x[0], reverse=True)
    seen = set()
    out = []
    for sc, idx in candidates[:top_n * 2]:
        if idx not in seen:
            seen.add(idx)
            out.append(idx)
    return out[:top_n]

# ---------- Mapping loader / knowledge loader ----------

def load_mapping_from_disk(path: str = FAISS_MAPPING_PATH) -> bool:
    global MAPPING, FLAT_TEXTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            MAPPING = json.load(f)
        # Guarantee minimal structure for each mapping entry
        for i, m in enumerate(MAPPING):
            if "path" not in m:
                m["path"] = f"root.unknown[{i}]"
            if "text" not in m:
                m["text"] = ""
            # Derive field and tour_index if missing
            if "field" not in m:
                # heuristics: extract last component after dot
                p = m.get("path", "")
                last = p.split(".")[-1]
                m["field"] = last
            if "tour_index" not in m:
                match = re.search(r"tours\[(\d+)\]", m.get("path", ""))
                if match:
                    m["tour_index"] = int(match.group(1))
                else:
                    m["tour_index"] = None
            # populate tour_name if available nearby
            if "tour_name" not in m:
                if m.get("field") == "tour_name":
                    m["tour_name"] = m.get("text")
                else:
                    m["tour_name"] = None
        FLAT_TEXTS = [m.get("text", "") for m in MAPPING]
        logger.info("Loaded mapping from %s (%d entries)", path, len(MAPPING))
        return True
    except Exception:
        logger.exception("Failed to load mapping from disk: %s", path)
        return False


def load_knowledge(path: str = KNOWLEDGE_PATH):
    global KNOW, MAPPING, FLAT_TEXTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOW = json.load(f)
    except Exception:
        logger.exception("Could not open knowledge.json; using empty knowledge")
        KNOW = {}

    # If a stable mapping file exists, prefer it
    if os.path.exists(FAISS_MAPPING_PATH):
        ok = load_mapping_from_disk(FAISS_MAPPING_PATH)
        if ok:
            index_tour_names()
            return

    # else, flatten KNOW into MAPPING preserving deterministic order
    MAPPING = []

    def scan(obj, prefix="root"):
        if isinstance(obj, dict):
            # ensure deterministic order by sorted keys
            for k in sorted(obj.keys()):
                scan(obj[k], f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            t = obj.strip()
            if t:
                MAPPING.append({"path": prefix, "text": t})
        else:
            try:
                s = str(obj).strip()
                if s:
                    MAPPING.append({"path": prefix, "text": s})
            except Exception:
                pass

    scan(KNOW)
    # try to derive field/tour_index/tour_name
    for i, m in enumerate(MAPPING):
        if "field" not in m:
            last = m.get("path", "").split(".")[-1]
            m["field"] = last
        if "tour_index" not in m:
            match = re.search(r"tours\[(\d+)\]", m.get("path", ""))
            if match:
                m["tour_index"] = int(match.group(1))
            else:
                m["tour_index"] = None
        if m.get("field") == "tour_name":
            m["tour_name"] = m.get("text")
        else:
            m.setdefault("tour_name", None)

    FLAT_TEXTS = [m.get("text", "") for m in MAPPING]
    index_tour_names()
    logger.info("Knowledge scanned into %d passages", len(MAPPING))

# ---------- Embeddings (with deterministic fallback) ----------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    # try OpenAI
    if client is not None:
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=short)
            if getattr(resp, "data", None) and len(resp.data) > 0:
                emb = resp.data[0].embedding
                return emb, len(emb)
        except Exception:
            logger.exception("OpenAI embedding failed; using fallback")
    # deterministic fallback
    try:
        h = abs(hash(short)) % (10 ** 12)
        dim = 1536 if "3-small" in EMBEDDING_MODEL else 3072
        vec = [((h >> (i % 32)) & 0xFF + (i % 7)) / 255.0 for i in range(dim)]
        return vec, dim
    except Exception:
        return [], 0

# ---------- Simple NumpyIndex (fallback) ----------
class NumpyIndex:
    def __init__(self, mat: Optional[np.ndarray] = None):
        if mat is None or getattr(mat, "size", 0) == 0:
            self.mat = np.empty((0, 0), dtype="float32")
            self.dim = None
        else:
            self.mat = mat.astype("float32")
            self.dim = mat.shape[1]

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

# ---------- Index management ----------

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
    return EMBEDDING_MODEL


def build_index(force_rebuild: bool = False) -> bool:
    global INDEX, FLAT_TEXTS, MAPPING, EMBEDDING_MODEL
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED and HAS_FAISS
        # try loading persisted index first
        if not force_rebuild:
            try:
                if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                    idx = faiss.read_index(FAISS_INDEX_PATH)
                    if load_mapping_from_disk(FAISS_MAPPING_PATH):
                        FLAT_TEXTS = [m.get("text", "") for m in MAPPING]
                        INDEX = idx
                        idx_dim = _index_dim(idx)
                        if idx_dim:
                            EMBEDDING_MODEL = choose_embedding_model_for_dim(idx_dim)
                        index_tour_names()
                        logger.info("FAISS index loaded from disk. n=%s dim=%s", getattr(idx, 'ntotal', 'unknown'), idx_dim)
                        return True
                if os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                    idx = NumpyIndex.load(FALLBACK_VECTORS_PATH)
                    if load_mapping_from_disk(FAISS_MAPPING_PATH):
                        FLAT_TEXTS = [m.get("text", "") for m in MAPPING]
                        INDEX = idx
                        idx_dim = getattr(idx, 'dim', None)
                        if idx_dim:
                            EMBEDDING_MODEL = choose_embedding_model_for_dim(int(idx_dim))
                        index_tour_names()
                        logger.info("Fallback index loaded from disk. n=%s dim=%s", idx.ntotal, idx_dim)
                        return True
            except Exception:
                logger.exception("Error loading persisted index; will rebuild")

        if not FLAT_TEXTS:
            logger.warning("No flattened texts to index (build aborted)")
            INDEX = None
            return False

        # build embeddings
        logger.info("Building embeddings for %d passages (model=%s)...", len(FLAT_TEXTS), EMBEDDING_MODEL)
        vectors = []
        dims = None
        for t in FLAT_TEXTS:
            emb, d = embed_text(t)
            if not emb:
                continue
            if dims is None:
                dims = d
            vectors.append(np.array(emb, dtype="float32"))

        if not vectors or dims is None:
            logger.warning("No vectors produced; index build aborted")
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
                    logger.exception("Failed to persist FAISS index/mapping")
                index_tour_names()
                logger.info("FAISS index built dims=%d n=%d", dims, index.ntotal)
                return True
            else:
                idx = NumpyIndex(mat)
                INDEX = idx
                try:
                    idx.save(FALLBACK_VECTORS_PATH)
                    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                        json.dump(MAPPING, f, ensure_ascii=False, indent=2)
                except Exception:
                    logger.exception("Failed to persist fallback vectors/mapping")
                index_tour_names()
                logger.info("Numpy fallback index built dims=%d n=%d", dims, idx.ntotal)
                return True
        except Exception:
            logger.exception("Error during index build")
            INDEX = None
            return False

# ---------- Query index (semantic search) ----------

def query_index(query: str, top_k: int = TOP_K) -> List[Tuple[float, dict]]:
    global INDEX
    if not query:
        return []
    if INDEX is None:
        built = build_index(force_rebuild=False)
        if not built or INDEX is None:
            logger.warning("Index not available; semantic search skipped")
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)

    idx_dim = _index_dim(INDEX)
    if idx_dim and vec.shape[1] != idx_dim:
        logger.info("Query dim mismatch: query=%s index=%s; attempting rebuild if possible", vec.shape[1], idx_dim)
        desired_model = choose_embedding_model_for_dim(idx_dim)
        if OPENAI_API_KEY:
            global EMBEDDING_MODEL
            EMBEDDING_MODEL = desired_model
            if not build_index(force_rebuild=True):
                logger.error("Rebuild failed; cannot perform search")
                return []
            emb2, d2 = embed_text(query)
            if not emb2:
                return []
            vec = np.array(emb2, dtype="float32").reshape(1, -1)
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        else:
            logger.error("No OPENAI_API_KEY; cannot rebuild model-matched index")
            return []

    try:
        D, I = INDEX.search(vec, top_k)
    except Exception:
        logger.exception("Error executing index.search")
        return []

    results: List[Tuple[float, dict]] = []
    try:
        scores = D[0].tolist() if getattr(D, "shape", None) else []
        idxs = I[0].tolist() if getattr(I, "shape", None) else []
        for score, idx in zip(scores, idxs):
            if idx < 0 or idx >= len(MAPPING):
                continue
            results.append((float(score), MAPPING[idx]))
    except Exception:
        logger.exception("Failed to parse search results")
    return results

# ---------- Deterministic retrieval helpers ----------

def get_passages_by_field(field_name: str, limit: int = 50, tour_indices: Optional[List[int]] = None) -> List[Tuple[float, dict]]:
    exact_matches: List[Tuple[float, dict]] = []
    global_matches: List[Tuple[float, dict]] = []

    for m in MAPPING:
        path = m.get("path", "")
        field = m.get("field") or ""
        if field == field_name or path.endswith(f".{field_name}") or f".{field_name}]" in path:
            is_exact = False
            if tour_indices:
                for ti in tour_indices:
                    if m.get("tour_index") == ti or f"tours[{ti}]" in path:
                        is_exact = True
                        break
            if is_exact:
                exact_matches.append((2.0, m))
            else:
                global_matches.append((1.0, m))

    all_results = exact_matches + global_matches
    all_results.sort(key=lambda x: x[0], reverse=True)
    return all_results[:limit]


def rerank_by_lexical(top_passages: List[Tuple[float, dict]], query: str, required_field: Optional[str] = None, tour_indices: Optional[List[int]] = None) -> List[Tuple[float, dict]]:
    # Combine semantic score with lexical match and tour/field boosts
    q_norm = normalize_text_simple(query)
    q_tokens = token_set(q_norm)
    scored = []
    for sem_score, m in top_passages:
        text = m.get("text", "")
        text_norm = normalize_text_simple(text)
        t_tokens = token_set(text_norm)
        token_overlap = len(q_tokens & t_tokens)
        overlap_score = token_overlap / max(1, len(q_tokens))
        field_boost = 0.0
        if required_field:
            if (m.get("field") == required_field) or (m.get("path", "").endswith(f".{required_field}")):
                field_boost = 0.8
        tour_boost = 0.0
        if tour_indices:
            for ti in tour_indices:
                if m.get("tour_index") == ti or f"tours[{ti}]" in m.get("path", ""):
                    tour_boost = 0.6
                    break
        # similarity fallback
        seq = sequence_similarity(text_norm, q_norm)
        final_score = sem_score * 0.6 + overlap_score * 0.25 + seq * 0.15 + field_boost + tour_boost
        scored.append((final_score, (sem_score, m)))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored]

# ---------- Session management ----------

def make_session_id() -> str:
    return str(uuid.uuid4())


def get_session(session_id: Optional[str] = None) -> Tuple[str, dict]:
    # returns (session_id, session_data)
    if not session_id:
        session_id = request.cookies.get("session_id")
    if REDIS_CLIENT is not None:
        key = f"rbw:session:{session_id}" if session_id else None
        if key:
            try:
                raw = REDIS_CLIENT.get(key)
                if raw:
                    data = json.loads(raw)
                    # update last_activity
                    data["last_activity"] = datetime.utcnow().isoformat()
                    REDIS_CLIENT.set(key, json.dumps(data), ex=SESSION_TIMEOUT)
                    return session_id, data
            except Exception:
                logger.exception("Redis session get failed")
    # fallback to memory store
    if not session_id or session_id not in USER_SESSIONS:
        session_id = make_session_id()
        USER_SESSIONS[session_id] = {
            "created_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "last_tour_index": None,
            "last_tour_name": None,
            "last_field": None,
            "conversation_count": 0
        }
    else:
        USER_SESSIONS[session_id]["last_activity"] = datetime.utcnow().isoformat()
    return session_id, USER_SESSIONS[session_id]


def save_session(session_id: str, data: dict):
    if REDIS_CLIENT is not None:
        try:
            REDIS_CLIENT.set(f"rbw:session:{session_id}", json.dumps(data), ex=SESSION_TIMEOUT)
            return
        except Exception:
            logger.exception("Redis session save failed; falling back to memory")
    USER_SESSIONS[session_id] = data

# ---------- Heuristics / auto-correction ----------

def detect_field_from_message(message: str, prefer_previous: Optional[str] = None) -> Optional[str]:
    m = normalize_text_simple(message)
    # direct keyword mapping
    for k, kw_list in KEYWORD_FIELD_MAP.items():
        for kw in kw_list:
            if kw in m:
                canon = CANONICAL_FIELD_MAP.get(k, k)
                return canon
    # some heuristics
    if any(x in m for x in ["bao gồm", "gồm"]):
        return "includes"
    if any(x in m for x in ["ăn", "bữa", "thực đơn"]):
        return "meals"
    if any(x in m for x in ["giá", "chi phí", "bao nhiêu"]):
        return "price"
    # prefer previous field if message is short and ambiguous
    if prefer_previous and len(m.split()) <= 4:
        return prefer_previous
    return None

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
        return jsonify({"error": "reindex not allowed"}), 403
    load_knowledge()
    ok = build_index(force_rebuild=True)
    return jsonify({"ok": ok, "count": len(FLAT_TEXTS)})


@app.route("/chat", methods=["POST"])
def chat():
    session_id, session_data = get_session()
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Bạn chưa nhập câu hỏi."})

    # detect field
    requested_field = detect_field_from_message(user_message, prefer_previous=session_data.get("last_field"))

    # detect tour mentions
    tour_indices = find_tour_indices_from_message(user_message)

    # update session context
    if tour_indices:
        session_data["last_tour_index"] = tour_indices[0]
        # find tour_name
        for m in MAPPING:
            if m.get("tour_index") == tour_indices[0] and m.get("field") == "tour_name":
                session_data["last_tour_name"] = m.get("text")
                break
        else:
            session_data["last_tour_name"] = None
        session_data["conversation_count"] = 1
    else:
        if session_data.get("last_tour_index") is not None:
            tour_indices = [session_data.get("last_tour_index")]
            session_data["conversation_count"] = session_data.get("conversation_count", 0) + 1

    # reset context when too general
    if session_data.get("conversation_count", 0) > 8:
        # decay
        session_data["last_tour_index"] = None
        session_data["last_tour_name"] = None
        session_data["last_field"] = None
        session_data["conversation_count"] = 0

    # If no explicit field and we have previous field context and short question, reuse
    if not requested_field and session_data.get("last_field") and len(user_message.split()) <= 6:
        requested_field = session_data.get("last_field")

    # store last_field if determined
    if requested_field:
        session_data["last_field"] = requested_field

    # Deterministic retrieval: prefer exact tour+field passages
    top_results: List[Tuple[float, dict]] = []
    if requested_field == "tour_name":
        top_results = get_passages_by_field("tour_name", limit=1000, tour_indices=None)
    elif requested_field and tour_indices:
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=tour_indices)
        if not top_results:
            # fallback to global field
            top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
    elif requested_field:
        # field requested but no tour -> give global field passages first, else semantic
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
        if not top_results:
            top_results = query_index(user_message, TOP_K)
    else:
        # no field detected -> semantic search first
        top_results = query_index(user_message, TOP_K)

    # Two-tier reranking
    top_results = rerank_by_lexical(top_results, user_message, required_field=requested_field, tour_indices=tour_indices)

    # Confidence heuristics: determine if deterministic reply is safe
    confidence = 0.0
    if top_results:
        # base on top semantic score or lexical-derived boost
        sem = top_results[0][0] if top_results and isinstance(top_results[0][0], float) else 0.5
        confidence = min(1.0, 0.3 + sem)

    # Compose response
    reply = ""

    # Try LLM answer if available and confidence ambiguous
    system_prompt = compose_system_prompt(top_results, session_data.get("last_tour_name"))
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]

    if client is not None and confidence < 0.95:
        try:
            resp = client.chat.completions.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.15,
                max_tokens=int(data.get("max_tokens", 700)),
                top_p=0.95
            )
            if getattr(resp, "choices", None) and len(resp.choices) > 0:
                reply = resp.choices[0].message.content or ""
        except Exception:
            logger.exception("OpenAI chat failed; falling back to deterministic reply")

    # Deterministic fallback reply building (strict field isolation when tour known)
    if not reply:
        if top_results:
            # If field was tour_name -> list tours
            if requested_field == "tour_name":
                names = [m.get("text", "") for _, m in top_results]
                seen = set(); unique = [x for x in names if x and not (x in seen or seen.add(x))]
                reply = "Các tour hiện có:\n" + "\n".join(f"- {n}" for n in unique)
            elif requested_field and tour_indices:
                parts = []
                for ti in tour_indices:
                    tour_name = None
                    for m in MAPPING:
                        if m.get("tour_index") == ti and m.get("field") == "tour_name":
                            tour_name = m.get("text")
                            break
                    field_passages = [m.get("text", "") for _, m in top_results if m.get("tour_index") == ti]
                    if not field_passages:
                        # explicit fetch
                        field_passages = [m.get("text", "") for _, m in get_passages_by_field(requested_field, limit=TOP_K, tour_indices=[ti])]
                    if field_passages:
                        label = f'Tour "{tour_name}"' if tour_name else f"Tour #{ti}"
                        parts.append(label + ":\n" + "\n".join(f"- {t}" for t in field_passages))
                if parts:
                    reply = "\n\n".join(parts)
                else:
                    snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                    reply = f"Tôi tìm thấy:\n\n{snippets}"
            else:
                snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}"
        else:
            # No matches -> deterministic fallback message
            if session_data.get("last_tour_name"):
                reply = f"Hiện chưa tìm thấy nội dung cho tour '{session_data.get('last_tour_name')}' theo trường yêu cầu. Vui lòng nêu rõ tên tour khác hoặc hỏi trường khác."
            else:
                reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan. Vui lòng nêu rõ tên tour hoặc mô tả chi tiết hơn."

    # Logging trace
    trace = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "session_id": session_id,
        "user_message": user_message,
        "detected_field": requested_field,
        "detected_tours": tour_indices,
        "chosen_top_source": [m.get("path") for _, m in (top_results[:5] if top_results else [])],
        "confidence": confidence
    }
    logger.info("QUERY TRACE: %s", json.dumps(trace, ensure_ascii=False))

    # Save session and set cookie
    save_session(session_id, session_data)
    response = jsonify({
        "reply": reply,
        "sources": [m for _, m in top_results],
        "context_tour": session_data.get("last_tour_name"),
        "session_active": session_data.get("last_tour_name") is not None,
        "trace": trace
    })
    response.set_cookie("session_id", session_id, max_age=SESSION_TIMEOUT, httponly=True)
    return response

# ---------- Prompt composition ----------

def compose_system_prompt(top_passages: List[Tuple[float, dict]], context_tour: Optional[str] = None) -> str:
    header = "Bạn là trợ lý AI của Ruby Wings - chuyên tư vấn du lịch trải nghiệm.\n"
    if context_tour:
        header += f"NGỮ CẢNH: User đang hỏi về tour '{context_tour}'. Ưu tiên trả lời theo tour này.\n\n"
    header += (
        "NGUYÊN TẮC:\n"
        "1) ƯU TIÊN: Trả thông tin trực tiếp từ dữ liệu nội bộ (các nguồn dưới đây).\n"
        "2) Nếu thiếu: trả info chung hoặc nói rõ không đủ dữ liệu.\n"
        "3) Không bịa đặt, chỉ trích xuất thông tin có nguồn.\n\n"
    )
    if not top_passages:
        return header + "Không tìm thấy dữ liệu nội bộ phù hợp."
    content = header + "DỮ LIỆU NỘI BỘ (theo thứ tự liên quan):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"
    content += "\n---\nHãy trả lời ngắn gọn, chính xác, trích dẫn nguồn nếu cần."
    return content

# ---------- Initialization ----------
try:
    load_knowledge()
    # if mapping file exists, already loaded in load_knowledge
    # build index in background
    t = threading.Thread(target=build_index, kwargs={"force_rebuild": False}, daemon=True)
    t.start()
except Exception:
    logger.exception("Initialization error")

if __name__ == "__main__":
    # ensure mapping persisted for reproducibility
    if MAPPING and not os.path.exists(FAISS_MAPPING_PATH):
        try:
            with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                json.dump(MAPPING, f, ensure_ascii=False, indent=2)
        except Exception:
            logger.exception("Failed to persist mapping at startup")
    built = build_index(force_rebuild=False)
    if not built:
        logger.warning("Index not ready at startup; endpoint will attempt on-demand build.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
