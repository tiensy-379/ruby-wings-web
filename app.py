# app.py — Ultra-Pro Max
# - Compatible with openai==0.28.0
# - Two-stage retrieval: semantic search (faiss/numpy) -> vector rerank (exact cosine on embeddings)
# - Semantic intent classifier (keywords + embedding prototype fallback)
# - Fuzzy NER (tour name detection) with normalized Levenshtein and token overlap + auto-correction
# - Query preprocessing, structured logging, safe fallbacks
# - Keeps endpoints: /, /reindex, /chat, /stream (stream unchanged from prior design)
import os
import json
import threading
import logging
import re
import unicodedata
import time
from functools import lru_cache
from typing import List, Tuple, Dict, Optional

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import numpy as np
import openai

# Optional FAISS
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ---------------- logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw_ultra")

# ---------------- config ----------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
openai.api_key = OPENAI_API_KEY

KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true")

# Ultra-Pro params
INITIAL_RETRIEVAL_K = max(TOP_K, 20)       # retrieve 20 candidates first-stage
RERANK_TOP_K = TOP_K                       # return top_k after rerank
INTENT_EMB_WEIGHT = 0.6                    # weight for embedding similarity in intent classification
FUZZY_LEV_THRESHOLD = 0.35                 # normalized edit distance threshold for auto-correct (lower better)
TOKEN_OVERLAP_THRESHOLD = 0.2              # min token overlap ratio to consider match
FIELD_PROTOTYPE_TEXTS = {                  # friendly labels to create prototypes (used if no index prototypes)
    "tour_name": "tour list tour_name",
    "mission": "mission vision core values company mission",
    "summary": "summary overview brief tour summary",
    "style": "style vibe concept tour",
    "transport": "transportation transfer vehicle bus car",
    "includes": "includes itinerary schedule program details",
    "location": "location destination where to go",
    "price": "price cost fee package",
    "notes": "notes important things to know",
    "accommodation": "accommodation hotel homestay lodging",
    "meals": "meals food dining menu",
    "event_support": "event support additional services",
    "cancellation_policy": "cancellation policy refund rules",
    "booking_method": "booking how to book reserve",
    "who_can_join": "who can join eligible participants",
    "hotline": "hotline contact phone number"
}

app = Flask(__name__)
CORS(app)

# ---------------- global state ----------------
KNOW: Dict = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []  # list of {"path":..., "text":...}
INDEX = None
INDEX_LOCK = threading.Lock()

# ---- keyword field map (kept and extended) ----
KEYWORD_FIELD_MAP = {
  "tour_list": {
    "keywords": [
        "tên tour", "tour gì", "danh sách tour", "có những tour nào", "liệt kê tour",
        "show tour", "tour hiện có", "tour available", "liệt kê các tour đang có",
        "list tour", "tour đang bán", "tour hiện hành", "tour nào", "tours", "list tours"
    ],
    "field": "tour_name"
  },
  "mission": {
    "keywords": ["tầm nhìn", "sứ mệnh", "giá trị cốt lõi", "triết lý", "vision", "mission"],
    "field": "mission"
  },
  "summary": {
    "keywords": ["tóm tắt chương trình tour", "tóm tắt", "overview", "brief", "mô tả ngắn"],
    "field": "summary"
  },
  "style": {
    "keywords": ["phong cách hành trình", "tính chất hành trình", "style", "vibe tour"],
    "field": "style"
  },
  "transport": {
    "keywords": ["vận chuyển", "phương tiện", "di chuyển", "xe gì", "transportation"],
    "field": "transport"
  },
  "includes": {
    "keywords": ["lịch trình chi tiết", "chương trình chi tiết", "itinerary", "lịch trình", "includes"],
    "field": "includes"
  },
  "location": {
    "keywords": ["ở đâu", "đi đâu", "địa điểm", "location", "destination"],
    "field": "location"
  },
  "price": {
    "keywords": ["giá tour", "chi phí", "bao nhiêu tiền", "price", "cost"],
    "field": "price"
  },
  "notes": {
    "keywords": ["lưu ý", "ghi chú", "notes", "cần biết"],
    "field": "notes"
  },
  "accommodation": {
    "keywords": ["chỗ ở", "lưu trú", "khách sạn", "homestay", "accommodation"],
    "field": "accommodation"
  },
  "meals": {
    "keywords": ["ăn uống", "ẩm thực", "meals", "thực đơn", "bữa"],
    "field": "meals"
  },
  "event_support": {
    "keywords": ["hỗ trợ", "dịch vụ tăng cường", "event support", "dịch vụ bổ sung"],
    "field": "event_support"
  },
  "cancellation_policy": {
    "keywords": ["phí huỷ", "chính sách huỷ", "cancellation", "refund"],
    "field": "cancellation_policy"
  },
  "booking_method": {
    "keywords": ["đặt chỗ", "đặt tour", "booking", "cách đặt"],
    "field": "booking_method"
  },
  "who_can_join": {
    "keywords": ["phù hợp đối tượng", "ai tham gia", "who should join"],
    "field": "who_can_join"
  },
  "hotline": {
    "keywords": ["hotline", "số điện thoại", "liên hệ", "contact number"],
    "field": "hotline"
  }
}

# ---------------- utilities ----------------
def normalize_text_simple(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")  # remove diacritics
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def tokenize(s: str) -> List[str]:
    return [t for t in normalize_text_simple(s).split() if t]

# fast normalized Levenshtein (iterative) — returns normalized distance 0..1
def normalized_levenshtein(a: str, b: str) -> float:
    a = a or ""
    b = b or ""
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0:
        return 1.0
    if lb == 0:
        return 1.0
    # DP
    v0 = list(range(lb + 1))
    v1 = [0] * (lb + 1)
    for i in range(la):
        v1[0] = i + 1
        ai = a[i]
        for j in range(lb):
            cost = 0 if ai == b[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0
    dist = v0[lb]
    norm = dist / max(la, lb)
    return float(norm)

# Fuzzy tour detection + auto-correction
TOUR_NAME_TO_INDEX: Dict[str, int] = {}  # normalized -> index
TOUR_RAW_NAMES: Dict[int, str] = {}     # index -> raw name

def index_tour_names():
    global TOUR_NAME_TO_INDEX, TOUR_RAW_NAMES
    TOUR_NAME_TO_INDEX = {}
    TOUR_RAW_NAMES = {}
    for m in MAPPING:
        p = m.get("path", "")
        if p.endswith(".tour_name"):
            text = m.get("text", "") or ""
            n = normalize_text_simple(text)
            if not n:
                continue
            # extract index from path "root.tours[3].tour_name"
            mobj = re.search(r"tours\[(\d+)\]", p)
            if mobj:
                idx = int(mobj.group(1))
                TOUR_NAME_TO_INDEX[n] = idx
                TOUR_RAW_NAMES[idx] = text

def find_tour_indices_from_message(message: str) -> List[int]:
    """
    Aggressive fuzzy NER:
    - exact normalized substring
    - normalized tour name contained in message or vice versa
    - token overlap scoring
    - normalized edit distance threshold for autosuggest/correction
    Returns list of best matching tour indices (may be empty)
    """
    msg_n = normalize_text_simple(message)
    if not msg_n:
        return []
    # exact/substring matches (prefer longest)
    candidates = []
    for tnorm, idx in TOUR_NAME_TO_INDEX.items():
        if tnorm in msg_n or msg_n in tnorm:
            candidates.append((len(tnorm), idx, tnorm))
    if candidates:
        # pick longest matches first
        candidates.sort(reverse=True)
        # may return multiple if same length
        best_len = candidates[0][0]
        return [c[1] for c in candidates if c[0] == best_len]
    # token-overlap fuzzy
    msg_tokens = set(msg_n.split())
    overlap_scores = []
    for tnorm, idx in TOUR_NAME_TO_INDEX.items():
        toks = set(tnorm.split())
        if not toks:
            continue
        overlap = len(msg_tokens & toks) / len(toks)
        if overlap >= TOKEN_OVERLAP_THRESHOLD:
            overlap_scores.append((overlap, idx, tnorm))
    if overlap_scores:
        overlap_scores.sort(reverse=True)
        best = overlap_scores[0][0]
        return [s[1] for s in overlap_scores if s[0] == best]
    # normalized Levenshtein for near-miss: choose minimal normalized distance
    lev_scores = []
    for tnorm, idx in TOUR_NAME_TO_INDEX.items():
        d = normalized_levenshtein(msg_n, tnorm)
        lev_scores.append((d, idx, tnorm))
    lev_scores.sort(key=lambda x: x[0])
    if lev_scores and lev_scores[0][0] <= FUZZY_LEV_THRESHOLD:
        best = lev_scores[0][0]
        return [s[1] for s in lev_scores if abs(s[0] - best) < 1e-9][:1]
    return []

# ---------------- Index fallback (numpy) ----------------
class NumpyIndex:
    def __init__(self, mat=None):
        self.mat = mat.astype("float32") if (isinstance(mat, np.ndarray) and getattr(mat, "size", 0) > 0) else np.empty((0,0), dtype="float32")
        self.dim = None if self.mat.size == 0 else self.mat.shape[1]

    def search(self, qvec, k):
        if self.mat is None or self.mat.size == 0:
            return np.array([[]], dtype="float32"), np.array([[]], dtype="int64")
        q = qvec / (np.linalg.norm(qvec, axis=1, keepdims=True) + 1e-12)
        m = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        sims = np.dot(q, m.T)
        idx = np.argsort(-sims, axis=1)[:, :k]
        scores = np.take_along_axis(sims, idx, axis=1)
        return scores.astype("float32"), idx.astype("int64")

    @classmethod
    def load(cls, path):
        try:
            arr = np.load(path)
            return cls(arr["mat"])
        except Exception:
            return cls(None)

    def save(self, path):
        try:
            np.savez_compressed(path, mat=self.mat)
        except Exception:
            logger.exception("Failed to save fallback vectors")

# ---------------- Embedding helpers ----------------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    # try modern call
    try:
        resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=short)
        emb = None
        if isinstance(resp, dict) and "data" in resp and len(resp["data"]) > 0:
            emb = resp["data"][0].get("embedding") or resp["data"][0].get("vector")
        if emb:
            return emb, len(emb)
    except Exception:
        logger.debug("OpenAI embedding failed or not available (will fallback).", exc_info=False)
    # deterministic fallback
    try:
        h = abs(hash(short)) % (10 ** 12)
        fallback_dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(fallback_dim)]
        return vec, fallback_dim
    except Exception:
        return [], 0

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

# ---------------- Build / Load index ----------------
def load_knowledge(path=KNOWLEDGE_PATH):
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
            except:
                pass
    scan(KNOW)
    index_tour_names()
    logger.info("Knowledge loaded: %d passages", len(FLAT_TEXTS))

def build_index(force_rebuild=False):
    """
    Build or load index (faiss if available, else numpy fallback).
    """
    global INDEX, MAPPING, FLAT_TEXTS, EMBEDDING_MODEL
    with threading.Lock():
        use_faiss = FAISS_ENABLED and HAS_FAISS
        if not force_rebuild:
            # try loading existing FAISS
            if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = faiss.read_index(FAISS_INDEX_PATH)
                    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        MAPPING[:] = json.load(f)
                    FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                    idx_dim = _index_dim(idx)
                    if idx_dim:
                        EMBEDDING_MODEL = choose_embedding_model_for_dim(idx_dim)
                    INDEX = idx
                    index_tour_names()
                    logger.info("Loaded FAISS index from disk.")
                    return True
                except Exception:
                    logger.exception("Failed to load FAISS index; will rebuild.")
            # try loading fallback vectors
            if os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    arr = np.load(FALLBACK_VECTORS_PATH)
                    mat = arr["mat"]
                    idx = NumpyIndex(mat)
                    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        MAPPING[:] = json.load(f)
                    FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                    INDEX = idx
                    index_tour_names()
                    logger.info("Loaded fallback vectors.")
                    return True
                except Exception:
                    logger.exception("Failed to load fallback vectors; will rebuild.")
        # build fresh index
        if not FLAT_TEXTS:
            logger.warning("No flattened texts to index (build aborted).")
            INDEX = None
            return False
        logger.info("Building embeddings/index for %d passages (model=%s)...", len(FLAT_TEXTS), EMBEDDING_MODEL)
        vecs = []
        dims = None
        for text in FLAT_TEXTS:
            emb, d = embed_text(text)
            if not emb:
                continue
            if dims is None:
                dims = d
            vecs.append(np.array(emb, dtype="float32"))
        if not vecs or dims is None:
            logger.warning("No vectors produced; abort build.")
            INDEX = None
            return False
        try:
            mat = np.vstack(vecs).astype("float32")
            mat = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)
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
                logger.info("FAISS index built (dims=%d, n=%d).", dims, index.ntotal)
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
                logger.info("Numpy index built (dims=%d, n=%d).", dims, idx.mat.shape[0])
            return True
        except Exception:
            logger.exception("Error while building index")
            INDEX = None
            return False

# ---------------- Retrieval: 2-stage vector reranking ----------------
def semantic_initial_retrieval(query: str, k: int = INITIAL_RETRIEVAL_K):
    """
    Use INDEX to get top-k candidate ids and scores (first stage).
    Returns list of (score, mapping_entry, idx).
    """
    if INDEX is None:
        ok = build_index(force_rebuild=False)
        if not ok or INDEX is None:
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    try:
        D, I = INDEX.search(vec, k)
    except Exception:
        logger.exception("Initial retrieval search error")
        return []
    results = []
    arr_scores = D[0].tolist() if getattr(D, "shape", None) else []
    arr_idx = I[0].tolist() if getattr(I, "shape", None) else []
    for sc, idx in zip(arr_scores, arr_idx):
        if idx < 0 or idx >= len(MAPPING):
            continue
        results.append((float(sc), MAPPING[idx], int(idx)))
    return results

def rerank_candidates_by_embedding(query: str, candidates: List[Tuple[float, dict, int]], top_k: int = RERANK_TOP_K):
    """
    Recompute embeddings for the query and each candidate passage, compute cosine similarity,
    and return top_k sorted by similarity. This is the second-tier rerank (exact).
    """
    if not candidates:
        return []
    # compute query embedding
    q_emb, _ = embed_text(query)
    if not q_emb:
        return []
    qv = np.array(q_emb, dtype="float32")
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    reranked = []
    for _, m, idx in candidates:
        txt = m.get("text", "") or ""
        emb, _ = embed_text(txt)
        if not emb:
            score = 0.0
        else:
            pv = np.array(emb, dtype="float32")
            pv = pv / (np.linalg.norm(pv) + 1e-12)
            score = float(np.dot(qv, pv))
        reranked.append((score, m, idx))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return reranked[:top_k]

# ---------------- Semantic intent classifier ----------------
# build prototypes by embedding the FIELD_PROTOTYPE_TEXTS or using available field examples (cheap)
FIELD_PROTOTYPES: Dict[str, List[float]] = {}

def build_field_prototypes():
    """
    Create prototype embeddings for each target field using FIELD_PROTOTYPE_TEXTS.
    This is light-weight and avoids embedding all mapping entries again.
    """
    global FIELD_PROTOTYPES
    FIELD_PROTOTYPES = {}
    for field, phrase in FIELD_PROTOTYPE_TEXTS.items():
        emb, d = embed_text(phrase)
        if emb:
            FIELD_PROTOTYPES[field] = emb

def classify_intent(query: str) -> Optional[str]:
    """
    Hybrid classifier:
    1) keyword match strong (exact substrings) -> return field
    2) embedding similarity to field prototypes -> top score if above threshold
    3) else None
    """
    qn = normalize_text_simple(query)
    # 1) keyword-based
    for k, v in KEYWORD_FIELD_MAP.items():
        for kw in v["keywords"]:
            if kw in qn:
                return v["field"]
    # 2) embedding-based
    q_emb, _ = embed_text(query)
    if not q_emb or not FIELD_PROTOTYPES:
        return None
    qv = np.array(q_emb, dtype="float32")
    qv = qv / (np.linalg.norm(qv) + 1e-12)
    best_field = None
    best_score = -1.0
    for field, proto in FIELD_PROTOTYPES.items():
        pv = np.array(proto, dtype="float32")
        pv = pv / (np.linalg.norm(pv) + 1e-12)
        score = float(np.dot(qv, pv))
        if score > best_score:
            best_score = score
            best_field = field
    # require minimum confidence
    if best_score >= 0.6:
        return best_field
    return None

# ---------------- Helpers: get passages by field optionally restricted by tour indices ----------------
def get_passages_by_field(field_name: str, limit: int = 50, tour_indices: Optional[List[int]] = None):
    res = []
    for m in MAPPING:
        path = m.get("path", "")
        if path.endswith(f".{field_name}") or f".{field_name}" in path:
            if tour_indices:
                matched = False
                for ti in tour_indices:
                    if f"[{ti}]" in path:
                        matched = True
                        break
                if not matched:
                    continue
            res.append((1.0, m))
    return res[:limit]

# ---------------- Prompt composition ----------------
def compose_system_prompt(top_passages: List[Tuple[float, dict]]) -> str:
    header = (
        "Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn ngành du lịch trải nghiệm, retreat, thiền, khí công, hành trình chữa lành.\n"
        "Trả lời ngắn gọn, chính xác, tử tế.\n\n"
    )
    if not top_passages:
        return header + "Không tìm thấy dữ liệu nội bộ phù hợp."
    content = header + "Dữ liệu nội bộ (theo độ liên quan):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"
    content += "\n---\nƯu tiên trích dẫn dữ liệu nội bộ; không bịa; văn phong lịch sự."
    return content

# ---------------- Endpoints ----------------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "knowledge_count": len(FLAT_TEXTS),
        "index_exists": INDEX is not None,
        "index_dim": _index_dim(INDEX),
        "embedding_model": EMBEDDING_MODEL,
        "faiss_available": HAS_FAISS,
        "build_time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    })

@app.route("/reindex", methods=["POST"])
def reindex():
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
        return jsonify({"error": "reindex not allowed"}), 403
    load_knowledge()
    ok = build_index(force_rebuild=True)
    # rebuild prototypes after index built
    build_field_prototypes()
    return jsonify({"ok": ok, "count": len(FLAT_TEXTS)})

@app.route("/chat", methods=["POST"])
def chat():
    """
    Behavior:
    - Preprocess query (normalize)
    - Detect tour mentions (fuzzy NER)
    - Classify intent (keyword -> field)
    - If field detected: attempt per-tour field fetch (if tour mentioned), else global field fetch
    - If not detected: two-stage semantic retrieval + rerank
    - Compose system prompt and call OpenAI; deterministic fallback returns prioritized field texts
    """
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"reply": "Bạn chưa nhập câu hỏi."})
    logger.info("QUERY: %s", user_message)

    # preprocess & detect tour
    user_norm = normalize_text_simple(user_message)
    tour_indices = find_tour_indices_from_message(user_message)

    # classify intent
    field_by_keyword = classify_intent(user_message)

    top_results: List[Tuple[float, dict]] = []

    # If user explicitly asked for tour list
    if field_by_keyword == "tour_name":
        all_tours = get_passages_by_field("tour_name", limit=1000, tour_indices=None)
        top_results = all_tours
        override_field = "tour_name"
    elif field_by_keyword:
        # If tour mentioned, prefer field from that tour
        override_field = field_by_keyword
        if tour_indices:
            top_results = get_passages_by_field(field_by_keyword, limit=TOP_K, tour_indices=tour_indices)
            if not top_results:
                top_results = get_passages_by_field(field_by_keyword, limit=TOP_K, tour_indices=None)
        else:
            # no explicit tour mention — use global field passages
            top_results = get_passages_by_field(field_by_keyword, limit=TOP_K, tour_indices=None)
            if not top_results:
                # fallback to semantic retrieval
                candidates = semantic_initial_retrieval(user_message, k=INITIAL_RETRIEVAL_K)
                reranked = rerank_candidates_by_embedding(user_message, candidates, top_k=RERANK_TOP_K)
                top_results = [(s, m) for s, m, idx in reranked]
    else:
        override_field = None
        # two-stage semantic retrieval
        candidates = semantic_initial_retrieval(user_message, k=INITIAL_RETRIEVAL_K)
        # if a tour is mentioned, filter candidates to that tour first (in-candidate)
        if tour_indices and candidates:
            candidates = [c for c in candidates if any(f"[{ti}]" in c[1].get("path", "") for ti in tour_indices)]
            if not candidates:
                # if none matched, keep original candidates
                candidates = semantic_initial_retrieval(user_message, k=INITIAL_RETRIEVAL_K)
        reranked = rerank_candidates_by_embedding(user_message, candidates, top_k=RERANK_TOP_K)
        top_results = [(s, m) for s, m, idx in reranked]

    # Compose prompt (use top_results) and call OpenAI ChatCompletion
    system_prompt = compose_system_prompt(top_results)
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_message}]
    reply = ""
    if OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages, temperature=0.2,
                                                max_tokens=int(data.get("max_tokens", 700)))
            # robust parse
            if isinstance(resp, dict):
                choices = resp.get("choices", [])
                if choices:
                    first = choices[0]
                    if isinstance(first.get("message"), dict):
                        reply = first["message"].get("content", "") or ""
                    elif "text" in first:
                        reply = first.get("text", "")
                    else:
                        reply = str(first)
                else:
                    reply = str(resp)
            else:
                choices = getattr(resp, "choices", None)
                if choices and len(choices) > 0:
                    first = choices[0]
                    msg = getattr(first, "message", None)
                    if msg and isinstance(msg, dict):
                        reply = msg.get("content", "")
                    else:
                        reply = str(first)
                else:
                    reply = str(resp)
        except Exception:
            logger.exception("OpenAI chat failed; fallback to deterministic reply")

    # Deterministic fallback (ensures field-first answers, and per-tour isolation)
    if not reply:
        if top_results:
            # If requested_field == tour_name -> list
            if override_field == "tour_name" and all(("tour_name" in (m.get("path", "")) or ".tour_name" in m.get("path", "")) for _, m in top_results):
                names = [m.get("text", "") for _, m in top_results]
                seen = set()
                uniq = [x for x in names if not (x in seen or seen.add(x))]
                reply = "Các tour hiện có:\n" + "\n".join(f"- {n}" for n in uniq)
            else:
                # If we have tour_indices and override_field, show that field per tour
                if tour_indices and override_field:
                    parts = []
                    for ti in tour_indices:
                        # find tour name
                        tour_name = TOUR_RAW_NAMES.get(ti, None)
                        field_texts = [m.get("text", "") for _, m in top_results if f"[{ti}]" in m.get("path", "")]
                        if not field_texts:
                            # try fetching explicitly
                            field_texts = [mm.get("text", "") for _, mm in get_passages_by_field(override_field, tour_indices=[ti], limit=TOP_K)]
                        if field_texts:
                            label = f'Tour "{tour_name}"' if tour_name else f"Tour #{ti}"
                            parts.append(label + ":\n" + "\n".join(f"- {t}" for t in field_texts))
                    if parts:
                        reply = "\n\n".join(parts)
                    else:
                        # fallback snippet
                        snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                        reply = f"Tôi tìm thấy:\n\n{snippets}"
                else:
                    snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top_results[:5]])
                    reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}"
        else:
            reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan."

    # structured log
    logger.info("REPLY generated (len=%d) | intent=%s | tours=%s | results=%d",
                len(reply),
                override_field,
                ",".join(str(x) for x in tour_indices) if tour_indices else "",
                len(top_results))

    return jsonify({"reply": reply, "sources": [m for _, m in top_results]})

# Minimal streaming SSE (kept simple)
@app.route("/stream", methods=["POST"])
def stream():
    data = request.get_json() or {}
    user_message = (data.get("message") or "").strip()
    if not user_message:
        return jsonify({"error": "empty message"}), 400

    # For streaming we reuse chat() logic but call OpenAI streaming if available.
    # To keep compatibility and safety, we'll do non-stream deterministic response here
    resp = chat()
    return resp

# ---------------- Initialization ----------------
# load knowledge and build index & prototypes at import time for speed
load_knowledge()
build_index(force_rebuild=False)
build_field_prototypes()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
