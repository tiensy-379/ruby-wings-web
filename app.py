#!/usr/bin/env python3
"""
app.py - Improved Ruby Wings AI chat server (Flask)

Goals / behavior (concise):
- Prioritize returning the correct "field" for a tour when user asks (price, transport, itinerary, etc.)
- If a tour name is mentioned, restrict answers to that tour first.
- Keep friendly "healing travel" tone when using LLM, but strictly cite sources and avoid hallucination.
- Keep session context (last_tour_index, recent_entities) so follow-ups like "còn chỗ không?" resolve correctly.
- Use helpers_bundle shims if present (intent detection, coref, clarifier, session_store, metrics, synonyms).
- FAISS fallback to numpy index; will attempt to load persisted indices first.
"""

import os
import json
import threading
import logging
import re
import unicodedata
from functools import lru_cache
from typing import List, Tuple, Dict, Optional, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import subprocess
import sys

# Try FAISS; fallback to numpy-only index if missing
HAS_FAISS = False
try:
    import faiss  # type: ignore
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# OpenAI new SDK compatibility
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw")

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Helpers bundle (optional shims) ----------
try:
    import intent_classifier
    import coref
    import clarifier
    import session_store
    import metrics
    import helpers_bundle
except Exception:
    intent_classifier = None
    coref = None
    clarifier = None
    session_store = None
    metrics = None
    helpers_bundle = None

# If helpers_bundle provides synonyms loader, use it
if helpers_bundle and hasattr(helpers_bundle, "load_synonyms"):
    SYNONYMS = helpers_bundle.load_synonyms()
else:
    SYNONYMS = {}

# ---------- Keyword -> field mapping (priority) ----------
KEYWORD_FIELD_MAP: Dict[str, Dict] = {
    "tour_list": {"keywords": ["tên tour","tour gì","danh sách tour","liệt kê tour","có những tour nào","liệt kê các tour"], "field":"tour_name"},
    "price": {"keywords": ["giá tour","chi phí","bao nhiêu tiền","price","cost"], "field":"price"},
    "transport": {"keywords": ["vận chuyển","phương tiện","xe gì","di chuyển","transport"], "field":"transport"},
    "includes": {"keywords": ["chương trình chi tiết","chi tiết hành trình","itinerary","includes"], "field":"includes"},
    "location": {"keywords": ["ở đâu","đi đâu","địa danh","điểm đến","destination","location"], "field":"location"},
    "duration": {"keywords": ["mấy ngày","ngày","đêm","kéo dài","duration"], "field":"duration"},
    "notes": {"keywords": ["lưu ý","ghi chú","notes"], "field":"notes"},
    "accommodation": {"keywords": ["chỗ ở","khách sạn","homestay","accommodation"], "field":"accommodation"},
    "meals": {"keywords": ["ăn uống","bữa","thực đơn","meals"], "field":"meals"},
    "booking_method": {"keywords": ["đặt chỗ","đặt tour","booking","cách đặt"], "field":"booking_method"},
    "hotline": {"keywords": ["hotline","số điện thoại","liên hệ","contact"], "field":"hotline"},
}

# ---------- Global state ----------
KNOW: Dict = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []  # list of {"path": "...", "text": "..."}
INDEX: Optional[Any] = None  # faiss index or NumpyIndex
INDEX_LOCK = threading.Lock()

TOUR_NAME_TO_INDEX: Dict[str, int] = {}  # normalized tour name -> index

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

def load_knowledge(path: str = KNOWLEDGE_PATH):
    """Flatten knowledge.json into MAPPING and FLAT_TEXTS."""
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

# ---------- Tour name indexing & detection ----------
def index_tour_names():
    """Populate TOUR_NAME_TO_INDEX from MAPPING entries that end with .tour_name."""
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
                # prefer first seen or longer name heuristic
                prev = TOUR_NAME_TO_INDEX.get(norm)
                if prev is None:
                    TOUR_NAME_TO_INDEX[norm] = idx

def find_tour_indices_from_message(message: str) -> List[int]:
    """
    Detect tour indices in user's message.
    Steps: exact normalized match, synonyms match, token overlap fuzzy match.
    """
    if not message:
        return []
    msg_n = normalize_text_simple(message)
    if not msg_n:
        return []

    # 1) exact name match
    if msg_n in TOUR_NAME_TO_INDEX:
        return [TOUR_NAME_TO_INDEX[msg_n]]

    # 2) synonyms map (keys normalized)
    for canon, syns in SYNONYMS.items():
        if canon in msg_n:
            # pick first matching canonical if exists in TOUR_NAME_TO_INDEX
            if canon in TOUR_NAME_TO_INDEX:
                return [TOUR_NAME_TO_INDEX[canon]]
        for s in syns:
            if s in msg_n:
                if canon in TOUR_NAME_TO_INDEX:
                    return [TOUR_NAME_TO_INDEX[canon]]

    # 3) token overlap fuzzy
    tokens_msg = set(msg_n.split())
    scores = []
    for norm_name, idx in TOUR_NAME_TO_INDEX.items():
        tset = set(norm_name.split())
        if not tset:
            continue
        common = tokens_msg & tset
        if common:
            scores.append((len(common), len(tset), idx))
    if not scores:
        return []
    # prefer higher common count, tie-break by smaller tour token size (more specific)
    scores.sort(key=lambda x: (-x[0], x[1]))
    best = scores[0][0]
    selected = [s[2] for s in scores if s[0] == best]
    return sorted(set(selected))

# ---------- Field passages retrieval ----------
def get_passages_by_field(field_name: str, limit: int = 50, tour_indices: Optional[List[int]] = None) -> List[Tuple[float, dict]]:
    """
    Return passages whose path ends with field_name.
    If tour_indices provided, PRIORITIZE entries matching those tour index brackets.
    Score 2.0 for exact tour match, 1.0 for global match.
    """
    exact_matches = []
    global_matches = []
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

# ---------- Embeddings & Index access ----------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    """
    Return (embedding list, dim). Prefer OpenAI client if configured; fallback deterministic.
    """
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    if OPENAI_API_KEY and OpenAI is not None:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=short)
            if getattr(resp, "data", None):
                emb = resp.data[0].embedding
                return emb, len(emb)
        except Exception:
            logger.exception("OpenAI embedding failed; falling back.")
    # deterministic fallback (1536-dim)
    try:
        h = abs(hash(short)) % (10 ** 12)
        dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]
        return vec, dim
    except Exception:
        return [], 0

class NumpyIndex:
    """Simple in-memory numpy index with cosine-similarity (normalized dot)."""
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
    return EMBEDDING_MODEL

# ---------- Index load/build helpers ----------
def load_mapping_from_disk(path=FAISS_MAPPING_PATH):
    global MAPPING, FLAT_TEXTS
    try:
        with open(path, "r", encoding="utf-8") as f:
            file_map = json.load(f)
        if file_map:
            MAPPING[:] = file_map
            FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
            index_tour_names()
            logger.info("Loaded mapping from %s (%d entries)", path, len(MAPPING))
            return True
    except Exception:
        logger.exception("Failed to load mapping from disk")
    return False

def build_index(force_rebuild: bool = False) -> bool:
    """
    Try load persisted index; otherwise build from FLAT_TEXTS using embed_text.
    Prefer FAISS if enabled and available, else NumpyIndex fallback.
    """
    global INDEX, MAPPING, FLAT_TEXTS, EMBEDDING_MODEL
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED and HAS_FAISS

        # Try loading
        if not force_rebuild:
            if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = faiss.read_index(FAISS_INDEX_PATH)
                    if load_mapping_from_disk(FAISS_MAPPING_PATH):
                        INDEX = idx
                        idx_dim = _index_dim(idx)
                        if idx_dim:
                            EMBEDDING_MODEL = choose_embedding_model_for_dim(idx_dim)
                        index_tour_names()
                        logger.info("FAISS index loaded from disk.")
                        return True
                except Exception:
                    logger.exception("Failed to load FAISS index; will rebuild.")
            if os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = NumpyIndex.load(FALLBACK_VECTORS_PATH)
                    if load_mapping_from_disk(FAISS_MAPPING_PATH):
                        INDEX = idx
                        idx_dim = getattr(idx, "dim", None)
                        if idx_dim:
                            EMBEDDING_MODEL = choose_embedding_model_for_dim(int(idx_dim))
                        index_tour_names()
                        logger.info("Fallback vectors loaded from disk.")
                        return True
                except Exception:
                    logger.exception("Failed to load fallback vectors; will rebuild.")

        if not FLAT_TEXTS:
            logger.warning("No flattened texts to index; build aborted.")
            INDEX = None
            return False

        logger.info("Building embeddings for %d passages (model=%s)...", len(FLAT_TEXTS), EMBEDDING_MODEL)
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
            logger.warning("No vectors produced; index build aborted.")
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
                logger.info("FAISS index built (dims=%d, n=%d).", dims, index.ntotal)
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
                logger.info("Numpy fallback index built (dims=%d, n=%d).", dims, idx.ntotal)
                return True
        except Exception:
            logger.exception("Error while building index")
            INDEX = None
            return False

def query_index(query: str, top_k: int = TOP_K) -> List[Tuple[float, dict]]:
    """Semantic search using INDEX; returns list of (score, mapping_entry)."""
    global INDEX
    if not query:
        return []
    if INDEX is None:
        built = build_index(force_rebuild=False)
        if not built or INDEX is None:
            logger.warning("Index not available; semantic search skipped.")
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)

    idx_dim = _index_dim(INDEX)
    if idx_dim and vec.shape[1] != idx_dim:
        logger.info("Query dim mismatch; attempting rebuild with matching model.")
        desired_model = choose_embedding_model_for_dim(idx_dim)
        if OPENAI_API_KEY:
            global EMBEDDING_MODEL
            EMBEDDING_MODEL = desired_model
            rebuilt = build_index(force_rebuild=True)
            if not rebuilt:
                logger.error("Rebuild failed; cannot perform search.")
                return []
            emb2, _ = embed_text(query)
            if not emb2:
                return []
            vec = np.array(emb2, dtype="float32").reshape(1, -1)
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        else:
            logger.error("No OPENAI_API_KEY; cannot rebuild model-matched index.")
            return []

    try:
        D, I = INDEX.search(vec, top_k)
    except Exception:
        logger.exception("Index.search failed")
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
        logger.exception("Failed to parse search results")
    return results

# ---------- Session helpers ----------
_in_memory_sessions: Dict[str, Dict[str,Any]] = {}

def get_session(session_id: Optional[str]) -> Tuple[str, Dict[str,Any]]:
    if not session_id:
        session_id = f"anon-{int(time.time()*1000)}"
    data = None
    if session_store and hasattr(session_store, "load_session"):
        try:
            data = session_store.load_session(session_id)
        except Exception:
            data = None
    if data is None:
        data = _in_memory_sessions.get(session_id, {})
    data.setdefault("last_tour_index", None)
    data.setdefault("recent_entities", [])
    data.setdefault("last_clarify_options", None)  # store clarify choices if asked
    return session_id, data

def save_session(session_id: str, data: Dict[str,Any]):
    if session_store and hasattr(session_store, "save_session"):
        try:
            session_store.save_session(session_id, data)
            return
        except Exception:
            pass
    _in_memory_sessions[session_id] = data

# ---------- Prompt composition ----------
def compose_system_prompt_passages(top_passages: List[Tuple[float, dict]]) -> str:
    header = (
        "Bạn là Ruby Wings AI, trợ lý du lịch trải nghiệm và chữa lành. "
        "Trả lời thân thiện, ấm áp và ngắn gọn. "
        "LUÔN chỉ sử dụng thông tin trong 'DỮ LIỆU NỘI BỘ' được cung cấp và trích dẫn nguồn. "
        "Nếu không có thông tin, trả lời: 'Không có thông tin trong tài liệu.'"
        "\n\nDỮ LIỆU NỘI BỘ (theo liên quan):\n"
    )
    if not top_passages:
        return header + "Không tìm thấy dữ liệu nội bộ phù hợp."
    content = header
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"
    content += "\n---\nNhớ: chỉ dùng dữ liệu trên; không bịa; giọng nói thân thiện."
    return content

def call_llm_for_answer(query: str, top_passages: List[Tuple[float, dict]]) -> str:
    """Call OpenAI chat (new SDK) to produce friendly answer that cites sources."""
    if not OPENAI_API_KEY or OpenAI is None:
        # deterministic fallback: join top passages succinctly
        if top_passages:
            snippets = []
            for i, (_, m) in enumerate(top_passages[:3], start=1):
                snippets.append(f"[{i}] {m.get('text','')}")
            return "Tôi tìm thấy:\n" + "\n\n".join(snippets)
        return "Không có thông tin trong tài liệu."
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        system_prompt = compose_system_prompt_passages(top_passages)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        resp = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=512,
            top_p=0.95
        )
        if resp.choices and len(resp.choices) > 0:
            return resp.choices[0].message.content.strip()
    except Exception:
        logger.exception("OpenAI chat failed - falling back to deterministic reply.")
    # fallback
    if top_passages:
        snippets = []
        for i, (_, m) in enumerate(top_passages[:3], start=1):
            snippets.append(f"[{i}] {m.get('text','')}")
        return "Tìm thấy:\n" + "\n\n".join(snippets)
    return "Không có thông tin trong tài liệu."

# ---------- Chat route ----------
@app.route("/chat", methods=["POST"])
def chat():
    """
    Behavior:
    - Determine requested_field via intent_classifier (if available) or KEYWORD_FIELD_MAP.
    - Detect tour mention(s) and restrict results accordingly.
    - If ambiguous tours and clarifier indicates to ask, return clarify payload.
    - Update session when a specific tour is selected.
    - Answer using retrieval-first approach; LLM for friendly phrasing but only with retrieval evidence.
    """
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or data.get("q") or "").strip()
    session_id = data.get("session_id") or request.headers.get("X-Session-Id")
    if not user_message:
        return jsonify({"error": "empty message"}), 400

    sid, session_data = get_session(session_id)

    # 1) If user is responding to a previous clarify question by number, handle selection
    last_opts = session_data.get("last_clarify_options")
    if last_opts and re.fullmatch(r"\d+", user_message.strip()):
        sel = int(user_message.strip())
        if 1 <= sel <= len(last_opts):
            chosen = last_opts[sel-1]
            # chosen is {"tour_index":int,"tour_name":str}
            session_data["last_tour_index"] = chosen.get("tour_index")
            session_data["recent_entities"].append({"type":"tour","id": chosen.get("tour_index"), "ts": int(time.time())})
            session_data["last_clarify_options"] = None
            save_session(sid, session_data)
            return jsonify({"answer": f'Đã chọn: {chosen.get("tour_name")}', "session_id": sid})

    # 2) Intent / field detection
    requested_field = None
    confidence = 0.0
    if intent_classifier and hasattr(intent_classifier, "detect_field"):
        try:
            requested_field, confidence = intent_classifier.detect_field(user_message)
        except Exception:
            requested_field, confidence = None, 0.0

    # Fallback to keyword map if none detected
    if not requested_field:
        text_l = user_message.lower()
        for _, v in KEYWORD_FIELD_MAP.items():
            for kw in v["keywords"]:
                if kw in text_l:
                    requested_field = v["field"]
                    confidence = max(confidence, 0.5)
                    break
            if requested_field:
                break

    # 3) tour detection
    tour_indices = find_tour_indices_from_message(user_message)
    # if no explicit tour mention, but session has last_tour_index and message contains pronouns or booking keywords, prefer that
    if not tour_indices and session_data.get("last_tour_index") is not None:
        # use coref to check if pronoun refers to last tour, or presence of booking-related verbs
        if coref and hasattr(coref, "resolve_pronoun"):
            res = coref.resolve_pronoun(user_message, session_data)
            if res and res.get("resolved_entity") is not None:
                tour_indices = [res["resolved_entity"]]
        # simple heuristic: words like "còn chỗ", "còn vé" -> use last_tour_index
        if not tour_indices and re.search(r"\bcòn\b.*\b(chỗ|vé|slot)\b", user_message.lower()):
            tour_indices = [session_data.get("last_tour_index")]

    # 4) Retrieval decision
    top_results = []
    # If explicit request for list of tours
    if requested_field == "tour_name":
        top_results = get_passages_by_field("tour_name", limit=200)
    elif requested_field and tour_indices:
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=tour_indices)
        if not top_results:
            top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
    elif requested_field:
        top_results = get_passages_by_field(requested_field, limit=TOP_K, tour_indices=None)
        if not top_results:
            top_results = query_index(user_message, TOP_K)
    else:
        # no explicit field -> semantic search
        top_results = query_index(user_message, TOP_K)

    # 5) Clarifier check: ask only when multiple candidate tours and low confidence
    candidate_tour_ids = []
    # extract tour ids from top_results paths when available
    for _, m in top_results[:6]:
        p = m.get("path", "")
        mm = re.search(r"\[(\d+)\]", p)
        if mm:
            candidate_tour_ids.append(int(mm.group(1)))
    candidate_tour_ids = sorted(set(candidate_tour_ids))

    ask_clarify = False
    if clarifier and hasattr(clarifier, "need_clarify"):
        try:
            ask_clarify = clarifier.need_clarify(requested_field, candidate_tour_ids, confidence)
        except Exception:
            ask_clarify = False

    if ask_clarify and candidate_tour_ids:
        # build options (limit 4) and store in session
        opts = []
        for ti in candidate_tour_ids[:4]:
            # find representative tour_name
            name = None
            for m in MAPPING:
                if m.get("path","").endswith(f"tours[{ti}].tour_name"):
                    name = m.get("text")
                    break
            if not name:
                name = f"Tour #{ti}"
            opts.append({"tour_index": ti, "tour_name": name})
        session_data["last_clarify_options"] = opts
        save_session(sid, session_data)
        q = clarifier.build_clarify_question(opts) if clarifier and hasattr(clarifier, "build_clarify_question") else (
            "Bạn đang nói đến tour nào? Chọn số:\n" + "\n".join(f"{i+1}) {o['tour_name']}" for i,o in enumerate(opts))
        )
        metrics and hasattr(metrics, "incr") and metrics.incr("clarify_asked")  # type: ignore
        return jsonify({"clarify": True, "question": q, "session_id": sid})

    # 6) If top_results include an exact tour match and we can determine tour_index, update session
    if candidate_tour_ids:
        # prefer first candidate as selected context (do not overwrite if user explicitly didn't want)
        selected = candidate_tour_ids[0]
        session_data["last_tour_index"] = selected
        session_data["recent_entities"].append({"type":"tour","id": selected, "ts": int(time.time())})
        session_data["recent_entities"] = session_data["recent_entities"][-10:]
        save_session(sid, session_data)

    # 7) Compose final answer: attempt LLM for friendly phrasing but only with retrieved evidence
    final_answer = call_llm_for_answer(user_message, top_results)
    metrics and hasattr(metrics, "incr") and metrics.incr("chat_requests")  # type: ignore

    # prepare sources list (paths)
    sources = [m.get("path") for _, m in top_results[:5]]

    return jsonify({"answer": final_answer, "sources": sources, "session_id": sid})

# ---------- Reindex endpoint ----------
@app.route("/reindex", methods=["POST"])
def reindex():
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
        return jsonify({"error": "reindex not allowed (set RBW_ALLOW_REINDEX=1 or provide X-RBW-ADMIN)"}), 403
    # reload knowledge then run build_index.py (safer, reuse offline builder)
    load_knowledge()
    try:
        proc = subprocess.run([sys.executable, "build_index.py"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1800)
        out = proc.stdout.decode("utf-8", "ignore")
        err = proc.stderr.decode("utf-8", "ignore")
        # try to reload mapping/index
        load_mapping_from_disk(FAISS_MAPPING_PATH)
        build_index(force_rebuild=False)
        return jsonify({"ok": True, "out": out, "err": err})
    except Exception as e:
        logger.exception("Reindex failed")
        return jsonify({"error": str(e)}), 500

# ---------- Health & root ----------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "ok": True,
        "mapping_count": len(MAPPING),
        "index_present": INDEX is not None,
        "faiss": HAS_FAISS and FAISS_ENABLED
    })

@app.route("/", methods=["GET"])
def root():
    return jsonify({"service":"Ruby Wings AI", "status":"ok", "mapping_entries": len(MAPPING)})

# ---------- Initialization ----------
try:
    load_knowledge()
    # If mapping file exists replace MAPPING to ensure stable ordering
    if os.path.exists(FAISS_MAPPING_PATH):
        try:
            with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                file_map = json.load(f)
            if file_map and (len(file_map) == len(MAPPING) or len(MAPPING) == 0):
                MAPPING[:] = file_map
                FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
                index_tour_names()
                logger.info("Mapping overwritten from disk mapping.json")
        except Exception:
            logger.exception("Could not load FAISS_MAPPING_PATH at startup; proceeding with runtime-scan mapping.")
    # build index in background to avoid blocking startup
    t = threading.Thread(target=build_index, kwargs={"force_rebuild": False}, daemon=True)
    t.start()
except Exception:
    logger.exception("Initialization error")

if __name__ == "__main__":
    # ensure mapping persisted
    if MAPPING and not os.path.exists(FAISS_MAPPING_PATH):
        try:
            with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                json.dump(MAPPING, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    built = build_index(force_rebuild=False)
    if not built:
        logger.warning("Index not ready at startup; endpoints will attempt on-demand build.")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))