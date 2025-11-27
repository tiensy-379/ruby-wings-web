# app.py — Optimized for openai==0.28.0 with auto-detect embedding dim
# Modified only to add keyword → field priority mapping per requirements.
import os, json, threading, logging
from functools import lru_cache
from typing import List, Tuple
from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import openai
import numpy as np

# FAISS import
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw")

# config (defaults)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
openai.api_key = OPENAI_API_KEY

KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

# default embedding model (will be overridden if index exists)
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true")

app = Flask(__name__)
CORS(app)

# global state
KNOW = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []
INDEX = None
INDEX_LOCK = threading.Lock()

# ---------------- Keyword → field mapping (priority)
# Each key maps to the target JSON field to prioritize
KEYWORD_FIELD_MAP = {
    # tour listing
    "tour_list": {
        "keywords": [
            "tên tour", "tour gì", "danh sách tour", "có những tour nào", "liệt kê tour",
            "show tour", "tour hiện có", "tour available", "list tour", "tour đang bán",
            "tour hiện hành", "tour nào", "tours", "list tours", "show tours"
        ],
        "field": "tour_name"
    },
    # mission
    "mission": {
        "keywords": [
            "tầm nhìn", "sứ mệnh", "giá trị cốt lõi", "triết lý", "định hướng phát triển",
            "mục tiêu doanh nghiệp", "mục tiêu hoạt động", "vision", "mission", "core values"
        ],
        "field": "mission"
    },
    # summary
    "summary": {
        "keywords": [
            "tóm tắt chương trình tour", "tóm tắt", "tóm lược", "tổng hợp", "overview",
            "giới thiệu nhanh", "mô tả ngắn", "short description", "brief"
        ],
        "field": "summary"
    },
    # style
    "style": {
        "keywords": [
            "phong cách hành trình", "tính chất hành trình", "concept tour", "vibe tour",
            "kiểu tour", "đặc trưng hành trình", "style of trip", "style"
        ],
        "field": "style"
    },
    # transport
    "transport": {
        "keywords": [
            "vận chuyển", "vận tải", "phương tiện", "đi bằng gì", "di chuyển bằng gì",
            "xe gì", "phương tiện sử dụng", "transportation", "transfer"
        ],
        "field": "transport"
    },
    # includes / itinerary / detailed program
    "includes": {
        "keywords": [
            "lịch trình chi tiết", "chương trình", "chương trình chi tiết",
            "chi tiết hành trình", "chương trình tour", "itinerary", "schedule",
            "timeline chuyến đi", "lịch trình", "program", "detailed itinerary"
        ],
        "field": "includes"
    },
    # location
    "location": {
        "keywords": [
            "ở đâu", "đi đâu", "địa phương nào", "nơi nào", "tỉnh nào", "thành phố nào",
            "điểm đến nào", "destination", "location", "vùng nào"
        ],
        "field": "location"
    },
    # price
    "price": {
        "keywords": [
            "giá tour", "giá tham quan", "chi phí", "bao nhiêu tiền", "giá trọn gói",
            "giá người lớn", "giá trẻ em", "price", "cost", "giá vé"
        ],
        "field": "price"
    },
    # notes
    "notes": {
        "keywords": [
            "lưu ý gì", "ghi chú", "điểm chú ý", "cần chú ý", "cần biết", "notes",
            "important notes", "lưu ý đặc biệt"
        ],
        "field": "notes"
    },
    # accommodation
    "accommodation": {
        "keywords": [
            "chỗ ở", "nơi lưu trú", "ngủ nghỉ ở đâu", "khách sạn", "homestay",
            "resort", "nhà nghỉ", "tiêu chuẩn lưu trú", "accommodation"
        ],
        "field": "accommodation"
    },
    # meals
    "meals": {
        "keywords": [
            "ăn uống", "ẩm thực", "đặc sản", "ăn gì", "meals", "thực đơn",
            "ăn uống trong tour", "meals included"
        ],
        "field": "meals"
    },
    # event support / support service
    "event_support": {
        "keywords": [
            "hỗ trợ", "bổ trợ", "giúp đỡ", "tăng cường", "dịch vụ tăng cường",
            "dịch vụ gia tăng", "support service", "additional support", "event support"
        ],
        "field": "event_support"
    },
    # cancellation policy
    "cancellation_policy": {
        "keywords": [
            "phí huỷ tour", "phí huỷ hành trình", "hoãn hành trình", "đổi ngày",
            "đổi lịch", "refund policy", "cancellation rules", "chính sách huỷ"
        ],
        "field": "cancellation_policy"
    },
    # booking method
    "booking_method": {
        "keywords": [
            "phương pháp đặt chỗ", "cách đặt chỗ", "đặt tour", "cách book",
            "booking", "đặt như thế nào", "quy trình đặt tour", "đặt chỗ"
        ],
        "field": "booking_method"
    },
    # who can join
    "who_can_join": {
        "keywords": [
            "phù hợp đối tượng", "ai tham gia", "người tham gia", "độ tuổi phù hợp",
            "đối tượng khách", "phù hợp với ai", "participant type", "who should join"
        ],
        "field": "who_can_join"
    },
    # hotline / contact
    "hotline": {
        "keywords": [
            "số điện thoại liên hệ", "nhân viên tư vấn", "gặp trực tiếp nhân viên",
            "hotline", "số nóng", "gọi ngay", "contact number", "tư vấn viên", "liên hệ"
        ],
        "field": "hotline"
    }
}

# numpy fallback index
class NumpyIndex:
    def __init__(self, mat=None):
        self.mat = mat.astype("float32") if (isinstance(mat, np.ndarray) and mat.size>0) else np.empty((0,0), dtype="float32")
        self.dim = None if self.mat.size==0 else self.mat.shape[1]

    def search(self, qvec, k):
        if self.mat.size == 0:
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
            mat = arr["mat"]
            return cls(mat=mat)
        except Exception:
            return cls(None)

    def save(self, path):
        try:
            np.savez_compressed(path, mat=self.mat)
        except Exception:
            logger.exception("Failed to save fallback vectors")

# load knowledge (flatten)
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
    logger.info("Knowledge loaded: %d passages", len(FLAT_TEXTS))

# helper - determine index dim
def _index_dim(idx):
    try:
        d = getattr(idx, "d", None)
        if isinstance(d, int) and d>0:
            return d
    except:
        pass
    try:
        d = getattr(idx, "dim", None)
        if isinstance(d, int) and d>0:
            return d
    except:
        pass
    try:
        if HAS_FAISS and isinstance(idx, faiss.Index):
            return int(idx.d)
    except:
        pass
    return None

# automatic embedding model selection from index dim
def choose_embedding_model_for_dim(dim):
    if dim == 1536:
        return "text-embedding-3-small"
    if dim == 3072:
        return "text-embedding-3-large"
    return os.environ.get("EMBEDDING_MODEL", EMBEDDING_MODEL)

# embedding function (uses current EMBEDDING_MODEL variable)
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    try:
        resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=short)
        emb = None
        if isinstance(resp, dict) and "data" in resp and len(resp["data"])>0:
            emb = resp["data"][0].get("embedding") or resp["data"][0].get("vector")
        if emb:
            return emb, len(emb)
    except Exception:
        logger.exception("OpenAI embedding failed; using deterministic fallback")
    try:
        h = abs(hash(short)) % (10**12)
        fallback_dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(fallback_dim)]
        return vec, fallback_dim
    except Exception:
        return [], 0

# build or load index - plus auto-detect embedding model if index exists
def build_index(force_rebuild=False):
    global INDEX, MAPPING, FLAT_TEXTS, EMBEDDING_MODEL
    with INDEX_LOCK:
        use_faiss = FAISS_ENABLED and HAS_FAISS

        if not force_rebuild:
            if use_faiss and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    idx = faiss.read_index(FAISS_INDEX_PATH)
                    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        MAPPING = json.load(f)
                    FLAT_TEXTS[:] = [m.get("text","") for m in MAPPING]
                    idx_dim = _index_dim(idx)
                    if idx_dim:
                        EMBEDDING_MODEL = choose_embedding_model_for_dim(idx_dim)
                        logger.info("Detected index dim=%s -> using embedding model=%s", idx_dim, EMBEDDING_MODEL)
                    INDEX = idx
                    logger.info("FAISS index loaded from disk.")
                    return True
                except Exception:
                    logger.exception("Failed loading FAISS index; will rebuild.")

            if os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    arr = np.load(FALLBACK_VECTORS_PATH)
                    mat = arr["mat"]
                    idx = NumpyIndex(mat)
                    with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                        MAPPING = json.load(f)
                    FLAT_TEXTS[:] = [m.get("text","") for m in MAPPING]
                    idx_dim = getattr(idx, "dim", None)
                    if idx_dim:
                        EMBEDDING_MODEL = choose_embedding_model_for_dim(int(idx_dim))
                        logger.info("Detected fallback vectors dim=%s -> using embedding model=%s", idx_dim, EMBEDDING_MODEL)
                    INDEX = idx
                    logger.info("Numpy fallback index loaded.")
                    return True
                except Exception:
                    logger.exception("Failed loading fallback vectors; will rebuild.")

        if not FLAT_TEXTS:
            logger.warning("No flattened texts to index (build aborted).")
            return False

        logger.info("Building embeddings for %d passages (model=%s)...", len(FLAT_TEXTS), EMBEDDING_MODEL)
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
                    faiss.write_index(index, FAISS_INDEX_PATH)
                    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
                        json.dump(MAPPING, f, ensure_ascii=False, indent=2)
                except Exception:
                    logger.exception("Failed to persist FAISS index/mapping")
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
                logger.info("Numpy index built (dims=%d, n=%d).", dims, idx.mat.shape[0])
                return True
        except Exception:
            logger.exception("Error while building index")
            INDEX = None
            return False

# query index (with dimension safety)
def query_index(query: str, top_k=TOP_K):
    global INDEX
    if not query:
        return []
    if INDEX is None:
        ok = build_index(force_rebuild=False)
        if not ok or INDEX is None:
            logger.warning("Index not available")
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    vec = np.array(emb, dtype="float32").reshape(1, -1)
    vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)

    idx_dim = _index_dim(INDEX)
    if idx_dim and vec.shape[1] != idx_dim:
        logger.error("Query dim %s != index dim %s. Will attempt rebuild with matching model.", vec.shape[1], idx_dim)
        desired_model = choose_embedding_model_for_dim(idx_dim)
        if OPENAI_API_KEY:
            global EMBEDDING_MODEL
            EMBEDDING_MODEL = desired_model
            logger.info("Setting EMBEDDING_MODEL=%s and rebuilding index...", EMBEDDING_MODEL)
            rebuilt = build_index(force_rebuild=True)
            if not rebuilt:
                logger.error("Rebuild failed; abort search.")
                return []
            emb2, d2 = embed_text(query)
            if not emb2:
                return []
            vec = np.array(emb2, dtype="float32").reshape(1,-1)
            vec = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
        else:
            logger.error("No OPENAI_API_KEY; cannot rebuild model-matched index.")
            return []

    try:
        D, I = INDEX.search(vec, top_k)
    except Exception:
        logger.exception("Index search error")
        return []

    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(MAPPING):
            continue
        results.append((float(score), MAPPING[idx]))
    return results

# ---------------- Helper: get passages by exact field name ----------------
def get_passages_by_field(field_name: str, limit: int = 50):
    """
    Return list of (score, mapping_entry) for entries whose path ends with the given field.
    Score is fixed (1.0) so they are prioritized.
    """
    res = []
    for m in MAPPING:
        # path examples: root.tours[0].tour_name or root.about_company.mission
        path = m.get("path", "")
        # exact field at end or contains ".<field>" before brackets
        if path.endswith(f".{field_name}") or f".{field_name}" in path:
            res.append((1.0, m))
    # keep original order, limit
    return res[:limit]

# compose system prompt
def compose_system_prompt(top_passages):
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

# endpoints
app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "knowledge_count": len(FLAT_TEXTS),
        "index_exists": INDEX is not None,
        "index_dim": _index_dim(INDEX),
        "embedding_model": EMBEDDING_MODEL,
        "faiss_available": HAS_FAISS,
        "faiss_enabled": FAISS_ENABLED
    })

@app.route("/reindex", methods=["POST"])
def reindex():
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and os.environ.get("RBW_ALLOW_REINDEX","") != "1":
        return jsonify({"error":"reindex not allowed"}), 403
    ok = build_index(force_rebuild=True)
    return jsonify({"ok": ok, "count": len(FLAT_TEXTS)})

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json() or {}
    user_message = data.get("message","").strip()
    if not user_message:
        return jsonify({"reply":"Bạn chưa nhập câu hỏi."})

    # --- intent: keyword detection to prioritize specific field ---
    text_l = user_message.lower()
    top_override = []
    # iterate KEYWORD_FIELD_MAP in insertion order to prioritize
    for k, v in KEYWORD_FIELD_MAP.items():
        for kw in v["keywords"]:
            if kw in text_l:
                field = v["field"]
                top_override = get_passages_by_field(field, limit=TOP_K)
                # special behavior: if listing tours (tour_name) and user asked for list,
                # we return all tour_name entries (not limited strictly by TOP_K)
                if field == "tour_name":
                    # get all tour_name passages
                    all_tours = get_passages_by_field("tour_name", limit=1000)
                    top_override = all_tours
                break
        if top_override:
            break

    # If override found, use it first; else run semantic search
    if top_override:
        top = top_override
    else:
        top_k = int(data.get("top_k", TOP_K))
        top = query_index(user_message, top_k)

    system_prompt = compose_system_prompt(top)
    messages = [{"role":"system","content":system_prompt}, {"role":"user","content":user_message}]

    # call OpenAI ChatCompletion (SDK v0.28)
    reply = ""
    if OPENAI_API_KEY:
        try:
            resp = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages, temperature=0.2, max_tokens=int(data.get("max_tokens",700)))
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
            logger.exception("OpenAI chat failed; fallback to internal snippet")

    if not reply:
        if top:
            # If override was tour_name listing, format as list of names
            # detect if top entries are all from tour_name
            if all(("tour_name" in (m.get("path","")) or ".tour_name" in m.get("path","")) for _, m in top):
                names = [m.get("text","") for _, m in top]
                # deduplicate preserve order
                seen = set()
                names_u = [x for x in names if not (x in seen or seen.add(x))]
                reply = "Các tour hiện có:\n" + "\n".join(f"- {n}" for n in names_u)
            else:
                snippets = "\n\n".join([f"- {m.get('text')}" for _, m in top[:5]])
                reply = f"Tôi tìm thấy thông tin nội bộ liên quan:\n\n{snippets}"
        else:
            reply = "Xin lỗi — hiện không có dữ liệu nội bộ liên quan."

    return jsonify({"reply": reply, "sources":[m for _, m in top]})

# init on import
load_knowledge()
build_index(force_rebuild=False)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",10000)))
