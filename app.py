# app.py — "HOÀN HẢO NHẤT" phiên bản tối ưu: Context-Aware + Rule-Based Priority + Vector Fallback
# Mục tiêu:
# 1. Ưu tiên 1: Trả lời đúng trường (Field) nếu khớp từ khóa (Rule-Based).
# 2. Ưu tiên 2: Tìm kiếm ngữ nghĩa (Vector Search) nếu không khớp từ khóa chính xác.
# 3. Context Awareness: Nhớ tour đang nói đến, hỗ trợ chuyển ngữ cảnh khi nhắc địa danh.

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

# Try FAISS; fallback to numpy-only index if missing
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except Exception:
    HAS_FAISS = False

# ✅ OPENAI API MỚI
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("rbw")

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# ✅ KHỞI TẠO CLIENT OPENAI
client = None
if OPENAI_API_KEY and OpenAI is not None:
    client = OpenAI(api_key=OPENAI_API_KEY)
else:
    logger.warning("OPENAI_API_KEY not set — embeddings/chat will fallback to deterministic behavior.")

KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

# ---------- Session Management ----------
USER_SESSIONS = {}
SESSION_TIMEOUT = 600  # 10 phút

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Global state ----------
KNOWLEDGE_BASE: Dict[str, Any] = {} # Raw JSON data
TOURS: List[Dict] = []
KNOW: Dict = {} # Flat map for vector search
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []  # list of {"path": "...", "text": "..."}
INDEX = None
INDEX_LOCK = threading.Lock()

# Indices for Rule-Based Logic
TOUR_NAME_TO_INDEX: Dict[str, int] = {}
LOCATION_INDEX: Dict[str, List[int]] = {}

# ---------- Keyword -> field mapping (priority) ----------
# Đây là "kim chỉ nam" để bot trả lời đúng trường
KEYWORD_FIELD_MAP = {
    "tour_list": {
        "keywords": ["danh sach tour", "co nhung tour nao", "list tour", "cac tour", "tour hien co", "tour nao", "tour gi"],
        "field": "LIST_TOURS"
    },
    "price": {
        "keywords": ["gia", "chi phi", "bao nhieu tien", "tien tour", "cost", "price", "bang gia", "gia tour"],
        "field": "price"
    },
    "includes": {
        "keywords": ["lich trinh", "chuong trinh", "di dau", "tham quan", "chi tiet", "itinerary", "schedule", "includes"],
        "field": "includes"
    },
    "duration": {
        "keywords": ["thoi gian", "may ngay", "bao lau", "ngay dem", "may dem", "duration", "khoi hanh"],
        "field": "duration"
    },
    "transport": {
        "keywords": ["xe", "di chuyen", "phuong tien", "o to", "may bay", "transport", "di bang gi"],
        "field": "transport"
    },
    "meals": {
        "keywords": ["an uong", "thuc don", "bua an", "an gi", "meals", "food", "am thuc"],
        "field": "meals"
    },
    "accommodation": {
        "keywords": ["o dau", "khach san", "nha nghi", "homestay", "ngu dau", "hotel", "luu tru"],
        "field": "accommodation"
    },
    "notes": {
        "keywords": ["luu y", "chu y", "can mang theo", "trang phuc", "note", "chuan bi"],
        "field": "notes"
    },
    "policy": {
         "keywords": ["huy tour", "hoan tien", "chinh sach", "policy", "refund"],
         "field": "cancellation_policy" # Giả sử trong JSON có trường này hoặc notes
    },
    "contact": {
        "keywords": ["lien he", "sdt", "so dien thoai", "hotline", "tu van", "gap nhan vien", "nhan vien"],
        "field": "CONTACT_INFO"
    }
}

# ---------- Utilities ----------
def normalize_text_simple(s: str) -> str:
    """Lowercase, remove diacritics, strip punctuation."""
    if not s: return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_raw_data():
    """Load raw JSON for Rule-Based lookup"""
    global KNOWLEDGE_BASE, TOURS, TOUR_NAME_TO_INDEX, LOCATION_INDEX
    try:
        if os.path.exists(KNOWLEDGE_PATH):
            with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                KNOWLEDGE_BASE = json.load(f)
                TOURS = KNOWLEDGE_BASE.get("tours", [])
                
            # Build Rule-Based Indices
            TOUR_NAME_TO_INDEX = {}
            LOCATION_INDEX = {}
            
            for idx, tour in enumerate(TOURS):
                # Index Name
                norm_name = normalize_text_simple(tour.get("tour_name", ""))
                TOUR_NAME_TO_INDEX[norm_name] = idx
                
                # Index Location (keywords)
                loc_str = normalize_text_simple(tour.get("location", ""))
                parts = [p.strip() for p in loc_str.split()]
                for p in parts:
                    if len(p) > 2:
                        if p not in LOCATION_INDEX: LOCATION_INDEX[p] = []
                        if idx not in LOCATION_INDEX[p]: LOCATION_INDEX[p].append(idx)
            
            logger.info(f"✅ Loaded {len(TOURS)} tours for Rule-Based Engine.")
    except Exception as e:
        logger.error(f"Failed to load raw data: {e}")

# ---------- Vector Search / Embedding Logic (Preserved) ----------
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    """Generate embedding (OpenAI or Deterministic Fallback)"""
    if not text: return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    
    if client is not None:
        try:
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=short)
            if resp.data:
                emb = resp.data[0].embedding
                return emb, len(emb)
        except Exception:
            logger.warning("OpenAI embedding failed, using fallback.")
    
    # Fallback
    h = abs(hash(short)) % (10 ** 12)
    dim = 1536
    vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]
    return vec, dim

class NumpyIndex:
    """Simple in-memory index fallback"""
    def __init__(self, mat=None):
        self.mat = mat.astype("float32") if mat is not None else None
    def search(self, qvec, k):
        if self.mat is None: return np.array([[]]), np.array([[]])
        q = qvec.astype("float32")
        q /= (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
        m = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        sims = np.dot(q, m.T)
        idx = np.argsort(-sims, axis=1)[:, :k]
        return np.take_along_axis(sims, idx, axis=1), idx

def load_index():
    global INDEX, MAPPING, FLAT_TEXTS
    with INDEX_LOCK:
        if HAS_FAISS and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
            try:
                INDEX = faiss.read_index(FAISS_INDEX_PATH)
                with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                    MAPPING = json.load(f)
                logger.info("✅ FAISS Index loaded.")
                return
            except Exception: pass
            
        if os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
            try:
                arr = np.load(FALLBACK_VECTORS_PATH)
                INDEX = NumpyIndex(arr["mat"])
                with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                    MAPPING = json.load(f)
                logger.info("✅ Numpy Index loaded.")
                return
            except Exception: pass
        
        logger.warning("⚠️ No vector index found. Search will rely on Rule-Based only.")

# ---------- Core Logic Functions ----------

def get_session(session_id: str):
    """Get or create user session"""
    now = datetime.now()
    if session_id not in USER_SESSIONS:
        USER_SESSIONS[session_id] = {
            "last_tour_idx": None,
            "last_interaction": now,
            "history": []
        }
    else:
        if (now - USER_SESSIONS[session_id]["last_interaction"]).seconds > SESSION_TIMEOUT:
            USER_SESSIONS[session_id] = {"last_tour_idx": None, "last_interaction": now, "history": []}
        USER_SESSIONS[session_id]["last_interaction"] = now
    return USER_SESSIONS[session_id]

def find_tour_context(msg_norm: str) -> Tuple[Optional[int], List[int]]:
    """
    Return: (Exact Tour Index, List of Location-Matched Tour Indices)
    """
    # 1. Check Tour Name (Exact/Substring)
    best_match = None
    max_len = 0
    for name_norm, idx in TOUR_NAME_TO_INDEX.items():
        if name_norm in msg_norm:
            if len(name_norm) > max_len:
                max_len = len(name_norm)
                best_match = idx
    
    # 2. Check Location
    loc_matches = set()
    for loc_kw, indices in LOCATION_INDEX.items():
        if loc_kw in msg_norm:
            for idx in indices:
                loc_matches.add(idx)
                
    return best_match, list(loc_matches)

def detect_intent(msg_norm: str) -> Optional[str]:
    """Detect what field user is asking about"""
    for key, config in KEYWORD_FIELD_MAP.items():
        for kw in config["keywords"]:
            if kw in msg_norm:
                return config["field"]
    return None

def search_vector_db(query: str, tour_idx: Optional[int] = None, limit=3):
    """Semantic search, optionally boosted by tour context"""
    if INDEX is None: return []
    emb, _ = embed_text(query)
    if not emb: return []
    
    scores, indices = INDEX.search(np.array([emb], dtype="float32"), k=TOP_K * 2) # Fetch more to filter
    results = []
    
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(MAPPING): continue
        item = MAPPING[idx]
        path = item.get("path", "")
        
        # Boost score if matches context tour
        final_score = score
        if tour_idx is not None and f"[{tour_idx}]" in path:
            final_score += 0.3 # Boost logic
            
        results.append((final_score, item["text"], path))
        
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:limit]

# ---------- MAIN GENERATION LOGIC ----------

def generate_response(user_msg: str, session_id: str) -> str:
    session = get_session(session_id)
    msg_norm = normalize_text_simple(user_msg)
    
    # 1. Detect Context Change
    detected_tour_idx, loc_matches = find_tour_context(msg_norm)
    current_tour_idx = session["last_tour_idx"]
    
    # Logic chuyển context
    context_switched = False
    if detected_tour_idx is not None:
        current_tour_idx = detected_tour_idx
        session["last_tour_idx"] = current_tour_idx
        context_switched = True
    elif loc_matches:
        if len(loc_matches) == 1:
            current_tour_idx = loc_matches[0]
            session["last_tour_idx"] = current_tour_idx
            context_switched = True
        elif len(loc_matches) > 1:
            # Nhiều tour cùng địa danh -> Hỏi lại
            names = [TOURS[i]["tour_name"] for i in loc_matches]
            return f"Có {len(names)} tour đi qua địa điểm bạn hỏi:\n" + "\n".join([f"- {n}" for n in names]) + "\n\nBạn quan tâm tour nào?"

    # 2. Detect Intent
    intent = detect_intent(msg_norm)
    
    # 3. EXECUTION STRATEGY
    
    # STRATEGY A: Explicit Rule-Based Answer (Highest Priority)
    # Điều kiện: Có context tour + Có intent rõ ràng
    if current_tour_idx is not None and intent:
        # Trừ trường hợp hỏi list tour hoặc contact (global intents)
        if intent == "LIST_TOURS":
             pass # Handle below
        elif intent == "CONTACT_INFO":
             pass # Handle below
        else:
            tour = TOURS[current_tour_idx]
            data = tour.get(intent)
            tour_name = tour.get("tour_name")
            
            if data:
                response = ""
                if context_switched:
                    response += f"Về tour {tour_name}:\n"
                
                if isinstance(data, list):
                    response += "\n".join([f"- {item}" for item in data])
                else:
                    response += str(data)
                return response
            else:
                # Intent rõ nhưng data trống -> Fallback mềm
                return f"Thông tin về '{intent}' của tour {tour_name} hiện chưa được cập nhật chi tiết. Bạn vui lòng liên hệ nhân viên tư vấn để biết thêm nhé."

    # STRATEGY B: Global Rule Intents
    if intent == "LIST_TOURS":
        names = [t["tour_name"] for t in TOURS]
        return "Ruby Wings hiện có các tour:\n" + "\n".join([f"- {n}" for n in names]) + "\n\nBạn muốn xem chi tiết tour nào?"
        
    if intent == "CONTACT_INFO":
        return "Bạn có thể liên hệ hotline: 09xxxxx hoặc ghé văn phòng Ruby Wings để được hỗ trợ nhé!"

    # STRATEGY C: Vector Search (Semantic Fallback)
    # Khi không bắt được intent rõ ràng (ví dụ câu hỏi phức tạp: "tour này có leo núi không?")
    # Hoặc khi chưa có context.
    search_results = search_vector_db(user_msg, current_tour_idx)
    
    if search_results:
        # Kiểm tra score cao nhất
        best_score, best_text, best_path = search_results[0]
        if best_score > 0.4: # Threshold
            # Nếu context khớp, trả lời luôn
            return best_text
            
    # STRATEGY D: LLM Chat (Ultimate Fallback)
    if client:
        try:
            messages = [
                {"role": "system", "content": "Bạn là trợ lý du lịch Ruby Wings. Trả lời ngắn gọn, lịch sự."},
                {"role": "user", "content": user_msg}
            ]
            # Inject context if available
            if current_tour_idx is not None:
                tour = TOURS[current_tour_idx]
                context_str = f"Context: Đang nói về tour {tour['tour_name']}. Summary: {tour.get('summary','')}"
                messages.insert(1, {"role": "system", "content": context_str})
                
            resp = client.chat.completions.create(model=CHAT_MODEL, messages=messages, max_tokens=300)
            return resp.choices[0].message.content
        except Exception as e:
            logger.error(f"LLM Error: {e}")

    # Fallback cuối cùng
    if current_tour_idx:
        return f"Bạn đang hỏi về tour {TOURS[current_tour_idx]['tour_name']} phải không? Bạn có thể hỏi cụ thể về giá, lịch trình hay ăn uống."
    
    return "Xin chào! Tôi là trợ lý Ruby Wings. Bạn cần tìm thông tin về tour nào?"

# ---------- API Endpoints ----------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    msg = data.get("message", "")
    session_id = request.cookies.get("session_id") or str(uuid.uuid4())
    
    if not TOURS: load_raw_data()
    if INDEX is None: load_index()
    
    resp_text = generate_response(msg, session_id)
    
    response = jsonify({"response": resp_text, "session_id": session_id})
    response.set_cookie("session_id", session_id)
    return response

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok", 
        "tours": len(TOURS), 
        "vector_index": INDEX is not None,
        "faiss": HAS_FAISS
    })

if __name__ == "__main__":
    load_raw_data()
    load_index()
    app.run(host="0.0.0.0", port=5000)
