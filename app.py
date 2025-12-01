#!/usr/bin/env python3
"""
app.py — Ruby Wings Tour Chatbot với Ngữ Cảnh Tour Ưu Tiên (v2.0)
Tương thích hoàn toàn với knowledge.json, field_keywords.json, build_index.py mới
Đảm bảo chatbot nhớ ngữ cảnh và ưu tiên thông tin trong tour hiện tại
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
from typing import List, Dict, Optional, Tuple, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import difflib
from collections import defaultdict

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
logger = logging.getLogger("ruby_wings_chatbot")

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FIELD_KEYWORDS_PATH = os.environ.get("FIELD_KEYWORDS_PATH", "field_keywords.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")
TOP_K = int(os.environ.get("TOP_K", "8"))
TOP_K_CONTEXT = int(os.environ.get("TOP_K_CONTEXT", "15"))
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", str(60 * 10)))  # 10 phút
SESSION_STORE = os.environ.get("SESSION_STORE", "memory")  # or 'redis'
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CONTEXT_MEMORY = int(os.environ.get("CONTEXT_MEMORY", "5"))  # Số lượt hỏi giữ context tour

# Initialize OpenAI client if possible
client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("✅ OpenAI client initialized")
    except Exception:
        logger.exception("❌ OpenAI client init failed")
else:
    logger.info("ℹ️ OpenAI client not available; using deterministic responses")

# ---------- Flask app ----------
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ---------- Global state ----------
KNOWLEDGE_DATA: Dict = {}
MAPPING: List[dict] = []
METADATA: List[dict] = []
INDEX = None
INDEX_LOCK = threading.Lock()

# Từ khóa trường dữ liệu
FIELD_KEYWORDS: Dict[str, List[str]] = {}
REVERSE_FIELD_KEYWORDS: Dict[str, str] = {}  # Từ khóa -> trường

# Map tour name -> tour index
TOUR_NAME_TO_INDEX: Dict[str, int] = {}
TOUR_INDEX_TO_INFO: Dict[int, Dict] = {}

# Session backend
USER_SESSIONS: Dict[str, dict] = {}
if SESSION_STORE == "redis" and redis is not None:
    try:
        REDIS_CLIENT = redis.from_url(REDIS_URL)
        logger.info("✅ Using Redis session store: %s", REDIS_URL)
    except Exception:
        logger.exception("❌ Redis init failed; falling back to memory store")
        REDIS_CLIENT = None
else:
    REDIS_CLIENT = None

# ---------- Utilities ----------

def normalize_text(text: str) -> str:
    """Chuẩn hóa văn bản: lowercase, bỏ dấu, chuẩn hóa khoảng trắng"""
    if not text:
        return ""
    
    # Chuyển thành lowercase
    text = text.lower()
    
    # Loại bỏ dấu tiếng Việt
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Thay thế các ký tự đặc biệt bằng khoảng trắng
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def token_set(text: str) -> set:
    """Chuyển văn bản thành set token"""
    return set(normalize_text(text).split())


def jaccard_similarity(set1: set, set2: set) -> float:
    """Tính độ tương đồng Jaccard giữa 2 set"""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0


def levenshtein_similarity(a: str, b: str) -> float:
    """Tính độ tương đồng dựa trên Levenshtein (approximate)"""
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


def extract_keywords(text: str) -> List[str]:
    """Trích xuất từ khóa từ văn bản"""
    text = normalize_text(text)
    words = text.split()
    
    # Loại bỏ stopwords đơn giản (có thể mở rộng)
    stopwords = {'có', 'và', 'hoặc', 'cho', 'về', 'từ', 'đến', 'ở', 'tại', 
                 'là', 'của', 'với', 'bằng', 'theo', 'khi', 'nào', 'gì', 'bao', 'nhiêu'}
    
    keywords = [w for w in words if w not in stopwords and len(w) > 1]
    
    # Thêm bigram cho độ dài vừa
    if len(words) >= 2:
        for i in range(len(words) - 1):
            bigram = f"{words[i]}_{words[i+1]}"
            keywords.append(bigram)
    
    return keywords


# ---------- Field Keywords Management ----------

def load_field_keywords():
    """Tải từ khóa trường dữ liệu từ file"""
    global FIELD_KEYWORDS, REVERSE_FIELD_KEYWORDS
    
    if not os.path.exists(FIELD_KEYWORDS_PATH):
        logger.warning("⚠️ Field keywords file not found, using defaults")
        # Tạo keywords mặc định dựa trên cấu trúc knowledge
        create_default_field_keywords()
        return
    
    try:
        with open(FIELD_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        FIELD_KEYWORDS = {}
        REVERSE_FIELD_KEYWORDS = {}
        
        for field, keywords in data.items():
            if field == "__priority_rules__":
                continue
                
            # Chuẩn hóa field name
            if field.startswith("about_company."):
                norm_field = field
            elif field.startswith("faq."):
                norm_field = field
            elif field.startswith("contact."):
                norm_field = field
            else:
                norm_field = field.split('.')[-1]
            
            FIELD_KEYWORDS[norm_field] = [normalize_text(kw) for kw in keywords]
            
            # Tạo reverse mapping
            for keyword in keywords:
                norm_keyword = normalize_text(keyword)
                REVERSE_FIELD_KEYWORDS[norm_keyword] = norm_field
        
        logger.info(f"✅ Loaded {len(FIELD_KEYWORDS)} field keyword groups")
        logger.info(f"✅ Loaded {len(REVERSE_FIELD_KEYWORDS)} keyword mappings")
        
    except Exception as e:
        logger.error(f"❌ Failed to load field keywords: {e}")
        create_default_field_keywords()


def create_default_field_keywords():
    """Tạo từ khóa mặc định nếu không có file"""
    global FIELD_KEYWORDS, REVERSE_FIELD_KEYWORDS
    
    default_keywords = {
        "tour_name": ["tour này tên gì", "tour gì", "tên tour", "tour nào", "hành trình gì"],
        "summary": ["tóm tắt", "giới thiệu", "mô tả", "overview", "tổng quan"],
        "location": ["đi đâu", "địa điểm", "điểm đến", "location", "khu vực"],
        "duration": ["thời gian", "bao lâu", "mấy ngày", "duration", "kéo dài"],
        "price": ["giá", "chi phí", "bao nhiêu tiền", "price", "giá tour"],
        "includes": ["bao gồm", "gồm những gì", "nội dung", "includes", "hoạt động"],
        "notes": ["lưu ý", "chú ý", "notes", "cần biết", "yêu cầu"],
        "style": ["phong cách", "style", "concept", "định hướng", "loại hình"],
        "transport": ["phương tiện", "xe", "di chuyển", "transport", "vận chuyển"],
        "accommodation": ["ở đâu", "lưu trú", "khách sạn", "homestay", "accommodation"],
        "meals": ["ăn uống", "bữa ăn", "ẩm thực", "meals", "đồ ăn"],
        "event_support": ["hỗ trợ", "support", "dịch vụ", "event support", "chăm sóc"],
        "hotline": ["hotline", "số điện thoại", "liên hệ", "contact", "sdt"],
        "mission": ["sứ mệnh", "mission", "mục tiêu", "giá trị", "ý nghĩa"],
        "includes_extra": ["thêm gì", "extra", "bổ sung", "tùy chọn thêm"],
        "extras": ["không bao gồm", "ngoài giá", "phụ phí", "extras", "tự túc"],
        "additional": ["phụ thu", "extra fee", "chi phí thêm", "phát sinh"],
        "about_company.overview": ["giới thiệu công ty", "ruby wings là gì", "về ruby wings"],
        "about_company.mission": ["sứ mệnh công ty", "mission ruby wings", "tầm nhìn công ty"],
        "faq.cancellation_policy": ["chính sách hủy", "hủy tour", "refund", "hoàn tiền"],
        "faq.booking_method": ["đặt tour", "cách đặt", "book tour", "đăng ký"],
        "faq.who_can_join": ["ai tham gia", "đối tượng", "phù hợp với ai", "trẻ em có đi được không"],
        "contact.hotline": ["hotline công ty", "số điện thoại công ty", "liên hệ công ty"],
        "contact.email": ["email công ty", "gửi mail", "email liên hệ"],
        "contact.office_hours": ["giờ làm việc", "thời gian tư vấn", "mở cửa lúc nào"]
    }
    
    FIELD_KEYWORDS = default_keywords
    REVERSE_FIELD_KEYWORDS = {}
    
    for field, keywords in default_keywords.items():
        for keyword in keywords:
            REVERSE_FIELD_KEYWORDS[normalize_text(keyword)] = field
    
    logger.info("✅ Created default field keywords")


def detect_field_from_query(query: str, context_tour_index: Optional[int] = None) -> Tuple[Optional[str], float]:
    """
    Phát hiện trường dữ liệu từ câu hỏi
    Trả về (field_name, confidence_score)
    """
    query_norm = normalize_text(query)
    query_keywords = extract_keywords(query)
    
    best_field = None
    best_score = 0.0
    
    # Tìm kiếm trong reverse field keywords
    for keyword, field in REVERSE_FIELD_KEYWORDS.items():
        if keyword in query_norm:
            # Tính điểm dựa trên độ dài keyword và vị trí
            score = len(keyword.split('_')) * 0.3  # Ưu tiên bigram
            
            # Ưu tiên field trong context tour
            if context_tour_index is not None and not field.startswith(('about_company.', 'faq.', 'contact.')):
                score += 0.2
            
            if score > best_score:
                best_score = score
                best_field = field
    
    # Nếu không tìm thấy, sử dụng heuristic
    if best_score < 0.3:
        query_lower = query.lower()
        
        # Heuristic cho các trường phổ biến
        heuristics = [
            ("giá", "price", 0.5),
            ("bao nhiêu tiền", "price", 0.7),
            ("chi phí", "price", 0.6),
            ("thời gian", "duration", 0.5),
            ("mấy ngày", "duration", 0.6),
            ("bao lâu", "duration", 0.5),
            ("đi đâu", "location", 0.7),
            ("ở đâu", "location", 0.6),
            ("địa điểm", "location", 0.5),
            ("bao gồm", "includes", 0.6),
            ("gồm những gì", "includes", 0.7),
            ("ăn uống", "meals", 0.7),
            ("bữa ăn", "meals", 0.6),
            ("phương tiện", "transport", 0.6),
            ("xe", "transport", 0.5),
            ("ở đâu", "accommodation", 0.6),
            ("khách sạn", "accommodation", 0.5),
            ("hotline", "hotline", 0.8),
            ("số điện thoại", "hotline", 0.7),
            ("liên hệ", "hotline", 0.6),
            ("lưu ý", "notes", 0.7),
            ("cần biết", "notes", 0.5),
            ("phong cách", "style", 0.6),
            ("hỗ trợ", "event_support", 0.5),
            ("sứ mệnh", "mission", 0.7),
            ("tầm nhìn", "mission", 0.6)
        ]
        
        for keyword, field, base_score in heuristics:
            if keyword in query_lower:
                score = base_score
                
                # Ưu tiên trong context tour
                if context_tour_index is not None and not field.startswith(('about_company.', 'faq.', 'contact.')):
                    score += 0.2
                
                if score > best_score:
                    best_score = score
                    best_field = field
    
    return best_field, best_score


# ---------- Tour Detection ----------

def extract_tour_name_from_query(query: str) -> Optional[Tuple[str, int]]:
    """
    Trích xuất tên tour từ câu hỏi
    Trả về (tour_name, tour_index) nếu tìm thấy
    """
    query_norm = normalize_text(query)
    
    best_match = None
    best_score = 0.0
    
    for tour_name, tour_index in TOUR_NAME_TO_INDEX.items():
        # Kiểm tra xem tour_name có trong query không
        if tour_name in query_norm:
            score = len(tour_name.split()) * 0.3
            if score > best_score:
                best_score = score
                best_match = (tour_name, tour_index)
        
        # Kiểm tra độ tương đồng token
        else:
            tour_tokens = token_set(tour_name)
            query_tokens = token_set(query_norm)
            
            jaccard_score = jaccard_similarity(tour_tokens, query_tokens)
            if jaccard_score > 0.3 and jaccard_score > best_score:
                best_score = jaccard_score
                best_match = (tour_name, tour_index)
    
    return best_match if best_score > 0.3 else None


def find_tour_in_query(query: str) -> List[Tuple[str, int, float]]:
    """
    Tìm tất cả các tour có thể có trong câu hỏi
    Trả về danh sách (tour_name, tour_index, confidence_score)
    """
    results = []
    query_norm = normalize_text(query)
    
    for tour_name, tour_index in TOUR_NAME_TO_INDEX.items():
        score = 0.0
        
        # Exact match
        if tour_name in query_norm:
            score = 0.8
        
        # Partial match
        elif any(word in query_norm for word in tour_name.split()):
            tour_words = set(tour_name.split())
            query_words = set(query_norm.split())
            common_words = tour_words & query_words
            score = len(common_words) / len(tour_words) * 0.6
        
        # Similarity
        else:
            sim_score = levenshtein_similarity(tour_name, query_norm)
            if sim_score > 0.7:
                score = sim_score * 0.5
        
        if score > 0.3:
            results.append((tour_name, tour_index, score))
    
    # Sắp xếp theo điểm
    results.sort(key=lambda x: x[2], reverse=True)
    return results[:3]  # Chỉ lấy top 3


# ---------- Knowledge Loading ----------

def load_knowledge_data():
    """Tải dữ liệu knowledge từ file"""
    global KNOWLEDGE_DATA
    
    if not os.path.exists(KNOWLEDGE_PATH):
        logger.error(f"❌ Knowledge file not found: {KNOWLEDGE_PATH}")
        KNOWLEDGE_DATA = {}
        return
    
    try:
        with open(KNOWLEDGE_PATH, 'r', encoding='utf-8') as f:
            KNOWLEDGE_DATA = json.load(f)
        logger.info(f"✅ Loaded knowledge data with {len(KNOWLEDGE_DATA.get('tours', []))} tours")
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge data: {e}")
        KNOWLEDGE_DATA = {}


def load_mapping_data():
    """Tải mapping data từ file"""
    global MAPPING, METADATA, TOUR_NAME_TO_INDEX, TOUR_INDEX_TO_INFO
    
    if not os.path.exists(FAISS_MAPPING_PATH):
        logger.warning(f"⚠️ Mapping file not found: {FAISS_MAPPING_PATH}")
        MAPPING = []
        METADATA = []
        return
    
    try:
        with open(FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        # Kiểm tra cấu trúc mới
        if isinstance(mapping_data, dict) and "mapping" in mapping_data:
            MAPPING = mapping_data["mapping"]
            METADATA = mapping_data.get("metadata", [])
            logger.info(f"✅ Loaded {len(MAPPING)} mapping entries")
        else:
            # Cấu trúc cũ
            MAPPING = mapping_data
            METADATA = []
            logger.info(f"✅ Loaded {len(MAPPING)} mapping entries (legacy format)")
        
        # Build tour indices
        TOUR_NAME_TO_INDEX.clear()
        TOUR_INDEX_TO_INFO.clear()
        
        for entry in MAPPING:
            if entry.get("is_tour") and entry.get("tour_name") and entry.get("tour_index") is not None:
                tour_name_norm = normalize_text(entry["tour_name"])
                tour_index = entry["tour_index"]
                
                if tour_name_norm not in TOUR_NAME_TO_INDEX:
                    TOUR_NAME_TO_INDEX[tour_name_norm] = tour_index
                
                if tour_index not in TOUR_INDEX_TO_INFO:
                    TOUR_INDEX_TO_INFO[tour_index] = {
                        "name": entry["tour_name"],
                        "name_norm": tour_name_norm,
                        "fields": set()
                    }
                
                TOUR_INDEX_TO_INFO[tour_index]["fields"].add(entry.get("field", ""))
        
        logger.info(f"✅ Indexed {len(TOUR_NAME_TO_INDEX)} unique tours")
        
    except Exception as e:
        logger.error(f"❌ Failed to load mapping data: {e}")
        MAPPING = []
        METADATA = []


def get_tour_info(tour_index: int) -> Optional[Dict]:
    """Lấy thông tin chi tiết của một tour"""
    if tour_index in TOUR_INDEX_TO_INFO:
        info = TOUR_INDEX_TO_INFO[tour_index].copy()
        info["fields"] = list(info["fields"])
        return info
    return None


# ---------- Embedding & Index Management ----------

@lru_cache(maxsize=1024)
def get_text_embedding(text: str) -> np.ndarray:
    """Lấy embedding cho văn bản (có cache)"""
    if not text or not text.strip():
        return np.zeros(1536, dtype=np.float32)
    
    # Cắt ngắn nếu quá dài
    text = text[:4000]
    
    # Sử dụng OpenAI embedding nếu có
    if client and OPENAI_API_KEY:
        try:
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.warning(f"⚠️ OpenAI embedding failed: {e}")
    
    # Fallback: tạo embedding giả
    return generate_synthetic_embedding(text)


def generate_synthetic_embedding(text: str) -> np.ndarray:
    """Tạo embedding tổng hợp cho fallback"""
    text_norm = normalize_text(text)
    words = text_norm.split()
    
    # Tạo vector dựa trên hash của từ
    vector = np.zeros(1536, dtype=np.float32)
    
    for i, word in enumerate(words[:100]):  # Giới hạn 100 từ
        word_hash = hash(word) % 10000
        idx = word_hash % 1536
        vector[idx] += (i + 1) * 0.01  # Thêm trọng số theo vị trí
    
    # Chuẩn hóa
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector


def load_index():
    """Tải FAISS index hoặc fallback index"""
    global INDEX
    
    with INDEX_LOCK:
        if INDEX is not None:
            return INDEX
        
        # Thử tải FAISS index
        if HAS_FAISS and FAISS_ENABLED and os.path.exists(FAISS_INDEX_PATH):
            try:
                INDEX = faiss.read_index(FAISS_INDEX_PATH)
                logger.info(f"✅ Loaded FAISS index with {INDEX.ntotal} vectors")
                return INDEX
            except Exception as e:
                logger.warning(f"⚠️ Failed to load FAISS index: {e}")
        
        # Thử tải fallback vectors
        if os.path.exists(FALLBACK_VECTORS_PATH):
            try:
                data = np.load(FALLBACK_VECTORS_PATH)
                vectors = data['matrix']
                
                # Tạo index đơn giản
                class SimpleIndex:
                    def __init__(self, vectors):
                        self.vectors = vectors.astype(np.float32)
                        self.ntotal = vectors.shape[0]
                        
                        # Chuẩn hóa
                        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
                        self.vectors = self.vectors / (norms + 1e-12)
                    
                    def search(self, query_vector, k):
                        query_vector = query_vector.astype(np.float32)
                        query_norm = np.linalg.norm(query_vector)
                        if query_norm > 0:
                            query_vector = query_vector / query_norm
                        
                        similarities = np.dot(self.vectors, query_vector)
                        indices = np.argsort(-similarities)[:k]
                        distances = similarities[indices]
                        
                        return distances, indices
                
                INDEX = SimpleIndex(vectors)
                logger.info(f"✅ Loaded fallback index with {INDEX.ntotal} vectors")
                return INDEX
            except Exception as e:
                logger.warning(f"⚠️ Failed to load fallback vectors: {e}")
        
        # Tạo index rỗng
        logger.warning("⚠️ No index available, creating empty index")
        INDEX = None
        return None


# ---------- Search Functions ----------

def semantic_search(query: str, top_k: int = TOP_K, context_tour_index: Optional[int] = None) -> List[Tuple[float, Dict]]:
    """
    Tìm kiếm ngữ nghĩa trong index
    Trả về danh sách (score, passage) được sắp xếp
    """
    # Lấy embedding cho query
    query_embedding = get_text_embedding(query)
    
    # Tải index
    index = load_index()
    if index is None:
        return []
    
    # Thực hiện tìm kiếm
    try:
        distances, indices = index.search(query_embedding.reshape(1, -1), top_k * 3)  # Lấy nhiều hơn để filter
    except Exception as e:
        logger.error(f"❌ Search error: {e}")
        return []
    
    results = []
    query_norm = normalize_text(query)
    
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(MAPPING):
            continue
        
        passage = MAPPING[idx]
        
        # Tính điểm ngữ cảnh
        context_score = 1.0
        if context_tour_index is not None:
            if passage.get("tour_index") == context_tour_index:
                context_score = 2.0  # Ưu tiên cao cho tour hiện tại
            elif passage.get("tour_index") is not None:
                context_score = 0.5  # Giảm điểm cho tour khác
        
        # Tính điểm từ khóa
        passage_text = passage.get("text", "")
        passage_norm = normalize_text(passage_text)
        
        keyword_score = 0.0
        query_words = set(query_norm.split())
        passage_words = set(passage_norm.split())
        
        if query_words and passage_words:
            common = query_words & passage_words
            keyword_score = len(common) / len(query_words) * 0.5
        
        # Tổng điểm
        total_score = float(dist) * 0.6 + context_score * 0.3 + keyword_score * 0.1
        
        results.append((total_score, passage))
    
    # Sắp xếp và chỉ lấy top_k
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def field_specific_search(field: str, context_tour_index: Optional[int] = None) -> List[Tuple[float, Dict]]:
    """
    Tìm kiếm theo trường cụ thể
    Ưu tiên thông tin trong tour hiện tại
    """
    results = []
    
    for passage in MAPPING:
        passage_field = passage.get("field", "")
        
        # Kiểm tra xem có khớp field không
        field_match = False
        if field == passage_field:
            field_match = True
        elif field in FIELD_KEYWORDS and passage_field in FIELD_KEYWORDS[field]:
            field_match = True
        
        if not field_match:
            continue
        
        # Tính điểm
        score = 1.0
        
        # Ưu tiên tour hiện tại
        if context_tour_index is not None:
            if passage.get("tour_index") == context_tour_index:
                score = 3.0  # Rất cao cho đúng tour
            elif passage.get("tour_index") is not None:
                score = 0.5  # Thấp hơn cho tour khác
            else:
                score = 0.3  # Thấp nhất cho thông tin chung
        
        # Ưu tiên thông tin core
        if passage.get("is_core_info", False):
            score += 0.2
        
        results.append((score, passage))
    
    # Sắp xếp
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:TOP_K]


def hybrid_search(query: str, context_tour_index: Optional[int] = None) -> List[Tuple[float, Dict]]:
    """
    Tìm kiếm kết hợp: semantic + field-specific + context-aware
    """
    all_results = []
    
    # 1. Semantic search
    semantic_results = semantic_search(query, TOP_K, context_tour_index)
    all_results.extend(semantic_results)
    
    # 2. Detect field và field-specific search
    field, confidence = detect_field_from_query(query, context_tour_index)
    if field and confidence > 0.5:
        field_results = field_specific_search(field, context_tour_index)
        
        # Tăng điểm cho field-specific results
        boosted_results = []
        for score, passage in field_results:
            boosted_score = score * (1.0 + confidence * 0.5)
            boosted_results.append((boosted_score, passage))
        
        all_results.extend(boosted_results)
    
    # 3. Context-aware: thêm thông tin từ tour hiện tại
    if context_tour_index is not None:
        context_results = []
        for passage in MAPPING:
            if passage.get("tour_index") == context_tour_index:
                # Tính điểm dựa trên độ liên quan với query
                passage_text = passage.get("text", "")
                query_norm = normalize_text(query)
                passage_norm = normalize_text(passage_text)
                
                similarity = jaccard_similarity(
                    set(query_norm.split()),
                    set(passage_norm.split())
                )
                
                if similarity > 0.1:
                    score = 2.0 + similarity
                    context_results.append((score, passage))
        
        all_results.extend(context_results)
    
    # Loại bỏ trùng lặp và sắp xếp
    unique_results = {}
    for score, passage in all_results:
        passage_id = f"{passage.get('path', '')}:{passage.get('text', '')[:50]}"
        if passage_id not in unique_results or score > unique_results[passage_id][0]:
            unique_results[passage_id] = (score, passage)
    
    final_results = list(unique_results.values())
    final_results.sort(key=lambda x: x[0], reverse=True)
    
    return final_results[:TOP_K_CONTEXT]


# ---------- Session Management ----------

def create_session_id() -> str:
    """Tạo session ID mới"""
    return str(uuid.uuid4())


def get_session(session_id: Optional[str] = None) -> Tuple[str, Dict]:
    """Lấy hoặc tạo session"""
    if not session_id:
        session_id = request.cookies.get("session_id") if request else None
    
    # Redis session
    if REDIS_CLIENT is not None and session_id:
        try:
            key = f"session:{session_id}"
            data_json = REDIS_CLIENT.get(key)
            if data_json:
                data = json.loads(data_json)
                
                # Cập nhật thời gian
                data["last_activity"] = datetime.now().isoformat()
                REDIS_CLIENT.setex(key, SESSION_TIMEOUT, json.dumps(data))
                
                return session_id, data
        except Exception as e:
            logger.warning(f"⚠️ Redis session error: {e}")
    
    # Memory session hoặc tạo mới
    if not session_id or session_id not in USER_SESSIONS:
        session_id = create_session_id()
        USER_SESSIONS[session_id] = {
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "context_tour_index": None,
            "context_tour_name": None,
            "conversation_history": [],
            "query_count": 0
        }
    
    # Cập nhật thời gian
    USER_SESSIONS[session_id]["last_activity"] = datetime.now().isoformat()
    USER_SESSIONS[session_id]["query_count"] += 1
    
    return session_id, USER_SESSIONS[session_id]


def save_session(session_id: str, data: Dict):
    """Lưu session"""
    # Redis
    if REDIS_CLIENT is not None:
        try:
            key = f"session:{session_id}"
            REDIS_CLIENT.setex(key, SESSION_TIMEOUT, json.dumps(data))
            return
        except Exception as e:
            logger.warning(f"⚠️ Redis save error: {e}")
    
    # Memory
    USER_SESSIONS[session_id] = data


def update_session_context(session_data: Dict, query: str, tour_index: Optional[int] = None, tour_name: Optional[str] = None):
    """Cập nhật ngữ cảnh session"""
    
    # Lưu lịch sử hội thoại
    history = session_data.get("conversation_history", [])
    history.append({
        "query": query,
        "time": datetime.now().isoformat(),
        "tour_index": tour_index,
        "tour_name": tour_name
    })
    
    # Giữ chỉ 10 mục gần nhất
    if len(history) > 10:
        history = history[-10:]
    
    session_data["conversation_history"] = history
    
    # Cập nhật tour context nếu có
    if tour_index is not None:
        session_data["context_tour_index"] = tour_index
        session_data["context_tour_name"] = tour_name
        
        # Reset query count khi chuyển tour
        session_data["query_count"] = 1
    else:
        # Nếu không có tour mới, tăng query count
        session_data["query_count"] = session_data.get("query_count", 0) + 1
        
        # Nếu đã hỏi nhiều mà không nhắc đến tour, giảm context
        if session_data["query_count"] > CONTEXT_MEMORY:
            session_data["context_tour_index"] = None
            session_data["context_tour_name"] = None


# ---------- Response Generation ----------

def generate_deterministic_response(query: str, search_results: List[Tuple[float, Dict]], 
                                  context_tour_index: Optional[int] = None) -> str:
    """Tạo phản hồi xác định từ search results"""
    
    if not search_results:
        if context_tour_index:
            tour_name = TOUR_INDEX_TO_INFO.get(context_tour_index, {}).get("name", "tour này")
            return f"Hiện tôi chưa tìm thấy thông tin cụ thể về '{tour_name}' trong cơ sở dữ liệu. Bạn có thể hỏi về các tour khác hoặc liên hệ hotline 0332510486 để được tư vấn trực tiếp."
        else:
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp với câu hỏi của bạn. Vui lòng thử hỏi cụ thể hơn hoặc liên hệ Ruby Wings qua hotline 0332510486."
    
    # Nhóm kết quả theo tour
    results_by_tour = defaultdict(list)
    general_results = []
    
    for score, passage in search_results:
        tour_index = passage.get("tour_index")
        if tour_index is not None:
            results_by_tour[tour_index].append((score, passage))
        else:
            general_results.append((score, passage))
    
    # Xây dựng phản hồi
    response_parts = []
    
    # Ưu tiên tour trong context
    if context_tour_index and context_tour_index in results_by_tour:
        tour_name = TOUR_INDEX_TO_INFO.get(context_tour_index, {}).get("name", f"Tour #{context_tour_index}")
        response_parts.append(f"**Về tour '{tour_name}':**")
        
        for score, passage in results_by_tour[context_tour_index][:3]:
            text = passage.get("text", "")
            if text:
                response_parts.append(f"• {text}")
        
        # Xóa khỏi dict để không hiển thị lại
        del results_by_tour[context_tour_index]
    
    # Các tour khác
    for tour_index, tour_results in results_by_tour.items():
        if len(tour_results) > 0:
            tour_name = TOUR_INDEX_TO_INFO.get(tour_index, {}).get("name", f"Tour #{tour_index}")
            response_parts.append(f"\n**Tour '{tour_name}':**")
            
            for score, passage in tour_results[:2]:
                text = passage.get("text", "")
                if text:
                    response_parts.append(f"• {text}")
    
    # Thông tin chung
    if general_results and len(response_parts) < 3:
        response_parts.append("\n**Thông tin chung:**")
        for score, passage in general_results[:3]:
            text = passage.get("text", "")
            if text:
                response_parts.append(f"• {text}")
    
    # Nếu có nhiều tour, đề xuất chọn tour cụ thể
    if len(results_by_tour) > 1 and not context_tour_index:
        response_parts.append(f"\n💡 Tôi tìm thấy thông tin trong {len(results_by_tour)} tour. Vui lòng hỏi cụ thể về một tour để nhận thông tin chi tiết hơn.")
    
    response = "\n".join(response_parts)
    
    # Thêm thông tin liên hệ nếu cần
    if "hotline" not in response.lower() and "liên hệ" not in response.lower():
        response += "\n\n📞 Để biết thêm chi tiết hoặc đặt tour, vui lòng liên hệ Ruby Wings: 0332510486"
    
    return response


def generate_llm_response(query: str, search_results: List[Tuple[float, Dict]], 
                         context_tour_index: Optional[int] = None) -> str:
    """Tạo phản hồi sử dụng LLM (nếu có)"""
    
    if not client or not OPENAI_API_KEY:
        return generate_deterministic_response(query, search_results, context_tour_index)
    
    # Chuẩn bị context từ search results
    context_parts = []
    
    # Thêm thông tin tour context nếu có
    if context_tour_index:
        tour_info = get_tour_info(context_tour_index)
        if tour_info:
            context_parts.append(f"NGỮ CẢNH HIỆN TẠI: Người dùng đang hỏi về tour '{tour_info['name']}' (tour_index={context_tour_index})")
            context_parts.append("Hãy ưu tiên thông tin từ tour này trong câu trả lời.")
    
    # Thêm search results
    context_parts.append("\nTHÔNG TIN TÌM THẤY TỪ CƠ SỞ DỮ LIỆU:")
    
    added_passages = set()
    for score, passage in search_results[:8]:  # Giới hạn 8 passage
        passage_id = f"{passage.get('tour_index', 'general')}:{passage.get('text', '')[:50]}"
        
        if passage_id not in added_passages:
            tour_marker = ""
            if passage.get("tour_index") is not None:
                tour_name = TOUR_INDEX_TO_INFO.get(passage["tour_index"], {}).get("name", f"Tour #{passage['tour_index']}")
                tour_marker = f" [Tour: {tour_name}]"
            
            field_marker = f"[{passage.get('field', 'unknown')}]" if passage.get("field") else ""
            
            context_parts.append(f"\n{field_marker}{tour_marker} (Độ liên quan: {score:.2f}):")
            context_parts.append(passage.get("text", ""))
            added_passages.add(passage_id)
    
    context = "\n".join(context_parts)
    
    # System prompt
    system_prompt = f"""Bạn là trợ lý AI của Ruby Wings Travel - công ty chuyên tổ chức các tour du lịch trải nghiệm, retreat, thiền và hành trình chữa lành.

QUY TẮC TRẢ LỜI:
1. ƯU TIÊN CAO: Sử dụng thông tin từ NGỮ CẢNH HIỆN TẠI (nếu có) trước
2. Chỉ sử dụng thông tin từ cơ sở dữ liệu cung cấp bên dưới
3. KHÔNG bịa ra thông tin không có trong dữ liệu
4. Nếu không có thông tin, nói rõ "Tôi chưa tìm thấy thông tin về..."
5. Trả lời bằng tiếng Việt, tự nhiên, thân thiện
6. Giữ câu trả lời tập trung, không lan man
7. Nếu có thể, đề xuất hỏi thêm về các trường thông tin khác của tour

{context}

CÂU HỎI CỦA NGƯỜI DÙNG: {query}

Hãy trả lời dựa trên thông tin trên:"""
    
    try:
        response = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            max_tokens=800,
            top_p=0.9
        )
        
        return response.choices[0].message.content.strip()
    
    except Exception as e:
        logger.error(f"❌ LLM generation error: {e}")
        return generate_deterministic_response(query, search_results, context_tour_index)


# ---------- Main Chat Handler ----------

@app.route('/api/chat', methods=['POST'])
def chat_handler():
    """Xử lý chat request"""
    
    # Lấy session
    session_id, session_data = get_session()
    
    # Parse request
    try:
        data = request.get_json()
        query = data.get('message', '').strip()
        
        if not query:
            return jsonify({
                'reply': 'Vui lòng nhập câu hỏi của bạn.',
                'session_id': session_id,
                'context_tour': session_data.get('context_tour_name')
            })
    except Exception:
        return jsonify({
            'reply': 'Định dạng request không hợp lệ.',
            'session_id': session_id
        }), 400
    
    # 1. Phát hiện tour từ query
    tour_from_query = extract_tour_name_from_query(query)
    current_tour_index = None
    current_tour_name = None
    
    if tour_from_query:
        current_tour_name, current_tour_index = tour_from_query
    else:
        # Sử dụng tour từ context nếu có
        current_tour_index = session_data.get('context_tour_index')
        current_tour_name = session_data.get('context_tour_name')
    
    # 2. Tìm kiếm thông tin
    search_results = hybrid_search(query, current_tour_index)
    
    # 3. Tạo phản hồi
    try:
        reply = generate_llm_response(query, search_results, current_tour_index)
    except Exception as e:
        logger.error(f"❌ Response generation error: {e}")
        reply = generate_deterministic_response(query, search_results, current_tour_index)
    
    # 4. Cập nhật session context
    update_session_context(session_data, query, current_tour_index, current_tour_name)
    save_session(session_id, session_data)
    
    # 5. Chuẩn bị response
    response_data = {
        'reply': reply,
        'session_id': session_id,
        'context_tour': current_tour_name,
        'has_context': current_tour_index is not None,
        'sources_count': len(search_results)
    }
    
    # Thêm thông tin debug nếu cần
    if app.debug:
        response_data['debug'] = {
            'detected_tour_index': current_tour_index,
            'detected_tour_name': current_tour_name,
            'search_results_count': len(search_results),
            'session_query_count': session_data.get('query_count', 0)
        }
    
    # Set cookie
    response = jsonify(response_data)
    response.set_cookie(
        'session_id',
        session_id,
        max_age=SESSION_TIMEOUT,
        httponly=True,
        secure=False,  # Set True in production with HTTPS
        samesite='Lax'
    )
    
    return response


@app.route('/api/tours', methods=['GET'])
def list_tours():
    """API liệt kê tất cả tours"""
    
    tours = []
    for tour_index, tour_info in TOUR_INDEX_TO_INFO.items():
        tours.append({
            'id': tour_index,
            'name': tour_info['name'],
            'fields': list(tour_info.get('fields', [])),
            'has_full_info': True
        })
    
    return jsonify({
        'tours': tours,
        'count': len(tours)
    })


@app.route('/api/context', methods=['GET'])
def get_context():
    """API lấy ngữ cảnh hiện tại"""
    session_id = request.cookies.get('session_id')
    if not session_id:
        return jsonify({'context': None})
    
    _, session_data = get_session(session_id)
    
    return jsonify({
        'context_tour': session_data.get('context_tour_name'),
        'context_tour_index': session_data.get('context_tour_index'),
        'query_count': session_data.get('query_count', 0),
        'conversation_length': len(session_data.get('conversation_history', []))
    })


@app.route('/api/reset', methods=['POST'])
def reset_context():
    """API reset ngữ cảnh"""
    session_id = request.cookies.get('session_id')
    if not session_id:
        return jsonify({'success': False, 'message': 'No session'})
    
    _, session_data = get_session(session_id)
    
    # Reset context
    session_data['context_tour_index'] = None
    session_data['context_tour_name'] = None
    session_data['query_count'] = 0
    session_data['conversation_history'] = []
    
    save_session(session_id, session_data)
    
    return jsonify({'success': True, 'message': 'Context reset'})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'tours_count': len(TOUR_NAME_TO_INDEX),
        'mapping_count': len(MAPPING),
        'index_loaded': INDEX is not None,
        'openai_available': client is not None,
        'session_count': len(USER_SESSIONS)
    })


@app.route('/api/reindex', methods=['POST'])
def reindex():
    """Reindex endpoint (admin)"""
    # Kiểm tra auth đơn giản
    auth_key = request.headers.get('X-Admin-Key')
    if auth_key != os.environ.get('ADMIN_KEY', 'secret'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    # Reload data
    load_knowledge_data()
    load_mapping_data()
    load_field_keywords()
    
    # Clear index cache
    global INDEX
    INDEX = None
    load_index()
    
    return jsonify({
        'success': True,
        'message': 'Reindex completed',
        'tours_count': len(TOUR_NAME_TO_INDEX),
        'mapping_count': len(MAPPING)
    })


# ---------- Initialization ----------

def initialize_app():
    """Khởi tạo ứng dụng"""
    logger.info("🚀 Initializing Ruby Wings Chatbot...")
    
    # Tải dữ liệu
    load_knowledge_data()
    load_mapping_data()
    load_field_keywords()
    
    # Tải index
    load_index()
    
    # Log thông tin
    logger.info(f"✅ Knowledge: {len(KNOWLEDGE_DATA.get('tours', []))} tours")
    logger.info(f"✅ Mapping: {len(MAPPING)} passages")
    logger.info(f"✅ Field keywords: {len(FIELD_KEYWORDS)} fields")
    logger.info(f"✅ Tour index: {len(TOUR_NAME_TO_INDEX)} unique tours")
    logger.info(f"✅ OpenAI: {'Available' if client else 'Not available'}")
    logger.info(f"✅ FAISS: {'Available' if HAS_FAISS else 'Not available'}")
    logger.info(f"✅ Index: {'Loaded' if INDEX else 'Not loaded'}")
    
    logger.info("🎉 Ruby Wings Chatbot initialized successfully!")


# ---------- Cleanup ----------

@app.teardown_appcontext
def cleanup_session_store(exception=None):
    """Dọn dẹp session store khi ứng dụng shutdown"""
    # Đối với memory store, có thể clear sessions cũ
    current_time = datetime.now()
    expired_sessions = []
    
    for session_id, session_data in USER_SESSIONS.items():
        last_activity = session_data.get('last_activity')
        if last_activity:
            try:
                last_time = datetime.fromisoformat(last_activity)
                if (current_time - last_time).total_seconds() > SESSION_TIMEOUT:
                    expired_sessions.append(session_id)
            except ValueError:
                expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        USER_SESSIONS.pop(session_id, None)


# ---------- Main ----------

if __name__ == '__main__':
    # Khởi tạo
    initialize_app()
    
    # Chạy ứng dụng
    port = int(os.environ.get('PORT', 10000))
    debug = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"🌐 Starting server on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug, threaded=True)
else:
    # Khởi tạo khi chạy với WSGI
    initialize_app()