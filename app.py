#!/usr/bin/env python3
"""
app.py — Ruby Wings Chatbot với Ngữ Cảnh Tour Ưu Tiên và NLP Nâng Cao
Tối ưu cho Render với Python 3.8+ và package compatibility
"""

import os
import json
import re
import unicodedata
import threading
import logging
import uuid
import difflib
from datetime import datetime, timedelta
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Any
from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from collections import defaultdict

# ---------- Conditional Imports ----------
# Xử lý import linh hoạt cho các package optional

# Redis session store (optional)
try:
    import redis
    HAS_REDIS = True
except ImportError:
    redis = None
    HAS_REDIS = False

# FAISS vector search (optional)
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    faiss = None
    HAS_FAISS = False

# OpenAI SDK (optional)
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    OpenAI = None
    HAS_OPENAI = False

# NLP packages (optional)
try:
    from rapidfuzz import fuzz, process
    HAS_RAPIDFUZZ = True
except ImportError:
    fuzz = process = None
    HAS_RAPIDFUZZ = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except ImportError:
    TfidfVectorizer = cosine_similarity = None
    HAS_SKLEARN = False

try:
    import nltk
    from nltk.tokenize import word_tokenize
    HAS_NLTK = True
except ImportError:
    nltk = word_tokenize = None
    HAS_NLTK = False

try:
    import Levenshtein
    HAS_LEVENSHTEIN = True
except ImportError:
    Levenshtein = None
    HAS_LEVENSHTEIN = False

# ---------- Logging ----------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("ruby_wings_chatbot")

# ---------- Configuration ----------
# Lấy từ environment variables với giá trị mặc định
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
TOP_K_CONTEXT = int(os.environ.get("TOP_K_CONTEXT", "12"))
SESSION_TIMEOUT = int(os.environ.get("SESSION_TIMEOUT", "600"))  # 10 phút
SESSION_STORE = os.environ.get("SESSION_STORE", "memory")
REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
CONTEXT_MEMORY = int(os.environ.get("CONTEXT_MEMORY", "5"))
SIMILARITY_THRESHOLD = float(os.environ.get("SIMILARITY_THRESHOLD", "0.4"))
USE_TFIDF_FALLBACK = os.environ.get("USE_TFIDF_FALLBACK", "true").lower() in ("1", "true", "yes")
DEBUG = os.environ.get("DEBUG", "false").lower() in ("1", "true", "yes")

# ---------- Flask App ----------
app = Flask(__name__)
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# ---------- Global State ----------
KNOWLEDGE_DATA: Dict[str, Any] = {}
MAPPING: List[Dict[str, Any]] = []
METADATA: List[Dict[str, Any]] = []
VECTOR_INDEX = None
TFIDF_INDEX = None
INDEX_LOCK = threading.Lock()

# Field keywords và reverse mapping
FIELD_KEYWORDS: Dict[str, List[str]] = {}
REVERSE_KEYWORD_MAP: Dict[str, str] = {}

# Tour indices
TOUR_NAME_TO_INDEX: Dict[str, int] = {}
TOUR_INDEX_TO_INFO: Dict[int, Dict[str, Any]] = {}

# Session management
USER_SESSIONS: Dict[str, Dict[str, Any]] = {}
if SESSION_STORE == "redis" and HAS_REDIS:
    try:
        REDIS_CLIENT = redis.from_url(REDIS_URL)
        logger.info("✅ Redis session store initialized")
    except Exception as e:
        logger.warning(f"⚠️ Redis connection failed: {e}. Falling back to memory store.")
        REDIS_CLIENT = None
else:
    REDIS_CLIENT = None

# ---------- Text Processing Utilities ----------

def normalize_text(text: str) -> str:
    """
    Chuẩn hóa văn bản tiếng Việt
    - Lowercase
    - Loại bỏ dấu
    - Chuẩn hóa khoảng trắng
    - Giữ lại số và chữ
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Loại bỏ dấu tiếng Việt
    text = unicodedata.normalize('NFD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    # Thay thế các ký tự đặc biệt (giữ lại số và chữ)
    text = re.sub(r'[^\w\s\d]', ' ', text)
    
    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize văn bản thành các từ
    Sử dụng NLTK nếu có, fallback về split đơn giản
    """
    if HAS_NLTK:
        try:
            tokens = word_tokenize(text)
            return [token for token in tokens if token.isalnum()]
        except Exception:
            pass
    
    # Fallback: split đơn giản
    return [word for word in text.split() if word.isalnum()]


def extract_keywords(text: str) -> List[str]:
    """
    Trích xuất từ khóa từ văn bản
    Loại bỏ stopwords và từ ngắn
    """
    text_norm = normalize_text(text)
    tokens = tokenize_text(text_norm)
    
    # Stopwords tiếng Việt
    vietnamese_stopwords = {
        'có', 'và', 'hoặc', 'cho', 'về', 'từ', 'đến', 'ở', 'tại', 'là',
        'của', 'với', 'bằng', 'theo', 'khi', 'nào', 'gì', 'bao', 'nhiêu',
        'các', 'những', 'mấy', 'nhiều', 'ít', 'rất', 'quá', 'lắm', 'đã',
        'đang', 'sẽ', 'vẫn', 'cũng', 'đều', 'mọi', 'mỗi', 'từng', 'như',
        'nhưng', 'mà', 'nên', 'thì', 'làm', 'cần', 'phải', 'được', 'bị',
        'trong', 'ngoài', 'trên', 'dưới', 'trước', 'sau', 'giữa', 'bên'
    }
    
    # Lọc stopwords và từ ngắn
    keywords = []
    for token in tokens:
        if (len(token) > 1 and 
            token not in vietnamese_stopwords and
            not token.isdigit()):
            keywords.append(token)
    
    # Thêm bigram cho các từ liên tiếp
    if len(tokens) >= 2:
        for i in range(len(tokens) - 1):
            if len(tokens[i]) > 1 and len(tokens[i+1]) > 1:
                bigram = f"{tokens[i]}_{tokens[i+1]}"
                keywords.append(bigram)
    
    return list(set(keywords))  # Remove duplicates


def calculate_similarity(text1: str, text2: str) -> float:
    """
    Tính độ tương đồng giữa hai văn bản
    Sử dụng multiple methods và kết hợp kết quả
    """
    if not text1 or not text2:
        return 0.0
    
    text1_norm = normalize_text(text1)
    text2_norm = normalize_text(text2)
    
    scores = []
    
    # 1. RapidFuzz similarity (nếu có)
    if HAS_RAPIDFUZZ:
        try:
            # Weighted Ratio - tốt cho tiếng Việt
            score = fuzz.WRatio(text1_norm, text2_norm) / 100.0
            scores.append(score)
            
            # Token Sort Ratio (không quan tâm thứ tự từ)
            token_score = fuzz.token_sort_ratio(text1_norm, text2_norm) / 100.0
            scores.append(token_score * 0.8)
        except Exception:
            pass
    
    # 2. SequenceMatcher (fallback built-in)
    seq_score = difflib.SequenceMatcher(None, text1_norm, text2_norm).ratio()
    scores.append(seq_score * 0.7)
    
    # 3. Jaccard similarity trên keywords
    keywords1 = extract_keywords(text1)
    keywords2 = extract_keywords(text2)
    
    if keywords1 and keywords2:
        set1 = set(keywords1)
        set2 = set(keywords2)
        union = set1 | set2
        if union:
            jaccard = len(set1 & set2) / len(union)
            scores.append(jaccard * 0.6)
    
    # 4. Levenshtein distance (nếu có package)
    if HAS_LEVENSHTEIN:
        try:
            max_len = max(len(text1_norm), len(text2_norm))
            if max_len > 0:
                lev_dist = Levenshtein.distance(text1_norm, text2_norm)
                lev_score = 1 - (lev_dist / max_len)
                scores.append(lev_score * 0.5)
        except Exception:
            pass
    
    # Trả về điểm trung bình
    return sum(scores) / len(scores) if scores else 0.0


# ---------- TF-IDF Index (Fallback khi không có embeddings) ----------

class TFIDFIndex:
    """TF-IDF index cho text search fallback"""
    
    def __init__(self):
        self.vectorizer = None
        self.tfidf_matrix = None
        self.documents = []
        self.is_built = False
    
    def build(self, documents: List[Dict[str, Any]]) -> bool:
        """Xây dựng TF-IDF index từ documents"""
        if not HAS_SKLEARN or not documents:
            return False
        
        try:
            texts = []
            self.documents = []
            
            for doc in documents:
                text = doc.get("text", "")
                if text and len(text.strip()) > 10:  # Chỉ lấy text đủ dài
                    texts.append(text)
                    self.documents.append(doc)
            
            if len(texts) < 5:
                logger.warning("⚠️ Not enough documents for TF-IDF index")
                return False
            
            # Tạo vectorizer với các tham số tối ưu cho tiếng Việt
            self.vectorizer = TfidfVectorizer(
                max_features=2000,
                min_df=2,
                max_df=0.85,
                ngram_range=(1, 2),
                stop_words=None,
                token_pattern=r'\b\w+\b'
            )
            
            self.tfidf_matrix = self.vectorizer.fit_transform(texts)
            self.is_built = True
            
            logger.info(f"✅ TF-IDF index built with {len(texts)} documents")
            return True
            
        except Exception as e:
            logger.error(f"❌ TF-IDF build error: {e}")
            return False
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Tìm kiếm với TF-IDF"""
        if not self.is_built or not self.vectorizer or not self.tfidf_matrix:
            return []
        
        try:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
            
            # Lấy top_k kết quả
            top_indices = similarities.argsort()[::-1][:top_k]
            
            results = []
            for idx in top_indices:
                if idx < len(self.documents):
                    score = float(similarities[idx])
                    if score > 0.1:  # Ngưỡng tối thiểu
                        results.append((score, self.documents[idx]))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ TF-IDF search error: {e}")
            return []


# ---------- Field Keywords Management ----------

def load_field_keywords():
    """Tải từ khóa trường dữ liệu từ file"""
    global FIELD_KEYWORDS, REVERSE_KEYWORD_MAP
    
    FIELD_KEYWORDS = {}
    REVERSE_KEYWORD_MAP = {}
    
    if not os.path.exists(FIELD_KEYWORDS_PATH):
        logger.warning(f"⚠️ Field keywords file not found: {FIELD_KEYWORDS_PATH}")
        create_default_field_keywords()
        return
    
    try:
        with open(FIELD_KEYWORDS_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for field, keywords in data.items():
            if field.startswith("__"):  # Bỏ qua metadata
                continue
            
            # Chuẩn hóa field name
            if '.' in field:
                norm_field = field  # Giữ nguyên cho nested fields
            else:
                norm_field = field
            
            # Chuẩn hóa keywords
            norm_keywords = [normalize_text(kw) for kw in keywords]
            FIELD_KEYWORDS[norm_field] = norm_keywords
            
            # Tạo reverse mapping
            for keyword in norm_keywords:
                REVERSE_KEYWORD_MAP[keyword] = norm_field
        
        logger.info(f"✅ Loaded {len(FIELD_KEYWORDS)} field keyword groups")
        
    except Exception as e:
        logger.error(f"❌ Failed to load field keywords: {e}")
        create_default_field_keywords()


def create_default_field_keywords():
    """Tạo từ khóa mặc định nếu không có file"""
    global FIELD_KEYWORDS, REVERSE_KEYWORD_MAP
    
    default_keywords = {
        "tour_name": ["tour này tên gì", "tour gì", "tên tour", "tour nào", "hành trình gì", "chương trình gì"],
        "summary": ["tóm tắt", "giới thiệu", "mô tả", "overview", "tổng quan", "nội dung chính"],
        "location": ["đi đâu", "địa điểm", "điểm đến", "location", "khu vực", "vùng miền"],
        "duration": ["thời gian", "bao lâu", "mấy ngày", "duration", "kéo dài", "thời lượng"],
        "price": ["giá", "chi phí", "bao nhiêu tiền", "price", "giá tour", "mức giá"],
        "includes": ["bao gồm", "gồm những gì", "nội dung", "includes", "hoạt động", "điểm tham quan"],
        "notes": ["lưu ý", "chú ý", "notes", "cần biết", "yêu cầu", "chuẩn bị"],
        "style": ["phong cách", "style", "concept", "định hướng", "loại hình", "hình thức"],
        "transport": ["phương tiện", "xe", "di chuyển", "transport", "vận chuyển", "đi lại"],
        "accommodation": ["ở đâu", "lưu trú", "khách sạn", "homestay", "accommodation", "chỗ nghỉ"],
        "meals": ["ăn uống", "bữa ăn", "ẩm thực", "meals", "đồ ăn", "thức ăn"],
        "event_support": ["hỗ trợ", "support", "dịch vụ", "event support", "chăm sóc", "hỗ trợ đoàn"],
        "hotline": ["hotline", "số điện thoại", "liên hệ", "contact", "sdt", "gọi điện"],
        "mission": ["sứ mệnh", "mission", "mục tiêu", "giá trị", "ý nghĩa", "tầm nhìn"],
        "includes_extra": ["thêm gì", "extra", "bổ sung", "tùy chọn thêm", "dịch vụ thêm"],
        "extras": ["không bao gồm", "ngoài giá", "phụ phí", "extras", "tự túc", "tự chi trả"],
        "additional": ["phụ thu", "extra fee", "chi phí thêm", "phát sinh", "nâng cấp"],
        "about_company.overview": ["giới thiệu công ty", "ruby wings là gì", "về ruby wings", "công ty làm gì"],
        "about_company.mission": ["sứ mệnh công ty", "mission ruby wings", "tầm nhìn công ty", "giá trị cốt lõi"],
        "faq.cancellation_policy": ["chính sách hủy", "hủy tour", "refund", "hoàn tiền", "hủy đặt"],
        "faq.booking_method": ["đặt tour", "cách đặt", "book tour", "đăng ký", "đặt chỗ"],
        "faq.who_can_join": ["ai tham gia", "đối tượng", "phù hợp với ai", "trẻ em có đi được không"],
        "contact.hotline": ["hotline công ty", "số điện thoại công ty", "liên hệ công ty", "tổng đài"],
        "contact.email": ["email công ty", "gửi mail", "email liên hệ", "mail công ty"],
        "contact.office_hours": ["giờ làm việc", "thời gian tư vấn", "mở cửa lúc nào", "giờ hành chính"]
    }
    
    FIELD_KEYWORDS = default_keywords
    REVERSE_KEYWORD_MAP = {}
    
    for field, keywords in default_keywords.items():
        for keyword in keywords:
            norm_keyword = normalize_text(keyword)
            REVERSE_KEYWORD_MAP[norm_keyword] = field
    
    logger.info("✅ Created default field keywords")


def detect_field_from_query(query: str) -> Tuple[Optional[str], float]:
    """
    Phát hiện trường dữ liệu từ câu hỏi
    Trả về (field_name, confidence_score)
    """
    query_norm = normalize_text(query)
    
    best_field = None
    best_score = 0.0
    
    # Tìm kiếm trong reverse keyword map
    for keyword, field in REVERSE_KEYWORD_MAP.items():
        if keyword in query_norm:
            # Tính điểm dựa trên độ dài keyword
            score = min(len(keyword.split()), 3) * 0.2
            
            # Ưu tiên exact match
            if f" {keyword} " in f" {query_norm} ":
                score += 0.3
            
            if score > best_score:
                best_score = score
                best_field = field
    
    # Nếu không tìm thấy, sử dụng heuristic
    if best_score < 0.3:
        query_lower = query.lower()
        
        # Heuristic cho các trường phổ biến
        heuristics = [
            (["giá", "bao nhiêu tiền", "chi phí"], "price", 0.7),
            (["thời gian", "mấy ngày", "bao lâu"], "duration", 0.6),
            (["đi đâu", "ở đâu", "địa điểm"], "location", 0.6),
            (["bao gồm", "gồm những gì"], "includes", 0.5),
            (["ăn uống", "bữa ăn", "ẩm thực"], "meals", 0.5),
            (["phương tiện", "xe", "di chuyển"], "transport", 0.5),
            (["hotline", "số điện thoại", "liên hệ"], "hotline", 0.8),
            (["lưu ý", "cần biết", "chuẩn bị"], "notes", 0.5),
        ]
        
        for keywords, field, base_score in heuristics:
            for keyword in keywords:
                if keyword in query_lower:
                    if base_score > best_score:
                        best_score = base_score
                        best_field = field
                    break
    
    return best_field, best_score


# ---------- Tour Detection ----------

def load_tour_indices():
    """Xây dựng index cho tour từ mapping data"""
    global TOUR_NAME_TO_INDEX, TOUR_INDEX_TO_INFO
    
    TOUR_NAME_TO_INDEX.clear()
    TOUR_INDEX_TO_INFO.clear()
    
    for passage in MAPPING:
        tour_index = passage.get("tour_index")
        tour_name = passage.get("tour_name")
        
        if tour_index is not None and tour_name:
            # Lưu mapping từ tên tour chuẩn hóa đến index
            tour_name_norm = normalize_text(tour_name)
            if tour_name_norm not in TOUR_NAME_TO_INDEX:
                TOUR_NAME_TO_INDEX[tour_name_norm] = tour_index
            
            # Lưu thông tin tour
            if tour_index not in TOUR_INDEX_TO_INFO:
                TOUR_INDEX_TO_INFO[tour_index] = {
                    "name": tour_name,
                    "name_norm": tour_name_norm,
                    "fields": set()
                }
            
            # Thêm field vào set
            field = passage.get("field")
            if field:
                TOUR_INDEX_TO_INFO[tour_index]["fields"].add(field)
    
    logger.info(f"✅ Indexed {len(TOUR_NAME_TO_INDEX)} unique tours")


def extract_tour_from_query(query: str) -> Optional[Tuple[str, int, float]]:
    """
    Trích xuất tour từ câu hỏi với fuzzy matching
    Trả về (tour_name, tour_index, confidence)
    """
    if not TOUR_NAME_TO_INDEX:
        return None
    
    query_norm = normalize_text(query)
    
    # Sử dụng RapidFuzz nếu có
    if HAS_RAPIDFUZZ:
        try:
            # Tìm tour khớp nhất
            best_match = process.extractOne(
                query_norm,
                list(TOUR_NAME_TO_INDEX.keys()),
                scorer=fuzz.WRatio,
                score_cutoff=50  # Ngưỡng 50%
            )
            
            if best_match:
                tour_name_norm, score, _ = best_match
                tour_index = TOUR_NAME_TO_INDEX[tour_name_norm]
                tour_info = TOUR_INDEX_TO_INFO.get(tour_index)
                
                if tour_info:
                    confidence = score / 100.0
                    return tour_info["name"], tour_index, confidence
        except Exception as e:
            logger.warning(f"⚠️ RapidFuzz search error: {e}")
    
    # Fallback: tìm kiếm đơn giản
    best_match = None
    best_score = 0.0
    
    for tour_name_norm, tour_index in TOUR_NAME_TO_INDEX.items():
        tour_info = TOUR_INDEX_TO_INFO.get(tour_index)
        if not tour_info:
            continue
        
        # Tính similarity
        similarity = calculate_similarity(query, tour_info["name"])
        
        # Bonus nếu có từ khóa chung
        query_keywords = extract_keywords(query)
        tour_keywords = extract_keywords(tour_info["name"])
        common_keywords = set(query_keywords) & set(tour_keywords)
        if common_keywords:
            similarity += len(common_keywords) * 0.1
        
        if similarity > best_score and similarity > SIMILARITY_THRESHOLD:
            best_score = similarity
            best_match = (tour_info["name"], tour_index, similarity)
    
    return best_match


# ---------- Embedding & Vector Search ----------

@lru_cache(maxsize=1024)
def get_text_embedding(text: str) -> np.ndarray:
    """
    Lấy embedding cho văn bản
    Sử dụng OpenAI nếu có, fallback synthetic
    """
    if not text or not text.strip():
        return np.zeros(1536, dtype=np.float32)
    
    # Cắt ngắn nếu quá dài
    text = text[:4000]
    
    # Sử dụng OpenAI embedding nếu có
    if HAS_OPENAI and OPENAI_API_KEY:
        try:
            client = OpenAI(api_key=OPENAI_API_KEY)
            response = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=text
            )
            embedding = np.array(response.data[0].embedding, dtype=np.float32)
            return embedding
        except Exception as e:
            logger.warning(f"⚠️ OpenAI embedding failed: {e}")
    
    # Fallback: tạo embedding tổng hợp
    return generate_synthetic_embedding(text)


def generate_synthetic_embedding(text: str) -> np.ndarray:
    """Tạo embedding tổng hợp cho fallback"""
    text_norm = normalize_text(text)
    words = text_norm.split()[:100]  # Giới hạn 100 từ
    
    vector = np.zeros(1536, dtype=np.float32)
    
    for i, word in enumerate(words):
        # Tạo hash deterministic từ word
        word_hash = hash(word) % 10000
        
        # Phân bố vào vector
        for j in range(10):
            idx = (word_hash + i * j) % 1536
            vector[idx] += (i + 1) * 0.001
    
    # Chuẩn hóa
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector


def load_vector_index():
    """Tải FAISS index hoặc tạo fallback"""
    global VECTOR_INDEX
    
    with INDEX_LOCK:
        if VECTOR_INDEX is not None:
            return VECTOR_INDEX
        
        # Thử tải FAISS index
        if HAS_FAISS and FAISS_ENABLED and os.path.exists(FAISS_INDEX_PATH):
            try:
                VECTOR_INDEX = faiss.read_index(FAISS_INDEX_PATH)
                logger.info(f"✅ Loaded FAISS index with {VECTOR_INDEX.ntotal} vectors")
                return VECTOR_INDEX
            except Exception as e:
                logger.warning(f"⚠️ Failed to load FAISS index: {e}")
        
        # Thử tải fallback vectors
        if os.path.exists(FALLBACK_VECTORS_PATH):
            try:
                data = np.load(FALLBACK_VECTORS_PATH)
                
                if 'matrix' in data:
                    vectors = data['matrix']
                elif 'mat' in data:
                    vectors = data['mat']
                else:
                    logger.error("❌ Unknown format in vectors file")
                    return None
                
                # Tạo SimpleIndex
                class SimpleIndex:
                    def __init__(self, vectors):
                        self.vectors = vectors.astype(np.float32)
                        self.ntotal = vectors.shape[0]
                        
                        # Chuẩn hóa vectors
                        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True)
                        self.vectors = self.vectors / (norms + 1e-12)
                    
                    def search(self, query_vector, k):
                        query_vector = query_vector.astype(np.float32).reshape(1, -1)
                        query_norm = np.linalg.norm(query_vector)
                        if query_norm > 0:
                            query_vector = query_vector / query_norm
                        
                        similarities = np.dot(self.vectors, query_vector.T).flatten()
                        indices = np.argsort(-similarities)[:k]
                        distances = similarities[indices]
                        
                        return distances, indices
                
                VECTOR_INDEX = SimpleIndex(vectors)
                logger.info(f"✅ Loaded fallback index with {VECTOR_INDEX.ntotal} vectors")
                return VECTOR_INDEX
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load fallback vectors: {e}")
        
        logger.warning("⚠️ No vector index available")
        return None


# ---------- Search Functions ----------

def semantic_search(query: str, top_k: int = TOP_K, 
                   context_tour_index: Optional[int] = None) -> List[Tuple[float, Dict]]:
    """Tìm kiếm ngữ nghĩa với embeddings"""
    
    # Lấy embedding cho query
    query_embedding = get_text_embedding(query)
    
    # Tải index
    index = load_vector_index()
    if index is None:
        return []
    
    # Thực hiện tìm kiếm
    try:
        distances, indices = index.search(query_embedding.reshape(1, -1), top_k * 2)
    except Exception as e:
        logger.error(f"❌ Vector search error: {e}")
        return []
    
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(MAPPING):
            continue
        
        passage = MAPPING[idx]
        
        # Tính điểm ngữ cảnh
        score = float(dist)
        
        # Ưu tiên tour context
        if context_tour_index is not None:
            passage_tour_index = passage.get("tour_index")
            if passage_tour_index == context_tour_index:
                score *= 1.5  # Tăng điểm cho tour hiện tại
            elif passage_tour_index is not None:
                score *= 0.7  # Giảm điểm cho tour khác
        
        results.append((score, passage))
    
    # Sắp xếp
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def keyword_search(query: str, top_k: int = 5) -> List[Tuple[float, Dict]]:
    """Tìm kiếm theo từ khóa đơn giản"""
    query_keywords = extract_keywords(query)
    
    if not query_keywords:
        return []
    
    results = []
    
    for passage in MAPPING:
        passage_text = passage.get("text", "")
        passage_keywords = extract_keywords(passage_text)
        
        # Tính điểm Jaccard similarity
        if query_keywords and passage_keywords:
            common = set(query_keywords) & set(passage_keywords)
            if common:
                score = len(common) / len(query_keywords)
                
                # Thêm bonus cho exact match
                if HAS_RAPIDFUZZ:
                    fuzz_score = fuzz.partial_ratio(query, passage_text) / 100.0
                    score = score * 0.7 + fuzz_score * 0.3
                
                results.append((score, passage))
    
    results.sort(key=lambda x: x[0], reverse=True)
    return results[:top_k]


def hybrid_search(query: str, context_tour_index: Optional[int] = None) -> List[Tuple[float, Dict]]:
    """
    Tìm kiếm lai: semantic + keyword + field-specific
    """
    all_results = []
    
    # 1. Semantic search
    semantic_results = semantic_search(query, TOP_K, context_tour_index)
    all_results.extend(semantic_results)
    
    # 2. TF-IDF search (nếu có)
    if TFIDF_INDEX and USE_TFIDF_FALLBACK:
        tfidf_results = TFIDF_INDEX.search(query, TOP_K)
        # Giảm điểm TF-IDF để ưu tiên embeddings
        tfidf_results = [(score * 0.6, passage) for score, passage in tfidf_results]
        all_results.extend(tfidf_results)
    
    # 3. Field-specific search
    field, confidence = detect_field_from_query(query)
    if field and confidence > 0.4:
        # Tìm passages với field cụ thể
        field_passages = []
        for passage in MAPPING:
            if passage.get("field") == field:
                score = 1.0
                
                # Ưu tiên tour context
                if context_tour_index is not None:
                    if passage.get("tour_index") == context_tour_index:
                        score = 2.0
                    elif passage.get("tour_index") is not None:
                        score = 0.5
                
                field_passages.append((score, passage))
        
        # Sắp xếp và giới hạn
        field_passages.sort(key=lambda x: x[0], reverse=True)
        all_results.extend(field_passages[:TOP_K//2])
    
    # 4. Keyword search (fallback)
    if len(all_results) < 3:
        keyword_results = keyword_search(query, TOP_K)
        all_results.extend(keyword_results)
    
    # 5. Context-aware: thêm thông tin từ tour hiện tại
    if context_tour_index is not None:
        context_results = []
        for passage in MAPPING:
            if passage.get("tour_index") == context_tour_index:
                # Tính similarity với query
                passage_text = passage.get("text", "")
                similarity = calculate_similarity(query, passage_text)
                
                if similarity > 0.2:
                    score = 1.5 + similarity
                    context_results.append((score, passage))
        
        all_results.extend(context_results)
    
    # Loại bỏ trùng lặp
    unique_results = {}
    for score, passage in all_results:
        passage_id = passage.get("path", "") + ":" + passage.get("text", "")[:30]
        if passage_id not in unique_results or score > unique_results[passage_id][0]:
            unique_results[passage_id] = (score, passage)
    
    # Sắp xếp và trả về
    final_results = list(unique_results.values())
    final_results.sort(key=lambda x: x[0], reverse=True)
    
    return final_results[:TOP_K_CONTEXT]


# ---------- Session Management ----------

def create_session_id() -> str:
    """Tạo session ID mới"""
    return str(uuid.uuid4())


def get_session(session_id: Optional[str] = None) -> Tuple[str, Dict]:
    """Lấy hoặc tạo session"""
    
    # Lấy session_id từ cookie nếu không có
    if not session_id and request:
        session_id = request.cookies.get("session_id")
    
    # Redis session
    if REDIS_CLIENT and session_id:
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
            "query_count": 0,
            "conversation": []
        }
    
    # Cập nhật thời gian
    data = USER_SESSIONS[session_id]
    data["last_activity"] = datetime.now().isoformat()
    data["query_count"] = data.get("query_count", 0) + 1
    
    return session_id, data


def save_session(session_id: str, data: Dict):
    """Lưu session"""
    
    # Redis
    if REDIS_CLIENT:
        try:
            key = f"session:{session_id}"
            REDIS_CLIENT.setex(key, SESSION_TIMEOUT, json.dumps(data))
            return
        except Exception as e:
            logger.warning(f"⚠️ Redis save error: {e}")
    
    # Memory
    USER_SESSIONS[session_id] = data


def update_session_context(session_data: Dict, query: str, 
                          tour_index: Optional[int] = None, 
                          tour_name: Optional[str] = None):
    """Cập nhật ngữ cảnh session"""
    
    # Cập nhật tour context
    if tour_index is not None:
        session_data["context_tour_index"] = tour_index
        session_data["context_tour_name"] = tour_name
        session_data["query_count"] = 1  # Reset khi chuyển tour
    else:
        # Tăng query count
        session_data["query_count"] = session_data.get("query_count", 0) + 1
        
        # Nếu đã hỏi nhiều mà không nhắc đến tour, clear context
        if session_data["query_count"] > CONTEXT_MEMORY:
            session_data["context_tour_index"] = None
            session_data["context_tour_name"] = None
    
    # Lưu lịch sử hội thoại
    conversation = session_data.get("conversation", [])
    conversation.append({
        "query": query,
        "time": datetime.now().isoformat(),
        "tour_index": tour_index,
        "tour_name": tour_name
    })
    
    # Giữ chỉ 10 mục gần nhất
    if len(conversation) > 10:
        conversation = conversation[-10:]
    
    session_data["conversation"] = conversation


# ---------- Response Generation ----------

def generate_deterministic_response(query: str, 
                                   search_results: List[Tuple[float, Dict]], 
                                   context_tour_index: Optional[int] = None) -> str:
    """Tạo phản hồi xác định từ search results"""
    
    if not search_results:
        if context_tour_index:
            tour_name = TOUR_INDEX_TO_INFO.get(context_tour_index, {}).get("name", "tour này")
            return f"Hiện tôi chưa tìm thấy thông tin cụ thể về '{tour_name}' trong cơ sở dữ liệu. Bạn có thể hỏi về các trường khác như giá, thời gian, địa điểm, hoặc liên hệ hotline 0332510486 để được tư vấn trực tiếp."
        else:
            return "Xin lỗi, tôi chưa tìm thấy thông tin phù hợp với câu hỏi của bạn. Vui lòng thử hỏi cụ thể hơn (ví dụ: tên tour, giá tour, thời gian, địa điểm) hoặc liên hệ Ruby Wings qua hotline 0332510486."
    
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
        
        # Lấy các kết quả có điểm cao nhất
        top_results = sorted(results_by_tour[context_tour_index], key=lambda x: x[0], reverse=True)[:3]
        
        for score, passage in top_results:
            text = passage.get("text", "")
            if text:
                response_parts.append(f"• {text}")
        
        # Xóa để không hiển thị lại
        del results_by_tour[context_tour_index]
    
    # Các tour khác
    for tour_index, tour_results in results_by_tour.items():
        if tour_results:
            tour_name = TOUR_INDEX_TO_INFO.get(tour_index, {}).get("name", f"Tour #{tour_index}")
            response_parts.append(f"\n**Tour '{tour_name}':**")
            
            top_results = sorted(tour_results, key=lambda x: x[0], reverse=True)[:2]
            for score, passage in top_results:
                text = passage.get("text", "")
                if text:
                    response_parts.append(f"• {text}")
    
    # Thông tin chung
    if general_results and len(response_parts) < 3:
        response_parts.append("\n**Thông tin chung:**")
        top_general = sorted(general_results, key=lambda x: x[0], reverse=True)[:3]
        for score, passage in top_general:
            text = passage.get("text", "")
            if text:
                response_parts.append(f"• {text}")
    
    # Nếu có nhiều tour, đề xuất chọn tour cụ thể
    if len(results_by_tour) > 1 and not context_tour_index:
        response_parts.append(f"\n💡 Tôi tìm thấy thông tin trong {len(results_by_tour)} tour. Vui lòng hỏi cụ thể về một tour để nhận thông tin chi tiết hơn.")
    
    response = "\n".join(response_parts)
    
    # Thêm thông tin liên hệ nếu cần
    if "hotline" not in response.lower() and "liên hệ" not in response.lower():
        response += "\n\n📞 Để biết thêm chi tiết hoặc đặt tour, vui lòng liên hệ Ruby Wings: **0332510486**"
    
    return response


def generate_llm_response(query: str, 
                          search_results: List[Tuple[float, Dict]], 
                          context_tour_index: Optional[int] = None) -> str:
    """Tạo phản hồi sử dụng LLM (nếu có)"""
    
    # Kiểm tra OpenAI availability
    if not HAS_OPENAI or not OPENAI_API_KEY:
        return generate_deterministic_response(query, search_results, context_tour_index)
    
    # Chuẩn bị context
    context_parts = []
    
    # Thêm thông tin tour context
    if context_tour_index:
        tour_info = TOUR_INDEX_TO_INFO.get(context_tour_index)
        if tour_info:
            context_parts.append(f"NGỮ CẢNH: Người dùng đang hỏi về tour '{tour_info['name']}'")
            context_parts.append("Hãy ưu tiên thông tin từ tour này trong câu trả lời.\n")
    
    # Thêm search results
    context_parts.append("THÔNG TIN TỪ CƠ SỞ DỮ LIỆU RUBY WINGS:")
    
    added_passages = set()
    for i, (score, passage) in enumerate(search_results[:6], 1):
        passage_text = passage.get("text", "")
        if not passage_text:
            continue
        
        passage_id = hash(passage_text[:100])
        if passage_id in added_passages:
            continue
        
        added_passages.add(passage_id)
        
        # Thêm metadata
        tour_marker = ""
        tour_index = passage.get("tour_index")
        if tour_index is not None:
            tour_name = TOUR_INDEX_TO_INFO.get(tour_index, {}).get("name", f"Tour #{tour_index}")
            tour_marker = f" [Tour: {tour_name}]"
        
        field_marker = f"[{passage.get('field', 'unknown')}]"
        
        context_parts.append(f"\n{i}. {field_marker}{tour_marker}:")
        context_parts.append(passage_text)
    
    context = "\n".join(context_parts)
    
    # System prompt
    system_prompt = f"""Bạn là trợ lý AI của Ruby Wings Travel - công ty chuyên tổ chức các tour du lịch trải nghiệm, retreat, thiền và hành trình chữa lành.

{context}

QUY TẮC TRẢ LỜI:
1. Chỉ sử dụng thông tin từ cơ sở dữ liệu trên
2. KHÔNG tạo ra thông tin không có trong dữ liệu
3. Nếu không có thông tin, nói rõ "Tôi chưa tìm thấy thông tin về..."
4. Trả lời bằng tiếng Việt, tự nhiên, thân thiện
5. Giữ câu trả lời tập trung, không lan man
6. Nếu có thể, đề xuất hỏi thêm về các trường thông tin khác

Câu hỏi: {query}

Trả lời:"""
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
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


# ---------- API Routes ----------

@app.route('/api/chat', methods=['POST'])
def chat_handler():
    """Xử lý chat request"""
    
    # Lấy session
    session_id, session_data = get_session()
    
    # Parse request
    try:
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'Invalid request format',
                'reply': 'Vui lòng gửi yêu cầu dưới dạng JSON.'
            }), 400
        
        query = data.get('message', '').strip()
        if not query:
            return jsonify({
                'reply': 'Vui lòng nhập câu hỏi của bạn.',
                'session_id': session_id
            })
            
    except Exception as e:
        logger.error(f"❌ Request parsing error: {e}")
        return jsonify({
            'reply': 'Định dạng request không hợp lệ.',
            'session_id': session_id
        }), 400
    
    # 1. Phát hiện tour từ query
    tour_match = extract_tour_from_query(query)
    current_tour_index = None
    current_tour_name = None
    
    if tour_match:
        current_tour_name, current_tour_index, confidence = tour_match
        logger.info(f"🔍 Detected tour: {current_tour_name} (index={current_tour_index}, confidence={confidence:.2f})")
    else:
        # Sử dụng tour từ context nếu có
        current_tour_index = session_data.get('context_tour_index')
        current_tour_name = session_data.get('context_tour_name')
        if current_tour_index:
            logger.info(f"🔍 Using context tour: {current_tour_name} (index={current_tour_index})")
    
    # 2. Tìm kiếm thông tin
    search_results = hybrid_search(query, current_tour_index)
    logger.info(f"🔍 Found {len(search_results)} relevant passages")
    
    # 3. Tạo phản hồi
    try:
        # Sử dụng LLM nếu có OpenAI, nếu không dùng deterministic
        if HAS_OPENAI and OPENAI_API_KEY:
            reply = generate_llm_response(query, search_results, current_tour_index)
        else:
            reply = generate_deterministic_response(query, search_results, current_tour_index)
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
    
    # Thêm debug info nếu enabled
    if DEBUG:
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
            'has_info': True
        })
    
    return jsonify({
        'tours': sorted(tours, key=lambda x: x['id']),
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
        'conversation_length': len(session_data.get('conversation', []))
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
    session_data['conversation'] = []
    
    save_session(session_id, session_data)
    
    return jsonify({'success': True, 'message': 'Context reset'})


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    health_status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'service': 'ruby_wings_chatbot',
        'version': '2.0.0',
        'components': {
            'knowledge_data': len(KNOWLEDGE_DATA) > 0,
            'mapping': len(MAPPING) > 0,
            'tours_indexed': len(TOUR_NAME_TO_INDEX),
            'vector_index': VECTOR_INDEX is not None,
            'tfidf_index': TFIDF_INDEX is not None if TFIDF_INDEX else False,
            'openai': HAS_OPENAI and bool(OPENAI_API_KEY),
            'redis': REDIS_CLIENT is not None,
            'rapidfuzz': HAS_RAPIDFUZZ,
            'sklearn': HAS_SKLEARN,
            'nltk': HAS_NLTK
        },
        'counts': {
            'tours': len(TOUR_NAME_TO_INDEX),
            'passages': len(MAPPING),
            'sessions': len(USER_SESSIONS)
        }
    }
    
    return jsonify(health_status)


@app.route('/api/reindex', methods=['POST'])
def reindex_endpoint():
    """Reindex endpoint (admin only)"""
    # Simple auth check
    auth_key = request.headers.get('X-Admin-Key')
    if auth_key != os.environ.get('ADMIN_KEY', 'default_admin_key'):
        return jsonify({'success': False, 'message': 'Unauthorized'}), 401
    
    try:
        # Reload data
        load_knowledge_data()
        load_mapping_data()
        load_field_keywords()
        load_tour_indices()
        
        # Clear indexes
        global VECTOR_INDEX, TFIDF_INDEX
        VECTOR_INDEX = None
        TFIDF_INDEX = None
        
        # Reload vector index
        load_vector_index()
        
        # Rebuild TF-IDF index
        if HAS_SKLEARN and MAPPING and USE_TFIDF_FALLBACK:
            TFIDF_INDEX = TFIDFIndex()
            TFIDF_INDEX.build(MAPPING)
        
        return jsonify({
            'success': True,
            'message': 'Reindex completed',
            'tours': len(TOUR_NAME_TO_INDEX),
            'passages': len(MAPPING)
        })
        
    except Exception as e:
        logger.error(f"❌ Reindex error: {e}")
        return jsonify({
            'success': False,
            'message': f'Reindex failed: {str(e)}'
        }), 500


@app.route('/', methods=['GET'])
def home():
    """Home page"""
    return jsonify({
        'service': 'Ruby Wings Chatbot API',
        'version': '2.0.0',
        'endpoints': {
            'POST /api/chat': 'Chat with the bot',
            'GET /api/tours': 'List all tours',
            'GET /api/context': 'Get current context',
            'POST /api/reset': 'Reset context',
            'GET /api/health': 'Health check',
            'POST /api/reindex': 'Reindex data (admin)'
        },
        'status': 'operational'
    })


# ---------- Data Loading ----------

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
        
        # Validate structure
        if 'tours' not in KNOWLEDGE_DATA:
            logger.warning("⚠️ Knowledge data missing 'tours' key")
        
        logger.info(f"✅ Loaded knowledge data with {len(KNOWLEDGE_DATA.get('tours', []))} tours")
        
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge data: {e}")
        KNOWLEDGE_DATA = {}


def load_mapping_data():
    """Tải mapping data từ file"""
    global MAPPING, METADATA
    
    if not os.path.exists(FAISS_MAPPING_PATH):
        logger.warning(f"⚠️ Mapping file not found: {FAISS_MAPPING_PATH}")
        MAPPING = []
        METADATA = []
        return
    
    try:
        with open(FAISS_MAPPING_PATH, 'r', encoding='utf-8') as f:
            mapping_data = json.load(f)
        
        # Kiểm tra cấu trúc
        if isinstance(mapping_data, dict) and 'mapping' in mapping_data:
            MAPPING = mapping_data['mapping']
            METADATA = mapping_data.get('metadata', [])
        else:
            MAPPING = mapping_data
            METADATA = []
        
        # Đảm bảo mỗi passage có các trường cần thiết
        for i, passage in enumerate(MAPPING):
            if 'text' not in passage:
                passage['text'] = ''
            if 'field' not in passage:
                passage['field'] = 'unknown'
            if 'tour_index' not in passage:
                passage['tour_index'] = None
            if 'tour_name' not in passage:
                passage['tour_name'] = None
        
        logger.info(f"✅ Loaded {len(MAPPING)} mapping entries")
        
    except Exception as e:
        logger.error(f"❌ Failed to load mapping data: {e}")
        MAPPING = []
        METADATA = []


def initialize_app():
    """Khởi tạo ứng dụng"""
    logger.info("=" * 60)
    logger.info("🚀 Initializing Ruby Wings Chatbot v2.0")
    logger.info("=" * 60)
    
    # Tải dữ liệu
    load_knowledge_data()
    load_mapping_data()
    load_field_keywords()
    load_tour_indices()
    
    # Tải vector index
    load_vector_index()
    
    # Xây dựng TF-IDF index (nếu được enable)
    global TFIDF_INDEX
    if HAS_SKLEARN and MAPPING and USE_TFIDF_FALLBACK:
        TFIDF_INDEX = TFIDFIndex()
        if not TFIDF_INDEX.build(MAPPING):
            logger.warning("⚠️ TF-IDF index build failed")
            TFIDF_INDEX = None
    else:
        TFIDF_INDEX = None
    
    # Log system status
    logger.info("📊 System Status:")
    logger.info(f"  • Knowledge: {len(KNOWLEDGE_DATA.get('tours', []))} tours")
    logger.info(f"  • Mapping: {len(MAPPING)} passages")
    logger.info(f"  • Tours indexed: {len(TOUR_NAME_TO_INDEX)}")
    logger.info(f"  • Vector index: {'Loaded' if VECTOR_INDEX else 'Not available'}")
    logger.info(f"  • TF-IDF index: {'Built' if TFIDF_INDEX else 'Not available'}")
    logger.info(f"  • OpenAI: {'Available' if HAS_OPENAI and OPENAI_API_KEY else 'Not available'}")
    logger.info(f"  • Redis: {'Available' if REDIS_CLIENT else 'Not available'}")
    logger.info(f"  • RapidFuzz: {'Available' if HAS_RAPIDFUZZ else 'Not available'}")
    logger.info(f"  • Scikit-learn: {'Available' if HAS_SKLEARN else 'Not available'}")
    logger.info(f"  • NLTK: {'Available' if HAS_NLTK else 'Not available'}")
    logger.info("=" * 60)
    logger.info("🎉 Ruby Wings Chatbot initialized successfully!")
    logger.info("=" * 60)


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
            except (ValueError, TypeError):
                expired_sessions.append(session_id)
    
    for session_id in expired_sessions:
        USER_SESSIONS.pop(session_id, None)


# ---------- Main ----------

if __name__ == '__main__':
    # Khởi tạo ứng dụng
    initialize_app()
    
    # Chạy server
    port = int(os.environ.get('PORT', 10000))
    logger.info(f"🌐 Starting server on port {port}")
    
    # Chạy với Flask dev server (chỉ cho development)
    app.run(host='0.0.0.0', port=port, debug=DEBUG, threaded=True)
else:
    # Khởi tạo khi chạy với WSGI server (gunicorn)
    initialize_app()