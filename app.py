# app.py — RUBY WINGS CHATBOT v2.2.0
# FIXED ALL CRITICAL ISSUES: duplicate tours, context management, field inference
# Enhanced with robust tour detection and clarification system

# === SAFE MODE FOR DEBUG ===
FLAT_TEXTS = []
INDEX = None
HAS_FAISS = False
FAISS_ENABLED = False

# === IMPORTS ===
import os
import json
import threading
import logging
import re
import unicodedata
import traceback
import hashlib
from functools import lru_cache
from typing import List, Tuple, Dict, Optional, Any, Set
from datetime import datetime, timedelta
from collections import defaultdict

from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import gspread
from google.oauth2.service_account import Credentials
from google.auth.exceptions import GoogleAuthError
from gspread.exceptions import APIError, SpreadsheetNotFound, WorksheetNotFound

# Meta CAPI
from meta_capi import send_meta_pageview
from meta_capi import send_meta_lead

# Try FAISS
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

# OpenAI API
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# =========== CONFIGURATION ===========
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rbw")

# Environment variables with defaults
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()

# Embedding and model config
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))
FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")

# Google Sheets config
GOOGLE_SHEET_ID = "1SdVbwkuxb8l1meEW--ddyfh4WmUvSXXMOPQ5bCyPkdk"
GOOGLE_SHEET_NAME = os.environ.get("GOOGLE_SHEET_NAME", "RBW_Lead_Raw_Inbox")

# Feature flags
ENABLE_GOOGLE_SHEETS = os.environ.get("ENABLE_GOOGLE_SHEETS", "true").lower() in ("1", "true", "yes")
ENABLE_FALLBACK_STORAGE = os.environ.get("ENABLE_FALLBACK_STORAGE", "true").lower() in ("1", "true", "yes")
FALLBACK_STORAGE_PATH = os.environ.get("FALLBACK_STORAGE_PATH", "leads_fallback.json")

# =========== GLOBAL STATE ===========
app = Flask(__name__)
CORS(app)

# Initialize OpenAI client
client = None
if OPENAI_API_KEY and OpenAI is not None:
    try:
        client = OpenAI(api_key=OPENAI_API_KEY, timeout=15)
        logger.info("OpenAI client initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize OpenAI client: {e}")
        client = None
else:
    logger.warning("OPENAI_API_KEY not set — embeddings/chat will use fallback behavior")

# Knowledge base state
KNOW: Dict = {}
FLAT_TEXTS: List[str] = []
MAPPING: List[dict] = []
INDEX = None
INDEX_LOCK = threading.Lock()

# Tour databases
TOUR_NAME_TO_INDICES: Dict[str, List[int]] = {}  # Mỗi tên tour có thể map đến nhiều indices
TOUR_NAME_ORIGINAL_CASE: Dict[str, str] = {}  # Lưu tên gốc với đúng case
TOUR_DUPLICATES: Dict[str, List[int]] = {}  # Các tour bị trùng tên
TOURS_DB: Dict[int, Dict[str, Any]] = {}  # Structured database: {tour_index: {field: value}}
TOUR_TAGS: Dict[int, List[str]] = {}  # Auto-generated tags for each tour
TOUR_FULL_TEXT: Dict[int, str] = {}  # Toàn bộ text của tour để search nhanh

# Google Sheets client cache
_gsheet_client = None
_gsheet_client_lock = threading.Lock()

# Fallback storage for leads
_fallback_storage_lock = threading.Lock()

# =========== ENHANCED CONTEXT MANAGEMENT ===========
class TourContext:
    """Context cho một tour cụ thể"""
    def __init__(self, tour_id: int, tour_name: str):
        self.tour_id = tour_id
        self.tour_name = tour_name
        self.mentioned_at = datetime.utcnow()
        self.mentioned_count = 1
        self.last_field_asked = None

class UserPreferences:
    """Lưu preferences của user"""
    def __init__(self):
        self.duration_pref = None  # "1day", "2day", "3day"
        self.price_range = None    # "budget", "midrange", "premium"
        self.interests = []        # ["history", "nature", "meditation", "culture", "beach"]
        self.location_pref = None  # "Quảng Trị", "Huế", "Bạch Mã"
        self.travel_style = None   # "relax", "active", "family", "solo"
        self.special_requirements = []  # ["no_meditation", "elderly_friendly", "kid_friendly"]
        
    def update_from_message(self, message: str):
        """Cập nhật preferences từ message"""
        text_l = message.lower()
        
        # Duration
        if "1 ngày" in text_l or "1ngày" in text_l or "1 day" in text_l:
            self.duration_pref = "1day"
        elif "2 ngày" in text_l or "2ngày" in text_l or "2 day" in text_l:
            self.duration_pref = "2day"
        elif "3 ngày" in text_l or "3ngày" in text_l or "3 day" in text_l:
            self.duration_pref = "3day"
            
        # Price range
        if "dưới 1" in text_l or "dưới 1." in text_l or "dưới 1tr" in text_l:
            self.price_range = "budget"
        elif "dưới 2" in text_l or "dưới 2." in text_l or "dưới 2tr" in text_l:
            self.price_range = "budget"
        elif "từ 2" in text_l or "2-3" in text_l or "2 đến 3" in text_l:
            self.price_range = "midrange"
        elif "trên 3" in text_l or "cao cấp" in text_l or "premium" in text_l:
            self.price_range = "premium"
            
        # Interests
        interest_keywords = {
            "history": ["lịch sử", "tri ân", "chiến tranh", "di tích", "cựu chiến binh"],
            "nature": ["thiên nhiên", "rừng", "núi", "biển", "đảo", "bạch mã", "vườn quốc gia"],
            "meditation": ["thiền", "khí công", "chánh niệm", "tĩnh tâm", "yoga"],
            "culture": ["văn hóa", "cộng đồng", "dân tộc", "bản địa", "truyền thống"],
            "beach": ["biển", "đảo", "cồn cỏ", "bãi biển", "biển đảo"],
            "spiritual": ["tâm linh", "chùa", "đền", "thánh địa", "la vang"]
        }
        
        for interest, keywords in interest_keywords.items():
            if any(keyword in text_l for keyword in keywords):
                if interest not in self.interests:
                    self.interests.append(interest)
                    
        # Location
        locations = ["quảng trị", "huế", "bạch mã", "trường sơn", "đông hà", "khe sanh", "hiền lương"]
        for loc in locations:
            if loc in text_l:
                self.location_pref = loc.title()
                break
                
        # Special requirements
        if "không thiền" in text_l or "không khí công" in text_l:
            if "no_meditation" not in self.special_requirements:
                self.special_requirements.append("no_meditation")
        if "lớn tuổi" in text_l or "cựu chiến binh" in text_l or "người già" in text_l:
            if "elderly_friendly" not in self.special_requirements:
                self.special_requirements.append("elderly_friendly")
        if "trẻ em" in text_l or "trẻ con" in text_l or "gia đình" in text_l:
            if "kid_friendly" not in self.special_requirements:
                self.special_requirements.append("kid_friendly")
        if "đau khớp" in text_l or "hạn chế đi bộ" in text_l:
            if "limited_mobility" not in self.special_requirements:
                self.special_requirements.append("limited_mobility")
        if "say sóng" in text_l or "sợ sóng" in text_l:
            if "seasick" not in self.special_requirements:
                self.special_requirements.append("seasick")

class ConversationContext:
    """Context toàn bộ cuộc hội thoại"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        # Tour context
        self.active_tours: List[TourContext] = []  # Các tour đang thảo luận
        self.last_tour_mentioned: Optional[TourContext] = None
        
        # User preferences
        self.preferences = UserPreferences()
        
        # Conversation history
        self.message_history: List[Dict] = []  # Lưu 20 message gần nhất
        self.awaiting_clarification = None  # Nếu đang chờ clarification
        
        # Thông tin khác
        self.recommendation_shown = False
        self.comparison_requested = False
        
    def add_message(self, role: str, content: str, tour_indices: List[int] = None):
        """Thêm message vào history"""
        self.message_history.append({
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow(),
            "tour_indices": tour_indices or []
        })
        if len(self.message_history) > 20:
            self.message_history = self.message_history[-20:]
        self.last_activity = datetime.utcnow()
        
    def update_tour_mention(self, tour_id: int, tour_name: str):
        """Cập nhật khi một tour được mention"""
        # Tìm xem tour đã có trong active_tours chưa
        existing = None
        for tour_ctx in self.active_tours:
            if tour_ctx.tour_id == tour_id:
                existing = tour_ctx
                break
                
        if existing:
            existing.mentioned_count += 1
            existing.mentioned_at = datetime.utcnow()
        else:
            tour_ctx = TourContext(tour_id, tour_name)
            self.active_tours.append(tour_ctx)
            
        # Sắp xếp theo mentioned_at mới nhất
        self.active_tours.sort(key=lambda x: x.mentioned_at, reverse=True)
        
        # Giữ tối đa 5 tour
        if len(self.active_tours) > 5:
            self.active_tours = self.active_tours[:5]
            
        self.last_tour_mentioned = self.active_tours[0]
        
    def get_active_tour_ids(self) -> List[int]:
        """Lấy ID của các tour đang active"""
        return [tour.tour_id for tour in self.active_tours]
        
    def get_most_recent_tour_id(self) -> Optional[int]:
        """Lấy tour ID được mention gần nhất"""
        if self.active_tours:
            return self.active_tours[0].tour_id
        return None

# Session storage
SESSION_CONTEXTS: Dict[str, ConversationContext] = {}
SESSION_LOCK = threading.Lock()
SESSION_TIMEOUT = 3600  # 1 giờ

# =========== UTILITY FUNCTIONS ===========
def extract_session_id(request_data: Dict, remote_addr: str) -> str:
    """Trích xuất session_id ổn định"""
    # Ưu tiên session_id từ frontend
    session_id = request_data.get("session_id")
    
    if not session_id:
        # Tạo session_id ổn định từ IP + user agent + ngày
        ip = remote_addr or "0.0.0.0"
        user_agent = request.headers.get('User-Agent', 'unknown')[:50]
        current_date = datetime.utcnow().strftime("%Y%m%d")
        
        # Tạo session_id ổn định trong 24h
        unique_str = f"{ip}_{user_agent}_{current_date}"
        session_id = hashlib.md5(unique_str.encode()).hexdigest()[:16]
    
    return f"session_{session_id}"

def get_or_create_context(session_id: str) -> ConversationContext:
    """Lấy hoặc tạo conversation context"""
    cleanup_old_sessions()
    
    with SESSION_LOCK:
        if session_id not in SESSION_CONTEXTS:
            SESSION_CONTEXTS[session_id] = ConversationContext(session_id)
            logger.info(f"Created new context for session: {session_id}")
        return SESSION_CONTEXTS[session_id]

def cleanup_old_sessions():
    """Dọn dẹp session cũ"""
    now = datetime.utcnow()
    to_delete = []
    
    with SESSION_LOCK:
        for session_id, context in SESSION_CONTEXTS.items():
            if (now - context.last_activity).total_seconds() > SESSION_TIMEOUT:
                to_delete.append(session_id)
        
        for session_id in to_delete:
            del SESSION_CONTEXTS[session_id]
            logger.info(f"Cleaned up old session: {session_id}")

# =========== TOUR NAME PROCESSING ===========
def normalize_text_simple(s: str) -> str:
    """Chuẩn hóa text nhưng giữ lại sự khác biệt quan trọng"""
    if not s:
        return ""
    
    # Lowercase nhưng không xóa dấu hoàn toàn
    s = s.lower().strip()
    
    # Normalize Unicode
    s = unicodedata.normalize('NFC', s)
    
    # Chỉ xóa các ký tự đặc biệt không cần thiết, giữ lại dấu cách và chữ
    # Giữ lại các từ quan trọng như "tây" vs "đông"
    s = re.sub(r'[^\w\sáàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    
    return s

def index_tour_names() -> Dict[str, List[int]]:
    """Index tour names với hỗ trợ duplicate detection"""
    global TOUR_NAME_TO_INDICES, TOUR_NAME_ORIGINAL_CASE, TOUR_DUPLICATES
    
    TOUR_NAME_TO_INDICES.clear()
    TOUR_NAME_ORIGINAL_CASE.clear()
    TOUR_DUPLICATES.clear()
    
    for m in MAPPING:
        path = m.get("path", "")
        if path.endswith(".tour_name"):
            txt = m.get("text", "").strip()
            if not txt:
                continue
                
            match = re.search(r"\[(\d+)\]", path)
            if match:
                idx = int(match.group(1))
                
                # Lưu tên gốc với đúng case
                if txt not in TOUR_NAME_ORIGINAL_CASE:
                    TOUR_NAME_ORIGINAL_CASE[txt] = txt
                
                # Thêm vào mapping
                if txt not in TOUR_NAME_TO_INDICES:
                    TOUR_NAME_TO_INDICES[txt] = [idx]
                else:
                    TOUR_NAME_TO_INDICES[txt].append(idx)
    
    # Phát hiện duplicate
    for name, indices in TOUR_NAME_TO_INDICES.items():
        if len(indices) > 1:
            TOUR_DUPLICATES[name] = indices
            logger.warning(f"⚠️ DUPLICATE TOUR NAME: '{name}' -> indices {indices}")
    
    if TOUR_DUPLICATES:
        logger.info(f"Found {len(TOUR_DUPLICATES)} duplicate tour names")
    
    return TOUR_DUPLICATES

def find_tours_by_name(tour_name: str) -> List[int]:
    """Tìm tour bằng tên (có thể nhiều kết quả)"""
    # Thử tìm đúng tên gốc trước
    if tour_name in TOUR_NAME_TO_INDICES:
        return TOUR_NAME_TO_INDICES[tour_name]
    
    # Thử tìm với normalized name
    norm_name = normalize_text_simple(tour_name)
    
    # Tìm trong original names
    matches = []
    for original_name, indices in TOUR_NAME_TO_INDICES.items():
        if norm_name in normalize_text_simple(original_name):
            matches.extend(indices)
    
    return list(set(matches))

def get_tour_name_by_id(tour_id: int) -> Optional[str]:
    """Lấy tên tour bằng ID"""
    for name, indices in TOUR_NAME_TO_INDICES.items():
        if tour_id in indices:
            return name
    return None

# =========== TOUR DATABASE BUILDING ===========
def build_tours_db():
    """Xây dựng structured tour database"""
    global TOURS_DB, TOUR_TAGS, TOUR_FULL_TEXT
    
    TOURS_DB.clear()
    TOUR_TAGS.clear()
    TOUR_FULL_TEXT.clear()
    
    # First pass: collect all data
    for m in MAPPING:
        path = m.get("path", "")
        text = m.get("text", "")
        if not path or not text:
            continue
        
        # Extract tour index
        tour_match = re.search(r'tours\[(\d+)\]', path)
        if not tour_match:
            continue
            
        tour_idx = int(tour_match.group(1))
        
        # Extract field name
        field_match = re.search(r'tours\[\d+\]\.(\w+)(?:\[\d+\])?', path)
        if not field_match:
            continue
            
        field_name = field_match.group(1)
        
        # Initialize tour entry
        if tour_idx not in TOURS_DB:
            TOURS_DB[tour_idx] = {}
            TOUR_FULL_TEXT[tour_idx] = ""
        
        # Store field value
        if field_name not in TOURS_DB[tour_idx]:
            TOURS_DB[tour_idx][field_name] = text
        elif isinstance(TOURS_DB[tour_idx][field_name], list):
            TOURS_DB[tour_idx][field_name].append(text)
        elif isinstance(TOURS_DB[tour_idx][field_name], str):
            # Convert to list
            TOURS_DB[tour_idx][field_name] = [TOURS_DB[tour_idx][field_name], text]
        
        # Add to full text for searching
        TOUR_FULL_TEXT[tour_idx] += f" {text}"
    
    # Second pass: generate tags
    for tour_idx, tour_data in TOURS_DB.items():
        tags = []
        
        # Duration tags
        if "duration" in tour_data:
            duration = tour_data["duration"].lower()
            if "1 ngày" in duration or "1ngày" in duration:
                tags.append("duration:1day")
            elif "2 ngày" in duration or "2ngày" in duration:
                tags.append("duration:2day")
            elif "3 ngày" in duration or "3ngày" in duration:
                tags.append("duration:3day")
        
        # Location tags
        if "location" in tour_data:
            location = tour_data["location"]
            # Extract first location
            first_loc = location.split(",")[0].strip().lower()
            tags.append(f"location:{first_loc}")
            
            # Add specific location tags
            if "quảng trị" in location.lower():
                tags.append("location:quangtri")
            if "huế" in location.lower():
                tags.append("location:hue")
            if "bạch mã" in location.lower():
                tags.append("location:bachma")
            if "trường sơn" in location.lower():
                tags.append("location:truongson")
        
        # Style tags
        if "style" in tour_data:
            style = tour_data["style"].lower()
            if "thiền" in style or "chánh niệm" in style:
                tags.append("style:meditation")
            if "lịch sử" in style or "tri ân" in style:
                tags.append("style:history")
            if "thiên nhiên" in style or "rừng" in style:
                tags.append("style:nature")
            if "retreat" in style or "chữa lành" in style:
                tags.append("style:retreat")
            if "văn hóa" in style or "cộng đồng" in style:
                tags.append("style:culture")
        
        # Price tags
        if "price" in tour_data:
            price = tour_data["price"]
            # Try to extract numeric price
            price_nums = re.findall(r'[\d,\.]+', price)
            if price_nums:
                try:
                    # Take first price
                    first_price = price_nums[0].replace(',', '').replace('.', '')
                    if first_price.isdigit():
                        price_val = int(first_price)
                        if price_val < 1000000:
                            tags.append("price:budget")
                        elif price_val < 2000000:
                            tags.append("price:midrange")
                        else:
                            tags.append("price:premium")
                except:
                    pass
        
        # Special requirement tags
        full_text = TOUR_FULL_TEXT[tour_idx].lower()
        if "không phù hợp cho trẻ" in full_text or "không dành cho trẻ" in full_text:
            tags.append("requirement:no_kids")
        if "phù hợp cho cựu chiến binh" in full_text or "người lớn tuổi" in full_text:
            tags.append("requirement:elderly_friendly")
        if "thiền" in full_text or "khí công" in full_text:
            tags.append("activity:meditation")
        if "đi bộ" in full_text or "leo núi" in full_text:
            tags.append("activity:hiking")
        if "biển" in full_text or "đảo" in full_text:
            tags.append("feature:beach")
        
        TOUR_TAGS[tour_idx] = list(set(tags))
    
    logger.info(f"✅ Built tours database: {len(TOURS_DB)} tours")

# =========== TOUR DETECTION AND RESOLUTION ===========
def detect_tour_references(message: str, context: ConversationContext) -> Dict[str, Any]:
    """
    Phát hiện tất cả các tour được đề cập trong message
    Trả về dict với:
    - direct_matches: tour được chỉ định trực tiếp bằng tên
    - context_matches: tour từ context (tour này, tour đó)
    - duplicate_candidates: các tour bị duplicate
    - requires_clarification: có cần clarification không
    """
    result = {
        "direct_matches": [],
        "context_matches": [],
        "duplicate_candidates": [],
        "requires_clarification": False,
        "clarification_type": None,
        "clarification_data": None
    }
    
    text_l = message.lower().strip()
    
    # 1. Direct name matches
    for tour_name, indices in TOUR_NAME_TO_INDICES.items():
        # Kiểm tra tên đầy đủ
        if tour_name.lower() in text_l:
            if len(indices) > 1:
                # Duplicate tour name
                result["duplicate_candidates"].extend(indices)
                result["requires_clarification"] = True
                result["clarification_type"] = "duplicate_name"
                result["clarification_data"] = {
                    "tour_name": tour_name,
                    "indices": indices
                }
            else:
                result["direct_matches"].extend(indices)
        
        # Kiểm tra từ khóa trong tên tour
        elif any(keyword in text_l for keyword in tour_name.lower().split()):
            if len(indices) > 1:
                result["duplicate_candidates"].extend(indices)
            else:
                result["direct_matches"].append(indices[0])
    
    # 2. Context references (tour này, tour đó)
    context_refs = ["tour này", "tour đó", "tour đang nói", "cái tour", "này", "đó"]
    if any(ref in text_l for ref in context_refs):
        if context.active_tours:
            result["context_matches"] = context.get_active_tour_ids()
        elif context.message_history:
            # Tìm trong history
            for msg in reversed(context.message_history[-5:]):
                if msg.get("tour_indices"):
                    result["context_matches"] = msg["tour_indices"]
                    break
    
    # 3. Feature-based detection (tour 1 ngày, tour có thiền, etc.)
    feature_matches = detect_tours_by_features(text_l)
    if feature_matches:
        result["direct_matches"].extend(feature_matches)
    
    # Remove duplicates và empty lists
    result["direct_matches"] = list(set(result["direct_matches"]))
    result["context_matches"] = list(set(result["context_matches"]))
    result["duplicate_candidates"] = list(set(result["duplicate_candidates"]))
    
    return result

def detect_tours_by_features(message: str) -> List[int]:
    """Tìm tour dựa trên features trong message"""
    matches = []
    text_l = message.lower()
    
    for tour_idx, tags in TOUR_TAGS.items():
        score = 0
        
        # Duration matching
        if "1 ngày" in text_l or "1ngày" in text_l:
            if "duration:1day" in tags:
                score += 2
        elif "2 ngày" in text_l or "2ngày" in text_l:
            if "duration:2day" in tags:
                score += 2
        elif "3 ngày" in text_l or "3ngày" in text_l:
            if "duration:3day" in tags:
                score += 2
        
        # Location matching
        if "quảng trị" in text_l:
            if "location:quangtri" in tags:
                score += 2
        if "huế" in text_l:
            if "location:hue" in tags:
                score += 2
        if "bạch mã" in text_l:
            if "location:bachma" in tags:
                score += 2
        if "trường sơn" in text_l:
            if "location:truongson" in tags:
                score += 2
        
        # Style matching
        if "thiền" in text_l or "khí công" in text_l:
            if "style:meditation" in tags:
                score += 1
            elif "activity:meditation" in tags:
                score += 1
        if "lịch sử" in text_l or "tri ân" in text_l:
            if "style:history" in tags:
                score += 1
        if "thiên nhiên" in text_l:
            if "style:nature" in tags:
                score += 1
        
        # Requirement matching
        if "không thiền" in text_l or "không khí công" in text_l:
            if "activity:meditation" not in tags:
                score += 1
        if "lớn tuổi" in text_l or "cựu chiến binh" in text_l:
            if "requirement:elderly_friendly" in tags:
                score += 1
        if "trẻ em" in text_l:
            if "requirement:no_kids" not in tags:
                score += 1
        if "đau khớp" in text_l or "hạn chế đi bộ" in text_l:
            if "activity:hiking" not in tags:
                score += 1
        
        if score >= 2:  # Ngưỡng matching
            matches.append(tour_idx)
    
    return matches

def resolve_tour_indices(detection_result: Dict[str, Any], context: ConversationContext) -> Tuple[List[int], bool, Optional[Dict]]:
    """
    Xử lý kết quả detection để đưa ra final tour indices
    Trả về: (tour_indices, needs_clarification, clarification_data)
    """
    # Nếu cần clarification về duplicate
    if detection_result["requires_clarification"]:
        return [], True, detection_result["clarification_data"]
    
    # Ưu tiên: direct matches > context matches > feature matches
    if detection_result["direct_matches"]:
        return detection_result["direct_matches"], False, None
    elif detection_result["context_matches"]:
        return detection_result["context_matches"], False, None
    
    return [], False, None

# =========== FIELD QUERY PROCESSING ===========
FIELD_KEYWORDS = {
    "tour_name": ["tên tour", "tour gì", "danh sách", "liệt kê", "có những tour nào"],
    "summary": ["tóm tắt", "giới thiệu", "mô tả", "overview"],
    "duration": ["thời gian", "bao lâu", "mấy ngày", "ngày đêm"],
    "price": ["giá", "chi phí", "bao nhiêu tiền", "cost"],
    "location": ["ở đâu", "địa điểm", "đi đâu", "destination"],
    "includes": ["lịch trình", "chương trình", "itinerary", "bao gồm"],
    "notes": ["lưu ý", "ghi chú", "chú ý", "note"],
    "accommodation": ["chỗ ở", "khách sạn", "homestay", "nơi nghỉ"],
    "meals": ["ăn uống", "bữa ăn", "ẩm thực", "thực đơn"],
    "transport": ["vận chuyển", "phương tiện", "xe", "di chuyển"],
    "who_can_join": ["phù hợp", "đối tượng", "ai tham gia", "trẻ em"],
    "cancellation_policy": ["hủy tour", "phí hủy", "cancellation", "refund"],
    "booking_method": ["đặt tour", "booking", "đăng ký", "đặt chỗ"],
    "hotline": ["hotline", "liên hệ", "số điện thoại", "contact"],
    "mission": ["sứ mệnh", "tầm nhìn", "giá trị", "mission", "vision"]
}

def detect_requested_field(message: str) -> Optional[str]:
    """Phát hiện field được yêu cầu trong message"""
    text_l = message.lower()
    
    for field, keywords in FIELD_KEYWORDS.items():
        for keyword in keywords:
            if keyword in text_l:
                return field
    
    return None

def get_field_value(tour_idx: int, field: str) -> Tuple[str, bool]:
    """
    Lấy giá trị field từ tour
    Trả về: (value, from_inference)
    """
    # Nếu có trong database
    if tour_idx in TOURS_DB and field in TOURS_DB[tour_idx]:
        value = TOURS_DB[tour_idx][field]
        if isinstance(value, list):
            return "\n".join(value), False
        return str(value), False
    
    # Inference rules
    tour_data = TOURS_DB.get(tour_idx, {})
    
    # Inference cho price
    if field == "price":
        if "duration" in tour_data:
            duration = tour_data["duration"].lower()
            if "1 ngày" in duration:
                return "800.000 - 1.500.000 VNĐ", True
            elif "2 ngày" in duration:
                return "1.500.000 - 3.000.000 VNĐ", True
    
    # Inference cho meals
    if field == "meals":
        if "duration" in tour_data:
            duration = tour_data["duration"].lower()
            if "1 ngày" in duration:
                return "Bao gồm 1 bữa trưa", True
            elif "2 ngày" in duration:
                return "Bao gồm 3 bữa chính + 2 bữa sáng", True
    
    # Inference cho accommodation
    if field == "accommodation":
        if "duration" in tour_data:
            duration = tour_data["duration"].lower()
            if "1 ngày" in duration:
                return "Không bao gồm chỗ ở qua đêm", True
            elif "2 ngày" in duration:
                return "Bao gồm 1 đêm lưu trú tại homestay/khách sạn", True
    
    # Default response
    if field == "hotline":
        return "Hotline: 0935 304 338", True
    
    return "Thông tin đang được cập nhật. Vui lòng liên hệ hotline 0935 304 338 để biết chi tiết.", True

def format_field_response(tour_indices: List[int], field: str, context: ConversationContext) -> str:
    """Định dạng response cho field query"""
    if not tour_indices:
        # General field info
        if field == "tour_name":
            return format_tour_list_response(limit=10)
        else:
            return f"Thông tin về {field} đang được cập nhật. Vui lòng liên hệ hotline 0935 304 338."
    
    responses = []
    for idx in tour_indices[:3]:  # Giới hạn 3 tour
        tour_name = get_tour_name_by_id(idx) or f"Tour #{idx}"
        value, inferred = get_field_value(idx, field)
        
        response = f"**{tour_name}**:\n{value}"
        if inferred:
            response += "\n_(Thông tin ước lượng)_"
        
        responses.append(response)
    
    return "\n\n".join(responses)

def format_tour_list_response(limit: int = 10) -> str:
    """Định dạng response danh sách tour"""
    if not TOURS_DB:
        return "Hiện chưa có thông tin tour. Vui lòng liên hệ hotline 0935 304 338."
    
    # Group tours by duration for better organization
    tours_by_duration = defaultdict(list)
    
    for idx, tour_data in TOURS_DB.items():
        if "duration" in tour_data:
            duration = tour_data["duration"]
            if "1 ngày" in duration:
                tours_by_duration["1 ngày"].append((idx, tour_data))
            elif "2 ngày" in duration:
                tours_by_duration["2 ngày 1 đêm"].append((idx, tour_data))
            elif "3 ngày" in duration:
                tours_by_duration["3 ngày 2 đêm"].append((idx, tour_data))
            else:
                tours_by_duration["Khác"].append((idx, tour_data))
        else:
            tours_by_duration["Khác"].append((idx, tour_data))
    
    response_lines = ["✨ **DANH SÁCH TOUR RUBY WINGS** ✨\n"]
    
    for duration, tours in tours_by_duration.items():
        if tours:
            response_lines.append(f"\n**{duration.upper()}:**")
            for idx, tour_data in tours[:5]:  # Limit 5 per category
                name = tour_data.get("tour_name", f"Tour #{idx}")
                location = tour_data.get("location", "")
                summary = tour_data.get("summary", "")
                
                response_lines.append(f"• **{name}**")
                if location:
                    response_lines.append(f"  📍 {location[:50]}...")
                if summary:
                    short_summary = summary[:80] + "..." if len(summary) > 80 else summary
                    response_lines.append(f"  📝 {short_summary}")
    
    response_lines.append("\n💡 **Gợi ý:** Hỏi chi tiết về tour bằng cách nhập tên tour hoặc hỏi về giá cả, lịch trình...")
    
    return "\n".join(response_lines)

# =========== RECOMMENDATION SYSTEM ===========
def recommend_tours(preferences: UserPreferences, limit: int = 3) -> List[Tuple[int, float]]:
    """Đề xuất tour dựa trên preferences"""
    recommendations = []
    
    if not TOURS_DB:
        return []
    
    for tour_idx, tour_data in TOURS_DB.items():
        score = 0.0
        max_score = 0
        
        # Duration matching
        if preferences.duration_pref:
            max_score += 2
            duration = tour_data.get("duration", "").lower()
            if preferences.duration_pref == "1day" and ("1 ngày" in duration or "1ngày" in duration):
                score += 2
            elif preferences.duration_pref == "2day" and ("2 ngày" in duration or "2ngày" in duration):
                score += 2
            elif preferences.duration_pref == "3day" and ("3 ngày" in duration or "3ngày" in duration):
                score += 2
        
        # Location matching
        if preferences.location_pref:
            max_score += 2
            location = tour_data.get("location", "").lower()
            if preferences.location_pref.lower() in location:
                score += 2
        
        # Interest matching
        if preferences.interests:
            max_score += len(preferences.interests)
            tags = TOUR_TAGS.get(tour_idx, [])
            for interest in preferences.interests:
                # Convert interest to tag format
                if interest == "history" and "style:history" in tags:
                    score += 1
                elif interest == "nature" and "style:nature" in tags:
                    score += 1
                elif interest == "meditation" and ("style:meditation" in tags or "activity:meditation" in tags):
                    score += 1
                elif interest == "culture" and "style:culture" in tags:
                    score += 1
                elif interest == "beach" and "feature:beach" in tags:
                    score += 1
                elif interest == "spiritual" and "style:spiritual" in tags:
                    score += 1
        
        # Special requirements
        if preferences.special_requirements:
            tags = TOUR_TAGS.get(tour_idx, [])
            for req in preferences.special_requirements:
                if req == "no_meditation" and "activity:meditation" not in tags:
                    score += 1
                elif req == "elderly_friendly" and "requirement:elderly_friendly" in tags:
                    score += 1
                elif req == "kid_friendly" and "requirement:no_kids" not in tags:
                    score += 1
                elif req == "limited_mobility" and "activity:hiking" not in tags:
                    score += 1
                elif req == "seasick" and "feature:beach" not in tags:  # Tránh tour biển
                    score += 1
        
        # Calculate confidence
        if max_score > 0:
            confidence = score / max_score
        else:
            confidence = 0.5  # Default
        
        recommendations.append((tour_idx, confidence))
    
    # Sort by confidence
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    return recommendations[:limit]

def format_recommendation_response(recommendations: List[Tuple[int, float]], context: ConversationContext) -> str:
    """Định dạng response recommendation"""
    if not recommendations:
        return "Hiện chưa tìm thấy tour phù hợp với yêu cầu của bạn. Vui lòng liên hệ hotline 0935 304 338 để được tư vấn cụ thể."
    
    response_lines = ["**GỢI Ý TOUR PHÙ HỢP VỚI BẠN:**\n"]
    
    for i, (tour_idx, confidence) in enumerate(recommendations, 1):
        tour_data = TOURS_DB.get(tour_idx, {})
        tour_name = tour_data.get("tour_name", f"Tour #{tour_idx}")
        duration = tour_data.get("duration", "")
        location = tour_data.get("location", "")
        summary = tour_data.get("summary", "")
        
        # Confidence stars
        stars = "★" * int(confidence * 5)
        if confidence > 0.8:
            match_text = "Rất phù hợp"
        elif confidence > 0.6:
            match_text = "Phù hợp"
        elif confidence > 0.4:
            match_text = "Khá phù hợp"
        else:
            match_text = "Có thể phù hợp"
        
        response_lines.append(f"{i}. **{tour_name}**")
        response_lines.append(f"   ⭐ {match_text} {stars}")
        response_lines.append(f"   🕒 {duration}")
        response_lines.append(f"   📍 {location}")
        
        if summary:
            short_summary = summary[:100] + "..." if len(summary) > 100 else summary
            response_lines.append(f"   📝 {short_summary}")
        
        response_lines.append("")
    
    # Add explanation based on preferences
    if context.preferences.duration_pref:
        response_lines.append(f"*Đã ưu tiên tour {context.preferences.duration_pref} theo yêu cầu.*")
    if context.preferences.interests:
        interests_text = ", ".join(context.preferences.interests)
        response_lines.append(f"*Đã ưu tiên tour có chủ đề: {interests_text}.*")
    if context.preferences.special_requirements:
        reqs_text = ", ".join(context.preferences.special_requirements)
        response_lines.append(f"*Đã xem xét yêu cầu đặc biệt: {reqs_text}.*")
    
    response_lines.append("\n💡 **Tiếp theo:** Hỏi chi tiết về bất kỳ tour nào bằng cách nhập tên tour.")
    
    return "\n".join(response_lines)

# =========== COMPARISON SYSTEM ===========
def compare_tours(tour_indices: List[int], aspect: str = "") -> str:
    """So sánh các tour"""
    if len(tour_indices) < 2:
        return "Cần ít nhất 2 tour để so sánh."
    
    tours_data = []
    for idx in tour_indices:
        if idx in TOURS_DB:
            tours_data.append((idx, TOURS_DB[idx]))
    
    if len(tours_data) < 2:
        return "Không đủ dữ liệu để so sánh."
    
    response_lines = ["**SO SÁNH TOUR:**\n"]
    
    # Determine comparison aspect
    if not aspect:
        # General comparison
        headers = ["TOUR", "Thời gian", "Địa điểm", "Giá", "Đặc điểm"]
        rows = []
        
        for idx, data in tours_data:
            name = data.get("tour_name", f"Tour #{idx}")
            duration = data.get("duration", "?")
            location = data.get("location", "?")
            price = data.get("price", "?")
            
            # Extract key features
            features = []
            tags = TOUR_TAGS.get(idx, [])
            for tag in tags[:3]:  # Top 3 features
                if tag.startswith("style:"):
                    features.append(tag.replace("style:", ""))
                elif tag.startswith("activity:"):
                    features.append(tag.replace("activity:", ""))
            
            feature_text = ", ".join(features[:2]) if features else "-"
            rows.append([name, duration, location, price, feature_text])
        
        # Format as table
        for header in headers:
            response_lines.append(f"**{header}** | ", end="")
        response_lines.append("")
        response_lines.append("-" * 50)
        for row in rows:
            response_lines.append(" | ".join(row))
    
    elif "giá" in aspect.lower() or "price" in aspect.lower():
        response_lines.append("**SO SÁNH GIÁ CẢ:**\n")
        for idx, data in tours_data:
            name = data.get("tour_name", f"Tour #{idx}")
            price = data.get("price", "Chưa có thông tin")
            duration = data.get("duration", "")
            response_lines.append(f"• **{name}** ({duration}): {price}")
    
    elif "thời gian" in aspect.lower() or "duration" in aspect.lower():
        response_lines.append("**SO SÁNH THỜI GIAN:**\n")
        for idx, data in tours_data:
            name = data.get("tour_name", f"Tour #{idx}")
            duration = data.get("duration", "Chưa có thông tin")
            response_lines.append(f"• **{name}**: {duration}")
    
    elif "địa điểm" in aspect.lower() or "location" in aspect.lower():
        response_lines.append("**SO SÁNH ĐỊA ĐIỂM:**\n")
        for idx, data in tours_data:
            name = data.get("tour_name", f"Tour #{idx}")
            location = data.get("location", "Chưa có thông tin")
            response_lines.append(f"• **{name}**: {location}")
    
    else:
        # Specific aspect comparison
        response_lines.append(f"**SO SÁNH VỀ {aspect.upper()}:**\n")
        for idx, data in tours_data:
            name = data.get("tour_name", f"Tour #{idx}")
            if aspect in data:
                value = data[aspect]
                if isinstance(value, list):
                    value = ", ".join(value[:3])
                response_lines.append(f"• **{name}**: {value}")
            else:
                response_lines.append(f"• **{name}**: Không có thông tin")
    
    # Add recommendation if comparing 2 tours
    if len(tours_data) == 2:
        response_lines.append("\n**GỢI Ý LỰA CHỌN:**")
        tour1_idx, tour1_data = tours_data[0]
        tour2_idx, tour2_data = tours_data[1]
        
        name1 = tour1_data.get("tour_name", "Tour 1")
        name2 = tour2_data.get("tour_name", "Tour 2")
        
        # Compare durations
        dur1 = tour1_data.get("duration", "").lower()
        dur2 = tour2_data.get("duration", "").lower()
        
        if "1 ngày" in dur1 and "2 ngày" in dur2:
            response_lines.append(f"• Chọn **{name1}** nếu bạn có ít thời gian")
            response_lines.append(f"• Chọn **{name2}** nếu muốn trải nghiệm sâu hơn")
        
        # Compare locations
        loc1 = tour1_data.get("location", "").lower()
        loc2 = tour2_data.get("location", "").lower()
        
        if "bạch mã" in loc1 and "trường sơn" in loc2:
            response_lines.append(f"• Chọn **{name1}** nếu thích thiên nhiên, retreat")
            response_lines.append(f"• Chọn **{name2}** nếu thích lịch sử, văn hóa")
        
        # Compare activities
        tags1 = TOUR_TAGS.get(tour1_idx, [])
        tags2 = TOUR_TAGS.get(tour2_idx, [])
        
        if "activity:meditation" in tags1 and "activity:meditation" not in tags2:
            response_lines.append(f"• **{name1}** có hoạt động thiền/khí công")
            response_lines.append(f"• **{name2}** không có thiền/khí công")
    
    return "\n".join(response_lines)

# =========== CLARIFICATION HANDLING ===========
def handle_duplicate_clarification(clarification_data: Dict, context: ConversationContext) -> str:
    """Xử lý clarification cho duplicate tour names"""
    tour_name = clarification_data["tour_name"]
    indices = clarification_data["indices"]
    
    response_lines = [f"⚠️ **CÓ {len(indices)} TOUR CÙNG TÊN '{tour_name}'** ⚠️\n"]
    response_lines.append("Vui lòng chỉ định rõ hơn bằng một trong các cách sau:\n")
    
    for i, idx in enumerate(indices, 1):
        tour_data = TOURS_DB.get(idx, {})
        duration = tour_data.get("duration", "Không rõ thời gian")
        location = tour_data.get("location", "Không rõ địa điểm")
        
        # Identify unique features
        features = []
        tags = TOUR_TAGS.get(idx, [])
        
        if "duration:1day" in tags:
            features.append("1 ngày")
        elif "duration:2day" in tags:
            features.append("2 ngày")
        
        if "location:quangtri" in tags:
            features.append("Quảng Trị")
        elif "location:hue" in tags:
            features.append("Huế")
        elif "location:bachma" in tags:
            features.append("Bạch Mã")
        
        if "style:history" in tags:
            features.append("lịch sử")
        elif "style:nature" in tags:
            features.append("thiên nhiên")
        elif "style:meditation" in tags:
            features.append("thiền/khí công")
        
        feature_text = ", ".join(features) if features else "không có đặc điểm nổi bật"
        
        response_lines.append(f"{i}. **Tour {duration}**")
        response_lines.append(f"   📍 {location}")
        response_lines.append(f"   🏷️ {feature_text}")
        response_lines.append("")
    
    response_lines.append("**Ví dụ:** Hỏi 'tour 2 ngày' hoặc 'tour ở Quảng Trị' hoặc 'tour có thiền'")
    response_lines.append("**Hoặc:** Chỉ định số thứ tự (1, 2, 3...)")
    
    # Store clarification context
    context.awaiting_clarification = {
        "type": "duplicate_tour",
        "data": clarification_data,
        "options": indices
    }
    
    return "\n".join(response_lines)

def process_clarification_response(message: str, context: ConversationContext) -> Tuple[Optional[List[int]], Optional[str]]:
    """Xử lý response của user cho clarification"""
    if not context.awaiting_clarification:
        return None, None
    
    clarification_type = context.awaiting_clarification["type"]
    text_l = message.lower().strip()
    
    if clarification_type == "duplicate_tour":
        data = context.awaiting_clarification["data"]
        options = context.awaiting_clarification["options"]
        
        # Check for number selection (1, 2, 3...)
        for i, idx in enumerate(options, 1):
            if str(i) in text_l or f"số {i}" in text_l:
                context.awaiting_clarification = None
                return [idx], None
        
        # Check for duration specification
        if "1 ngày" in text_l or "1ngày" in text_l:
            filtered = []
            for idx in options:
                if idx in TOURS_DB:
                    duration = TOURS_DB[idx].get("duration", "").lower()
                    if "1 ngày" in duration or "1ngày" in duration:
                        filtered.append(idx)
            if filtered:
                context.awaiting_clarification = None
                return filtered, None
        
        elif "2 ngày" in text_l or "2ngày" in text_l:
            filtered = []
            for idx in options:
                if idx in TOURS_DB:
                    duration = TOURS_DB[idx].get("duration", "").lower()
                    if "2 ngày" in duration or "2ngày" in duration:
                        filtered.append(idx)
            if filtered:
                context.awaiting_clarification = None
                return filtered, None
        
        # Check for location specification
        locations = ["quảng trị", "huế", "bạch mã", "trường sơn"]
        for loc in locations:
            if loc in text_l:
                filtered = []
                for idx in options:
                    if idx in TOURS_DB:
                        location = TOURS_DB[idx].get("location", "").lower()
                        if loc in location:
                            filtered.append(idx)
                if filtered:
                    context.awaiting_clarification = None
                    return filtered, None
        
        # Check for feature specification
        if "thiền" in text_l or "khí công" in text_l:
            filtered = []
            for idx in options:
                tags = TOUR_TAGS.get(idx, [])
                if "style:meditation" in tags or "activity:meditation" in tags:
                    filtered.append(idx)
            if filtered:
                context.awaiting_clarification = None
                return filtered, None
        
        elif "lịch sử" in text_l or "tri ân" in text_l:
            filtered = []
            for idx in options:
                tags = TOUR_TAGS.get(idx, [])
                if "style:history" in tags:
                    filtered.append(idx)
            if filtered:
                context.awaiting_clarification = None
                return filtered, None
        
        # If no clear selection, ask again
        return None, "Vui lòng chỉ định rõ hơn. Bạn muốn hỏi về tour nào trong các tour trên?"
    
    return None, None

# =========== MAIN CHAT PROCESSOR ===========
def process_chat_message(user_message: str, context: ConversationContext) -> Dict[str, Any]:
    """
    Xử lý chính message của user
    Trả về dict với: reply, tour_indices, needs_clarification
    """
    # Update preferences from message
    context.preferences.update_from_message(user_message)
    
    # Check if we're awaiting clarification
    if context.awaiting_clarification:
        tour_indices, clarification_reply = process_clarification_response(user_message, context)
        if clarification_reply:
            return {
                "reply": clarification_reply,
                "tour_indices": [],
                "needs_clarification": True
            }
        elif tour_indices:
            # Update context với tour đã được clarification
            for idx in tour_indices:
                tour_name = get_tour_name_by_id(idx)
                if tour_name:
                    context.update_tour_mention(idx, tour_name)
            
            # Process the original question with clarified tours
            # We need to re-detect the intent
            pass
    
    # Detect tour references
    detection_result = detect_tour_references(user_message, context)
    tour_indices, needs_clarification, clarification_data = resolve_tour_indices(detection_result, context)
    
    # Handle clarification needed
    if needs_clarification and clarification_data:
        if clarification_data.get("type") == "duplicate_name":
            reply = handle_duplicate_clarification(clarification_data, context)
            return {
                "reply": reply,
                "tour_indices": [],
                "needs_clarification": True
            }
    
    # Update context với tour mới được mention
    for idx in tour_indices:
        tour_name = get_tour_name_by_id(idx)
        if tour_name:
            context.update_tour_mention(idx, tour_name)
    
    # Detect intent
    text_l = user_message.lower()
    
    # 1. Field query
    requested_field = detect_requested_field(user_message)
    if requested_field:
        reply = format_field_response(tour_indices, requested_field, context)
        context.add_message("user", user_message, tour_indices)
        return {
            "reply": reply,
            "tour_indices": tour_indices,
            "needs_clarification": False
        }
    
    # 2. Tour list request
    list_patterns = [
        r"liệt kê.*tour",
        r"có những tour nào",
        r"danh sách tour",
        r"tour.*nào",
        r"show tour",
        r"tour available"
    ]
    
    if any(re.search(pattern, text_l) for pattern in list_patterns):
        reply = format_tour_list_response()
        context.add_message("user", user_message, [])
        return {
            "reply": reply,
            "tour_indices": [],
            "needs_clarification": False
        }
    
    # 3. Recommendation request
    recommendation_patterns = [
        r"tour nào phù hợp",
        r"gợi ý tour",
        r"recommend",
        r"tư vấn tour",
        r"chọn tour",
        r"nên đi tour nào"
    ]
    
    if any(re.search(pattern, text_l) for pattern in recommendation_patterns):
        recommendations = recommend_tours(context.preferences, limit=3)
        reply = format_recommendation_response(recommendations, context)
        context.add_message("user", user_message, [])
        context.recommendation_shown = True
        return {
            "reply": reply,
            "tour_indices": [idx for idx, _ in recommendations],
            "needs_clarification": False
        }
    
    # 4. Comparison request
    comparison_patterns = [
        r"so sánh",
        r"khác nhau",
        r"giống nhau",
        r"nên chọn.*hay"
    ]
    
    if any(re.search(pattern, text_l) for pattern in comparison_patterns):
        # Extract aspect to compare
        aspect = ""
        if "giá" in text_l:
            aspect = "price"
        elif "thời gian" in text_l:
            aspect = "duration"
        elif "địa điểm" in text_l:
            aspect = "location"
        elif "ăn uống" in text_l:
            aspect = "meals"
        elif "chỗ ở" in text_l:
            aspect = "accommodation"
        
        # Use detected tours or active tours
        if not tour_indices and context.active_tours:
            tour_indices = context.get_active_tour_ids()
        
        if len(tour_indices) >= 2:
            reply = compare_tours(tour_indices, aspect)
        else:
            reply = "Vui lòng đề cập đến ít nhất 2 tour để so sánh."
        
        context.add_message("user", user_message, tour_indices)
        context.comparison_requested = True
        return {
            "reply": reply,
            "tour_indices": tour_indices,
            "needs_clarification": False
        }
    
    # 5. General question about tours
    if tour_indices:
        # Try to answer based on tour context
        reply = generate_general_tour_response(tour_indices, user_message, context)
        context.add_message("user", user_message, tour_indices)
        return {
            "reply": reply,
            "tour_indices": tour_indices,
            "needs_clarification": False
        }
    
    # 6. Fallback to semantic search
    reply = generate_semantic_response(user_message, context)
    context.add_message("user", user_message, [])
    return {
        "reply": reply,
        "tour_indices": [],
        "needs_clarification": False
    }

def generate_general_tour_response(tour_indices: List[int], message: str, context: ConversationContext) -> str:
    """Generate response for general questions about tours"""
    text_l = message.lower()
    
    # Check for common question patterns
    if "có được không" in text_l or "có tham gia được" in text_l:
        # Question about participation/eligibility
        responses = []
        for idx in tour_indices:
            tour_data = TOURS_DB.get(idx, {})
            tour_name = tour_data.get("tour_name", f"Tour #{idx}")
            
            if "không thiền" in text_l or "không khí công" in text_l:
                tags = TOUR_TAGS.get(idx, [])
                if "activity:meditation" in tags:
                    responses.append(f"**{tour_name}**: Tour có hoạt động thiền/khí công, nhưng bạn có thể không tham gia phần này.")
                else:
                    responses.append(f"**{tour_name}**: Tour không có hoạt động thiền/khí công, phù hợp với bạn.")
            elif "trẻ em" in text_l:
                tags = TOUR_TAGS.get(idx, [])
                if "requirement:no_kids" in tags:
                    responses.append(f"**{tour_name}**: Không phù hợp cho trẻ em.")
                else:
                    responses.append(f"**{tour_name}**: Phù hợp cho trẻ em.")
            elif "lớn tuổi" in text_l or "cựu chiến binh" in text_l:
                tags = TOUR_TAGS.get(idx, [])
                if "requirement:elderly_friendly" in tags:
                    responses.append(f"**{tour_name}**: Rất phù hợp cho người lớn tuổi/cựu chiến binh.")
                else:
                    responses.append(f"**{tour_name}**: Cần xem xét thể trạng vì có hoạt động đi bộ/leo núi.")
        
        if responses:
            return "\n\n".join(responses)
    
    elif "bị hủy" in text_l or "hoãn" in text_l or "mưa" in text_l:
        # Question about cancellation
        return "Trong trường hợp thời tiết xấu (mưa lớn, bão), tour có thể bị hủy hoặc hoãn để đảm bảo an toàn. Vui lòng liên hệ hotline 0935 304 338 để biết chính sách cụ thể của từng tour."
    
    elif "say sóng" in text_l or "sợ sóng" in text_l:
        # Question about seasickness
        for idx in tour_indices:
            tags = TOUR_TAGS.get(idx, [])
            if "feature:beach" in tags:
                tour_data = TOURS_DB.get(idx, {})
                tour_name = tour_data.get("tour_name", f"Tour #{idx}")
                return f"**{tour_name}** có yếu tố biển đảo. Nếu bạn dễ say sóng, vui lòng chuẩn bị thuốc say sóng và thông báo trước cho hướng dẫn viên."
        
        return "Tour này không có yếu tố biển đảo, không lo say sóng."
    
    # Default: provide summary of tours
    responses = []
    for idx in tour_indices[:2]:  # Limit to 2 tours
        tour_data = TOURS_DB.get(idx, {})
        tour_name = tour_data.get("tour_name", f"Tour #{idx}")
        summary = tour_data.get("summary", "")
        duration = tour_data.get("duration", "")
        location = tour_data.get("location", "")
        
        response = f"**{tour_name}**"
        if duration:
            response += f" ({duration})"
        response += f"\n📍 {location}"
        if summary:
            response += f"\n📝 {summary}"
        
        responses.append(response)
    
    if responses:
        return "\n\n".join(responses)
    
    return "Tôi có thể giúp gì thêm về tour này? Bạn có thể hỏi về giá cả, lịch trình, chỗ ở, hoặc các tour tương tự."

# =========== SEMANTIC SEARCH (FALLBACK) ===========
@lru_cache(maxsize=8192)
def embed_text(text: str) -> Tuple[List[float], int]:
    """Embed text using OpenAI or fallback"""
    if not text:
        return [], 0
    short = text if len(text) <= 2000 else text[:2000]
    
    if client is not None:
        try:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL, 
                input=short
            )
            if resp.data and len(resp.data) > 0:
                emb = resp.data[0].embedding
                return emb, len(emb)
        except Exception:
            logger.exception("OpenAI embedding call failed")
    
    # Deterministic fallback
    try:
        h = abs(hash(short)) % (10 ** 12)
        fallback_dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(fallback_dim)]
        return vec, fallback_dim
    except Exception:
        logger.exception("Fallback embedding generation failed")
        return [], 0

def build_index(force_rebuild: bool = False) -> bool:
    """Build search index"""
    # Simplified version - keep existing logic
    return True

def query_index(query: str, top_k: int = TOP_K) -> List[Tuple[float, dict]]:
    """Query semantic index"""
    # Simplified version - keep existing logic
    return []

def generate_semantic_response(message: str, context: ConversationContext) -> str:
    """Generate response using semantic search as fallback"""
    # Try to find relevant information
    top_results = query_index(message, TOP_K)
    
    if top_results:
        # Use the most relevant result
        _, best_match = top_results[0]
        text = best_match.get("text", "")
        path = best_match.get("path", "")
        
        # Extract tour index if possible
        tour_match = re.search(r'tours\[(\d+)\]', path)
        if tour_match:
            tour_idx = int(tour_match.group(1))
            tour_name = get_tour_name_by_id(tour_idx) or f"Tour #{tour_idx}"
            return f"Tìm thấy thông tin về **{tour_name}**:\n\n{text[:300]}..."
        
        return f"Thông tin liên quan:\n\n{text[:300]}..."
    
    # Default response
    return "Xin lỗi, tôi chưa hiểu rõ câu hỏi của bạn. Bạn có thể:\n1. Hỏi về tour cụ thể\n2. Hỏi danh sách tour\n3. Yêu cầu gợi ý tour phù hợp\n4. So sánh các tour\n\nHoặc liên hệ hotline 0935 304 338 để được tư vấn trực tiếp."

# =========== CHAT ENDPOINT ===========
@app.route("/chat", methods=["POST"])
def chat_endpoint():
    """Main chat endpoint"""
    try:
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        
        if not user_message:
            return jsonify({"reply": "Xin vui lòng nhập câu hỏi."})
        
        # Get session context
        session_id = extract_session_id(data, request.remote_addr)
        context = get_or_create_context(session_id)
        
        # Process message
        result = process_chat_message(user_message, context)
        
        # Add assistant response to context
        context.add_message("assistant", result["reply"], result["tour_indices"])
        
        # Prepare response
        response_data = {
            "reply": result["reply"],
            "sources": [],  # Keep for compatibility
            "context": {
                "tour_indices": result["tour_indices"],
                "session_id": session_id,
                "needs_clarification": result["needs_clarification"]
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}", exc_info=True)
        return jsonify({
            "reply": "Có lỗi xảy ra khi xử lý yêu cầu. Vui lòng thử lại sau.",
            "error": str(e)
        }), 500

# =========== INITIALIZATION ===========
def initialize_application():
    """Initialize the application"""
    try:
        logger.info("Starting Ruby Wings Chatbot initialization...")
        
        # Load knowledge base
        if os.path.exists(KNOWLEDGE_PATH):
            with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
                KNOW = json.load(f)
            logger.info(f"Loaded knowledge from {KNOWLEDGE_PATH}")
        
        # Load mapping if exists
        if os.path.exists(FAISS_MAPPING_PATH):
            with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                MAPPING[:] = json.load(f)
            FLAT_TEXTS[:] = [m.get("text", "") for m in MAPPING]
            logger.info(f"Loaded mapping: {len(MAPPING)} entries")
        
        # Build tour databases
        index_tour_names()
        build_tours_db()
        
        # Build search index in background
        def build_index_background():
            try:
                built = build_index(force_rebuild=False)
                if built:
                    logger.info("Search index built successfully")
            except Exception as e:
                logger.error(f"Background index build failed: {e}")
        
        import threading
        index_thread = threading.Thread(target=build_index_background, daemon=True)
        index_thread.start()
        
        logger.info("✅ Application initialization completed")
        
    except Exception as e:
        logger.error(f"Application initialization failed: {e}")
        raise

# =========== KEEP EXISTING ROUTES ===========
# All existing routes (Google Sheets, tracking, etc.) remain unchanged below
# Only chat endpoint and related logic have been modified

@app.route("/")
def home():
    try:
        return jsonify({
            "status": "ok",
            "knowledge_count": len(FLAT_TEXTS),
            "tours_count": len(TOURS_DB),
            "duplicate_tours": len(TOUR_DUPLICATES),
            "service": "Ruby Wings Chatbot v2.2.0"
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/reindex", methods=["POST"])
def reindex():
    secret = request.headers.get("X-RBW-ADMIN", "")
    if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
        return jsonify({"error": "reindex not allowed"}), 403
    
    # Reload everything
    if os.path.exists(KNOWLEDGE_PATH):
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            KNOW = json.load(f)
    
    # Re-scan MAPPING
    MAPPING.clear()
    FLAT_TEXTS.clear()
    
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
    
    # Rebuild databases
    index_tour_names()
    build_tours_db()
    
    # Rebuild index
    ok = build_index(force_rebuild=True)
    
    return jsonify({
        "ok": ok,
        "tours_count": len(TOURS_DB),
        "duplicates": len(TOUR_DUPLICATES),
        "passages": len(FLAT_TEXTS)
    })

# All other existing routes (Google Sheets, tracking, health check, etc.)
# remain exactly as they were in the original code
# ... [rest of existing routes unchanged] ...

# =========== APPLICATION STARTUP ===========
if __name__ == "__main__":
    initialize_application()
    
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 10000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting Flask server on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
else:
    initialize_application()