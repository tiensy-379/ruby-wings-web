#!/usr/bin/env python3
"""
helpers_bundle.py

Bundle of optional helper modules combined into one file for convenience:
- intent_classifier (detect_field)
- coref (resolve_pronoun)
- clarifier (need_clarify, build_clarify_question)
- session_store (Redis-backed load_session/save_session/get_redis)
- metrics (in-memory counters incr/get_metrics)
- synonyms loader (load_synonyms)
- test harness (run_test_harness)
Additionally includes a small utility to create shim module files so existing imports
(e.g., `import intent_classifier`) continue to work without editing app.py.

Usage:
1. Save this file at project root.
2. (Optional) create shims so app.py can import modules normally:
     python helpers_bundle.py --create-shims
   This will write small module files: intent_classifier.py, coref.py, clarifier.py,
   session_store.py, metrics.py that re-export functions from this bundle.
3. Run tests:
     python helpers_bundle.py --test-harness
"""

from typing import Tuple, Optional, Dict, Any, List
import os
import json
import re
import unicodedata
from collections import Counter
import sys

# ---------------------------
# INTENT CLASSIFIER
# ---------------------------
MANUAL_MAP_PATH = os.environ.get("INTENT_MAP_PATH", "intent_map.json")
try:
    with open(MANUAL_MAP_PATH, "r", encoding="utf-8") as f:
        INTENT_MAP = json.load(f)
except Exception:
    INTENT_MAP = {}

def _normalize(s: str) -> str:
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_field(text: str) -> Tuple[Optional[str], float]:
    """
    Return (field, confidence) in [0..1].
    Uses INTENT_MAP (if present) then fallback keyword rules.
    """
    if not text:
        return None, 0.0
    t = _normalize(text)

    # 1) manual map high-priority
    for key, spec in INTENT_MAP.items():
        for kw in spec.get("keywords", []):
            if _normalize(kw) in t:
                return spec.get("field"), 0.95

    # 2) small rule-based mapping (fast)
    generic = {
        "price": ["giá", "chi phí", "bao nhiêu", "cost", "price"],
        "duration": ["ngày", "đêm", "kéo dài", "mấy ngày", "duration"],
        "location": ["ở đâu", "đi đâu", "địa danh", "điểm đến", "location", "destination"],
        "booking_method": ["đặt", "đặt chỗ", "booking", "cách đặt"],
        "tour_name": ["tên tour", "các tour", "liệt kê tour", "list tour", "tours"],
        "hotline": ["hotline", "số điện thoại", "liên hệ", "contact"]
    }
    for fld, kws in generic.items():
        for kw in kws:
            if kw in t:
                conf = 0.75 if len(kw) > 3 else 0.65
                return fld, conf

    return None, 0.0

# ---------------------------
# COREFERENCE (coref)
# ---------------------------
PRONOUNS = ["nó", "đó", "cái đó", "này", "ấy", "còn"]

def _contains_pronoun(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    for p in PRONOUNS:
        if p in t:
            return True
    return False

def resolve_pronoun(text: str, session_entities: Dict[str, Any]) -> Dict[str, Any]:
    """
    Heuristic coreference resolver.
    Returns: {"resolved_entity": int|None, "resolved_text": text, "hints": {...}}
    session_entities expected keys: last_tour_index, recent_entities (list of dicts with 'type' and 'id').
    """
    out = {"resolved_entity": None, "resolved_text": text, "hints": {}}
    if not text:
        return out

    if _contains_pronoun(text):
        last = session_entities.get("last_tour_index")
        if last is not None:
            out["resolved_entity"] = last
            out["hints"]["reason"] = "last_tour_index"
            return out
        recent = session_entities.get("recent_entities") or []
        if recent:
            for ent in reversed(recent):
                if isinstance(ent, dict) and ent.get("type") == "tour" and ent.get("id") is not None:
                    out["resolved_entity"] = ent.get("id")
                    out["hints"]["reason"] = "recent_entities"
                    return out

    # Try simple regex to detect 'còn chỗ', 'còn vé'
    if re.search(r"\bcòn\b.*\b(chỗ|vé|slot|chỗ ngồi)\b", text.lower()):
        last = session_entities.get("last_tour_index")
        if last is not None:
            out["resolved_entity"] = last
            out["hints"]["reason"] = "booking_phrase_last_tour"
            return out

    return out

# ---------------------------
# CLARIFIER
# ---------------------------
def need_clarify(detected_field: Optional[str], tour_candidates: List[int], confidence: float) -> bool:
    """
    Return True when we should ask a clarifying question.
    """
    if not detected_field and len(tour_candidates) > 1:
        return True
    if detected_field and confidence < 0.6 and len(tour_candidates) > 1:
        return True
    return False

def build_clarify_question(tour_candidates: List[Dict]) -> str:
    """
    tour_candidates: list of {"tour_index":int,"tour_name":str}
    """
    if not tour_candidates:
        return "Bạn có thể nói rõ hơn được không?"
    pieces = []
    for i, t in enumerate(tour_candidates, start=1):
        name = t.get("tour_name") or f"Tour #{t.get('tour_index')}"
        pieces.append(f"{i}) {name}")
    return "Bạn đang nói đến tour nào? Chọn số hoặc gõ tên:\n" + " / ".join(pieces)

# ---------------------------
# SESSION STORE (Redis optional)
# ---------------------------
try:
    import redis as _redis  # optional dependency
except Exception:
    _redis = None

REDIS_URL = os.environ.get("REDIS_URL", "")
PREFIX = os.environ.get("RBW_SESSION_PREFIX", "rbw:session:")

def get_redis():
    if not _redis or not REDIS_URL:
        return None
    try:
        return _redis.from_url(REDIS_URL, decode_responses=True)
    except Exception:
        return None

def load_session(session_id: str):
    r = get_redis()
    if not r:
        return None
    try:
        data = r.get(PREFIX + session_id)
        if not data:
            return None
        return json.loads(data)
    except Exception:
        return None

def save_session(session_id: str, data: Dict, ttl: int = 300) -> bool:
    r = get_redis()
    if not r:
        return False
    try:
        r.setex(PREFIX + session_id, ttl, json.dumps(data, ensure_ascii=False))
        return True
    except Exception:
        return False

# ---------------------------
# METRICS
# ---------------------------
COUNTERS = Counter()

def incr(key: str, n: int = 1):
    try:
        COUNTERS[key] += n
    except Exception:
        pass

def get_metrics() -> Dict[str, int]:
    return dict(COUNTERS)

# ---------------------------
# SYNONYMS / MANUAL ALIASES
# ---------------------------
# Default inline synonyms (small). If file synonyms_manual.json exists, it will be loaded.
_DEFAULT_SYNONYMS = {
    "ho guom": ["hoguom", "hồ gươm", "ho guom", "ho-guom", "ho guom lake"],
    "phong nha": ["phong-nha", "phong nha ke bang", "phongnha"],
    "mua do": ["mua do", "mưa đỏ", "mua đỏ"]
}

def load_synonyms(path: str = "synonyms_manual.json") -> Dict[str, List[str]]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # normalize keys
                out = {}
                for k, vals in data.items():
                    out[_normalize(k)] = [ _normalize(v) for v in (vals or []) ]
                return out
    except Exception:
        pass
    # fallback to default
    return { _normalize(k): [ _normalize(v) for v in vals ] for k, vals in _DEFAULT_SYNONYMS.items() }

# ---------------------------
# TEST HARNESS (quick)
# ---------------------------
def run_test_harness(url: str = "http://localhost:10000/chat"):
    """
    Small test harness that sends a few queries to /chat and prints responses.
    Requires: `requests` package.
    """
    try:
        import requests
    except Exception:
        print("requests not installed. pip install requests")
        return
    tests = [
        {"message":"giá tour Mưa Đỏ"},
        {"message":"Tôi muốn đi Hồ Gươm"},
        {"message":"Tour Mưa Đỏ chỗ ở như thế nào?"},
        {"message":"còn chỗ không?"},
        {"message":"liệt kê các tour hiện có"}
    ]
    for t in tests:
        print("Q:", t["message"])
        try:
            r = requests.post(url, json=t, timeout=10)
            print("Status:", r.status_code)
            try:
                print(json.dumps(r.json(), ensure_ascii=False, indent=2))
            except Exception:
                print("Non-JSON response:", r.text)
        except Exception as e:
            print("Request failed:", e)
        print("-" * 60)

# ---------------------------
# SHIM CREATION (create small module files that re-export functions)
# ---------------------------
_SHIM_FILES = {
    "intent_classifier.py": "from helpers_bundle import detect_field\n",
    "coref.py": "from helpers_bundle import resolve_pronoun\n",
    "clarifier.py": "from helpers_bundle import need_clarify, build_clarify_question\n",
    "session_store.py": "from helpers_bundle import load_session, save_session, get_redis\n",
    "metrics.py": "from helpers_bundle import incr, get_metrics\n"
}

def create_shims(target_dir: str = "."):
    for fname, content in _SHIM_FILES.items():
        path = os.path.join(target_dir, fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            print("Wrote shim:", path)
        except Exception as e:
            print("Failed to write shim", path, ":", e)

# ---------------------------
# CLI
# ---------------------------
def _print_help():
    print("helpers_bundle.py CLI")
    print("  --create-shims   : create small shim module files in current directory")
    print("  --test-harness   : run quick local test harness against http://localhost:10000/chat")
    print("  --show-synonyms  : print loaded synonyms")
    print("  --help           : this message")

if __name__ == "__main__":
    if "--create-shims" in sys.argv:
        create_shims(".")
        print("Shims created. You can now import modules like intent_classifier, coref, etc.")
        sys.exit(0)
    if "--test-harness" in sys.argv:
        url = "http://localhost:10000/chat"
        # allow custom url
        for i, a in enumerate(sys.argv):
            if a == "--test-harness" and len(sys.argv) > i+1:
                url = sys.argv[i+1]
        run_test_harness(url)
        sys.exit(0)
    if "--show-synonyms" in sys.argv:
        s = load_synonyms()
        print(json.dumps(s, ensure_ascii=False, indent=2))
        sys.exit(0)
    _print_help()