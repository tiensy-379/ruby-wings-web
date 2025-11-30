#!/usr/bin/env python3
"""
common_utils.py - shared helpers to keep flattening and normalization consistent
Used by: app.py, build_index.py, entities.py
"""

import json
import os
import re
import unicodedata
from typing import Any, List, Dict

def normalize_text_simple(s: str) -> str:
    """Lowercase, remove diacritics, strip punctuation, collapse spaces."""
    if not s:
        return ""
    s = s.lower()
    s = unicodedata.normalize("NFD", s)
    s = "".join(ch for ch in s if unicodedata.category(ch) != "Mn")
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def flatten_json(path: str) -> List[Dict]:
    """
    Flatten a JSON file into list of {"path": prefix, "text": text}
    Preserves traversal order. Safe for values that are dict/list/str/scalar.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: List[Dict] = []

    def scan(obj: Any, prefix: str = "root"):
        if isinstance(obj, dict):
            for k, v in obj.items():
                scan(v, f"{prefix}.{k}")
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{prefix}[{i}]")
        elif isinstance(obj, str):
            t = obj.strip()
            if t:
                mapping.append({"path": prefix, "text": t})
        else:
            try:
                s = str(obj).strip()
                if s:
                    mapping.append({"path": prefix, "text": s})
            except Exception:
                pass

    scan(data, "root")
    return mapping