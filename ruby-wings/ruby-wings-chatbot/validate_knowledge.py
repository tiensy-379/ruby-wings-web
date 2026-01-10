#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
validate_knowledge.py
Chuẩn hóa knowledge.json thành knowledge.normalized.json
Đảm bảo mỗi tour đều có đủ 16 trường theo chuẩn app.py
"""

import json
import sys
from pathlib import Path

INPUT_FILE = "knowledge.json"
OUTPUT_FILE = "knowledge.normalized.json"

# 16 TRƯỜNG CHUẨN HÓA (theo app.py + build_index.py + tiêu chuẩn bạn yêu cầu)
REQUIRED_FIELDS = [
    "tour_name",
    "summary",
    "location",
    "duration",
    "price",
    "includes",
    "notes",
    "style",
    "transport",
    "accommodation",
    "meals",
    "event_support",
    "hotline",
    "additional",         # nếu không có thì thêm chuỗi rỗng
    "mission",            # dành cho retreat / nội dung văn hoá
    "includes_extra"      # dành cho phần mở rộng (retreat / corporate)
]

def load_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[ERROR] Không thể đọc {path}: {e}")
        sys.exit(1)


def normalize_tour(tour):
    """
    Bổ sung các trường còn thiếu.
    Giữ nguyên nội dung gốc nếu có.
    """
    normalized = {}

    for field in REQUIRED_FIELDS:
        if field in tour:
            normalized[field] = tour[field]
        else:
            # giá trị mặc định rỗng
            normalized[field] = "" if field not in ["includes"] else []

    return normalized


def normalize_knowledge(data):
    if "tours" not in data:
        print("[ERROR] File knowledge.json không có trường 'tours'")
        sys.exit(1)

    normalized_list = []

    for tour in data["tours"]:
        normalized_list.append(normalize_tour(tour))

    return {"tours": normalized_list}


def save_json(path, data):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[OK] Đã tạo file chuẩn hóa: {path}")
    except Exception as e:
        print(f"[ERROR] Không thể ghi file {path}: {e}")
        sys.exit(1)


def main():
    print("=== Ruby Wings – Knowledge Validator ===")

    raw = load_json(INPUT_FILE)
    normalized = normalize_knowledge(raw)
    save_json(OUTPUT_FILE, normalized)

    print("[DONE] knowledge.json → knowledge.normalized.json đã sẵn sàng cho build_index.py & app.py")


if __name__ == "__main__":
    main()
