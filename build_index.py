#!/usr/bin/env python3
# build_index.py — Bản TỐI ƯU NGỮ CẢNH TOUR (v2.0)
# Tương thích 100% với knowledge.json mới, ưu tiên context tour
# Xuất: faiss_index.bin, vectors.npz, faiss_mapping.json

import os, json, time, sys
import numpy as np
from typing import List, Dict, Optional, Tuple

# Try FAISS
try:
    import faiss
    HAS_FAISS = True
except Exception:
    faiss = None
    HAS_FAISS = False

# OpenAI API mới
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------- CONFIG ----------
OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FIELD_KEYWORDS_PATH = os.environ.get("FIELD_KEYWORDS_PATH", "field_keywords.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
META_PATH = os.environ.get("META_PATH", "faiss_meta.json")
EMBED_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
BATCH = int(os.environ.get("BUILD_BATCH_SIZE", "32"))

# Tự động nhận diện trường từ knowledge.json
def detect_canonical_fields(data: dict) -> List[str]:
    """Tự động phát hiện các trường từ cấu trúc knowledge.json"""
    fields = set()
    
    # Thêm các trường từ about_company
    if "about_company" in data:
        for key in data["about_company"].keys():
            fields.add(f"about_company.{key}")
    
    # Thêm các trường từ tours
    if "tours" in data and isinstance(data["tours"], list):
        for tour in data["tours"]:
            if isinstance(tour, dict):
                for key in tour.keys():
                    fields.add(key)
    
    # Thêm các trường từ faq
    if "faq" in data:
        for key in data["faq"].keys():
            fields.add(f"faq.{key}")
    
    # Thêm các trường từ contact
    if "contact" in data:
        for key in data["contact"].keys():
            fields.add(f"contact.{key}")
    
    return sorted(list(fields))

def synthetic_embedding(text: str, dim: int = 1536):
    """Fallback embedding khi không có API"""
    h = abs(hash(text)) % (10**12)
    return [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]

def embed_batch(texts: List[str], model: str):
    """Batch embed với fallback khi không có API"""
    if not OPENAI_KEY or OpenAI is None:
        dim = 1536 if "3-small" in model else 3072
        return [synthetic_embedding(t, dim) for t in texts]
    
    client = OpenAI(api_key=OPENAI_KEY)
    try:
        response = client.embeddings.create(model=model, input=texts)
        return [data.embedding for data in response.data]
    except Exception as e:
        print(f"⚠️ OpenAI embedding failed, using fallback: {e}")
        dim = 1536 if "3-small" in model else 3072
        return [synthetic_embedding(t, dim) for t in texts]

# ---------- ENHANCED FLATTEN KNOWLEDGE ----------
def flatten_knowledge() -> List[dict]:
    """Flatten knowledge.json với thông tin ngữ cảnh phong phú cho chatbot"""
    if not os.path.exists(KNOWLEDGE_PATH):
        raise FileNotFoundError(f"{KNOWLEDGE_PATH} không tồn tại")
    
    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    mapping = []
    
    # Load field keywords để hỗ trợ mapping chính xác
    field_keywords = {}
    if os.path.exists(FIELD_KEYWORDS_PATH):
        with open(FIELD_KEYWORDS_PATH, "r", encoding="utf-8") as f:
            field_keywords = json.load(f)
    
    def scan(obj, path="root", context: dict = None):
        """Quét đệ quy với ngữ cảnh tour"""
        if context is None:
            context = {"current_tour_index": None, "current_tour_name": None}
        
        if isinstance(obj, dict):
            # Kiểm tra xem có phải là một tour không
            if "tour_name" in obj and isinstance(obj["tour_name"], str):
                # Đây là một tour
                tour_index = len([m for m in mapping if m.get("is_tour")]) if "tours" in path else 0
                new_context = {
                    "current_tour_index": tour_index,
                    "current_tour_name": obj["tour_name"]
                }
                for key, value in obj.items():
                    scan(value, f"{path}.{key}", new_context)
            else:
                # Không phải tour, giữ nguyên context
                for key, value in obj.items():
                    scan(value, f"{path}.{key}", context)
        
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                # Nếu path chứa "tours", đây có thể là danh sách tours
                if "tours" in path and isinstance(item, dict) and "tour_name" in item:
                    tour_index = i
                    new_context = {
                        "current_tour_index": tour_index,
                        "current_tour_name": item.get("tour_name")
                    }
                    scan(item, f"{path}[{i}]", new_context)
                else:
                    scan(item, f"{path}[{i}]", context)
        
        elif isinstance(obj, str):
            text = obj.strip()
            if text:
                # Xác định field từ path
                field = path.split(".")[-1].split("[")[0]
                
                # Chuẩn hóa field dựa trên field_keywords
                normalized_field = field
                for main_field, keywords in field_keywords.items():
                    if field in keywords or any(field in kw for kw in keywords):
                        normalized_field = main_field.split(".")[-1] if "." in main_field else main_field
                        break
                
                # Tạo passage với metadata phong phú
                passage = {
                    "path": path,
                    "text": text,
                    "field": normalized_field,
                    "original_field": field,
                    "tour_index": context["current_tour_index"],
                    "tour_name": context["current_tour_name"],
                    "is_tour": context["current_tour_index"] is not None,
                    "context_score": 1.0 if context["current_tour_index"] is not None else 0.5,
                    "search_keywords": [],
                    "is_core_info": field in ["tour_name", "summary", "price", "duration"]
                }
                
                # Thêm từ khóa tìm kiếm từ field_keywords
                if normalized_field in field_keywords:
                    passage["search_keywords"] = field_keywords[normalized_field]
                
                # Thêm context text để cải thiện tìm kiếm ngữ cảnh
                if context["current_tour_name"]:
                    passage["context_text"] = f"{context['current_tour_name']} {text}"
                else:
                    passage["context_text"] = text
                
                mapping.append(passage)
        
        else:
            # Xử lý các kiểu dữ liệu khác (số, boolean)
            try:
                text = str(obj).strip()
                if text:
                    field = path.split(".")[-1].split("[")[0]
                    
                    passage = {
                        "path": path,
                        "text": text,
                        "field": field,
                        "original_field": field,
                        "tour_index": context["current_tour_index"],
                        "tour_name": context["current_tour_name"],
                        "is_tour": context["current_tour_index"] is not None,
                        "context_score": 1.0 if context["current_tour_index"] is not None else 0.5,
                        "search_keywords": [],
                        "is_core_info": False
                    }
                    
                    if field in field_keywords:
                        passage["search_keywords"] = field_keywords[field]
                    
                    mapping.append(passage)
            except:
                pass
    
    scan(data)
    
    # Thêm các passages đặc biệt để cải thiện tìm kiếm ngữ cảnh
    enhanced_mapping = []
    for passage in mapping:
        # Thêm passage gốc
        enhanced_mapping.append(passage)
        
        # Tạo passage tìm kiếm đa ngữ cảnh cho các tour
        if passage["is_tour"] and passage["tour_name"]:
            # Passage với tên tour + nội dung (cho tìm kiếm theo context)
            enhanced_passage = passage.copy()
            enhanced_passage["text"] = f"{passage['tour_name']}: {passage['text']}"
            enhanced_passage["context_score"] = 1.2  # Tăng điểm cho context rõ ràng
            enhanced_mapping.append(enhanced_passage)
            
            # Passage chỉ với tên tour cho field quan trọng
            if passage["is_core_info"]:
                tour_only_passage = passage.copy()
                tour_only_passage["text"] = passage["tour_name"]
                tour_only_passage["field"] = "tour_name_context"
                tour_only_passage["context_score"] = 1.5  # Điểm rất cao cho tên tour
                enhanced_mapping.append(tour_only_passage)
    
    print(f"📊 Thống kê mapping:")
    print(f"  - Tổng passages: {len(enhanced_mapping)}")
    print(f"  - Passages tour: {len([m for m in enhanced_mapping if m['is_tour']])}")
    print(f"  - Passages công ty: {len([m for m in enhanced_mapping if not m['is_tour']])}")
    print(f"  - Số tour duy nhất: {len(set([m['tour_name'] for m in enhanced_mapping if m['tour_name']]))}")
    
    return enhanced_mapping

# ---------- BUILD ENHANCED INDEX ----------
def build_enhanced_index():
    print("🚀 Bắt đầu xây dựng index nâng cao...")
    print("📖 Đọc và xử lý knowledge.json...")
    
    mapping = flatten_knowledge()
    
    if not mapping:
        raise RuntimeError("Không có dữ liệu nào để index - knowledge.json có thể trống")
    
    # Chuẩn bị texts cho embedding
    texts = []
    metadata = []
    
    for passage in mapping:
        # Ưu tiên sử dụng context_text nếu có
        text_to_embed = passage.get("context_text", passage["text"])
        texts.append(text_to_embed)
        metadata.append({
            "original_text": passage["text"],
            "field": passage["field"],
            "tour_index": passage["tour_index"],
            "tour_name": passage["tour_name"],
            "is_tour": passage["is_tour"],
            "context_score": passage["context_score"],
            "path": passage["path"],
            "is_core_info": passage.get("is_core_info", False)
        })
    
    n = len(texts)
    print(f"✅ Đã tạo {n} passages cho embedding")
    
    # Tạo embeddings theo batch
    print(f"🧠 Tạo embeddings sử dụng model: {EMBED_MODEL}")
    vectors = []
    
    for i in range(0, n, BATCH):
        batch_texts = texts[i:i+BATCH]
        batch_embeddings = embed_batch(batch_texts, EMBED_MODEL)
        vectors.extend(batch_embeddings)
        
        if (i // BATCH) % 5 == 0 or i + BATCH >= n:
            print(f"  ✅ Đã xử lý {len(vectors)}/{n} passages")
    
    # Chuyển thành numpy array
    matrix = np.array(vectors, dtype="float32")
    
    # Chuẩn hóa vectors cho cosine similarity
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / (norms + 1e-12)
    
    print(f"📐 Kích thước vector: {matrix.shape}")
    
    # Lưu vectors dự phòng
    np.savez_compressed(FALLBACK_VECTORS_PATH, matrix=matrix)
    print(f"💾 Đã lưu vectors dự phòng: {FALLBACK_VECTORS_PATH}")
    
    # Lưu mapping với metadata đầy đủ
    with open(FAISS_MAPPING_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "mapping": mapping,
            "metadata": metadata,
            "total_passages": n,
            "tour_count": len(set([m["tour_name"] for m in mapping if m["tour_name"]])),
            "fields": list(set([m["field"] for m in mapping]))
        }, f, ensure_ascii=False, indent=2)
    print(f"💾 Đã lưu mapping metadata: {FAISS_MAPPING_PATH}")
    
    # Xây dựng FAISS index nếu có
    if HAS_FAISS:
        dim = matrix.shape[1]
        print(f"🔨 Đang xây dựng FAISS index (dim={dim})...")
        
        # Sử dụng IndexFlatIP cho cosine similarity
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        
        faiss.write_index(index, FAISS_INDEX_PATH)
        print(f"💾 Đã lưu FAISS index: {FAISS_INDEX_PATH}")
    else:
        print("⚠️ FAISS không khả dụng, chỉ lưu vectors thô")
    
    # Lưu metadata hệ thống
    meta_info = {
        "created_at": time.time(),
        "created_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_passages": int(n),
        "dimension": int(matrix.shape[1]),
        "embedding_model": EMBED_MODEL,
        "has_faiss": HAS_FAISS,
        "tour_specific_passages": len([m for m in mapping if m["is_tour"]]),
        "company_passages": len([m for m in mapping if not m["is_tour"]]),
        "context_enhanced": True,
        "version": "2.0-tour-context-optimized"
    }
    
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta_info, f, indent=2)
    print(f"💾 Đã lưu metadata: {META_PATH}")
    
    # Tạo file tóm tắt
    summary = {
        "build_completed": True,
        "timestamp": time.time(),
        "statistics": {
            "total_passages": n,
            "tours_count": meta_info["tour_specific_passages"],
            "unique_tours": meta_info["tour_specific_passages"] // 10,  # Ước lượng
            "embedding_dimension": matrix.shape[1],
            "file_sizes": {
                "knowledge.json": os.path.getsize(KNOWLEDGE_PATH) if os.path.exists(KNOWLEDGE_PATH) else 0,
                "vectors.npz": os.path.getsize(FALLBACK_VECTORS_PATH) if os.path.exists(FALLBACK_VECTORS_PATH) else 0,
                "mapping.json": os.path.getsize(FAISS_MAPPING_PATH) if os.path.exists(FAISS_MAPPING_PATH) else 0
            }
        }
    }
    
    print("\n" + "="*60)
    print("🎉 XÂY DỰNG INDEX HOÀN TẤT!")
    print("="*60)
    print(f"📊 Thống kê cuối cùng:")
    print(f"  • Tổng số passages: {n}")
    print(f"  • Passages thuộc tour: {meta_info['tour_specific_passages']}")
    print(f"  • Passages thông tin công ty: {meta_info['company_passages']}")
    print(f"  • Chiều không gian embedding: {matrix.shape[1]}")
    print(f"  • Hỗ trợ FAISS: {'Có' if HAS_FAISS else 'Không'}")
    print(f"  • Ưu tiên ngữ cảnh tour: Có")
    print(f"\n📁 Các file đã tạo:")
    print(f"  • {FAISS_MAPPING_PATH}")
    print(f"  • {FALLBACK_VECTORS_PATH}")
    if HAS_FAISS:
        print(f"  • {FAISS_INDEX_PATH}")
    print(f"  • {META_PATH}")
    print("="*60)
    
    return True

def validate_index():
    """Kiểm tra index sau khi xây dựng"""
    print("\n🔍 Kiểm tra chất lượng index...")
    
    files_to_check = [
        (FAISS_MAPPING_PATH, "Mapping file"),
        (FALLBACK_VECTORS_PATH, "Vectors file"),
        (META_PATH, "Metadata file")
    ]
    
    if HAS_FAISS:
        files_to_check.append((FAISS_INDEX_PATH, "FAISS index file"))
    
    all_ok = True
    for file_path, description in files_to_check:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✅ {description}: {file_path} ({size:,} bytes)")
        else:
            print(f"  ❌ {description}: {file_path} - KHÔNG TỒN TẠI")
            all_ok = False
    
    if all_ok:
        print("✅ Tất cả file index đã được tạo thành công!")
        
        # Kiểm tra cấu trúc mapping
        try:
            with open(FAISS_MAPPING_PATH, "r", encoding="utf-8") as f:
                mapping_data = json.load(f)
            
            if "mapping" in mapping_data:
                tour_passages = [m for m in mapping_data["mapping"] if m.get("is_tour")]
                print(f"  • {len(tour_passages)} passages có thông tin tour")
                print(f"  • {len(set([m.get('tour_name') for m in tour_passages if m.get('tour_name')]))} tour duy nhất")
        except Exception as e:
            print(f"  ⚠️ Không thể kiểm tra chi tiết mapping: {e}")
    else:
        print("⚠️ Có vấn đề với một số file index")
    
    return all_ok

if __name__ == "__main__":
    print("="*60)
    print("BUILD INDEX - RUBY WINGS TOUR CHATBOT")
    print("="*60)
    
    try:
        start_time = time.time()
        
        # Xây dựng index
        build_enhanced_index()
        
        # Kiểm tra
        validate_index()
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"\n⏱️ Thời gian thực hiện: {elapsed:.2f} giây")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f"\n❌ LỖI: {e}")
        print("Vui lòng kiểm tra file knowledge.json và field_keywords.json")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ LỖI KHÔNG XÁC ĐỊNH: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)