# build_index.py
# Tạo FAISS index + metadata từ knowledge.json bằng OpenAI Embeddings
# Yêu cầu: openai, faiss, numpy

import os
import sys
import json
import time
import datetime
from typing import List
import numpy as np

# imports with helpful error messages
try:
    import faiss
except Exception:
    print("ERROR: Không import được faiss. Cài bằng: python -m pip install faiss-cpu", file=sys.stderr)
    raise

try:
    import openai
    from openai.error import OpenAIError
except Exception:
    print("ERROR: Không import được openai. Cài bằng: python -m pip install openai", file=sys.stderr)
    raise

# config
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    print("ERROR: OPENAI_API_KEY chưa đặt trong biến môi trường.", file=sys.stderr)
    sys.exit(1)
openai.api_key = OPENAI_KEY

KNOW = "knowledge.json"
INDEX_PATH = "index.faiss"
META_PATH = "index_metadata.json"
MODEL = "text-embedding-3-small"
BATCH_SIZE = 16
RETRY_LIMIT = 5
RETRY_BASE_DELAY = 1.0  # seconds

def load_docs(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} không tồn tại")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("knowledge.json phải là mảng (list) các document")
    return data

def texts_from_docs(docs: List[dict]):
    texts = []
    ids = []
    metas = []
    for i, d in enumerate(docs):
        # ưu tiên trường 'text' > 'description' > 'name' > serialize
        text = d.get("text") or d.get("description") or d.get("name") or ""
        if isinstance(text, (list, dict)):
            text = json.dumps(text, ensure_ascii=False)
        texts.append(text)
        ids.append(str(d.get("id", i)))
        metas.append(d)
    return texts, ids, metas

def call_embeddings(inputs: List[str], model: str):
    """Call OpenAI embedding with retries and exponential backoff."""
    attempt = 0
    while True:
        try:
            resp = openai.Embedding.create(model=model, input=inputs)
            return resp
        except OpenAIError as e:
            attempt += 1
            if attempt > RETRY_LIMIT:
                print(f"ERROR: Embedding failed after {RETRY_LIMIT} retries: {e}", file=sys.stderr)
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"Warning: Embedding request failed (attempt {attempt}/{RETRY_LIMIT}). Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)
        except Exception as e:
            attempt += 1
            if attempt > RETRY_LIMIT:
                print(f"ERROR: Unexpected error while fetching embeddings: {e}", file=sys.stderr)
                raise
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
            print(f"Warning: Unexpected error (attempt {attempt}/{RETRY_LIMIT}). Retrying in {delay:.1f}s...", file=sys.stderr)
            time.sleep(delay)

def get_embeddings(texts: List[str], model=MODEL, batch_size=BATCH_SIZE):
    all_emb = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = [t if (t and str(t).strip()) else " " for t in batch]
        resp = call_embeddings(inputs, model)
        emb_batch = [r["embedding"] for r in resp["data"]]
        all_emb.extend(emb_batch)
        # polite pause to avoid throttling
        time.sleep(0.1)
    arr = np.array(all_emb, dtype="float32")
    return arr

def build_faiss(embs: np.ndarray):
    if embs.size == 0:
        raise ValueError("Không có embedding để build index")
    dim = embs.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embs)
    return index

def write_metadata(ids, texts, metas, embeddings, model):
    faiss_version = getattr(faiss, "__version__", None) or getattr(faiss, "faiss_version", "")
    dimension = int(embeddings.shape[1]) if embeddings.size else 0
    mapping = { str(i): ids[i] for i in range(len(ids)) }

    documents = []
    for i, doc_id in enumerate(ids):
        documents.append({
            "id": doc_id,
            "text": texts[i] if texts[i] and str(texts[i]).strip() else ""
        })

    meta = {
        "documents": documents,
        "mapping": mapping,
        "total_documents": len(ids),
        "embedding_model": model,
        "dimension": dimension,
        "faiss_version": faiss_version,
        "created_at": datetime.datetime.utcnow().isoformat() + "Z",
        "docs_meta": metas
    }

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

def main():
    print("1) Load documents...", flush=True)
    docs = load_docs(KNOW)
    texts, ids, metas = texts_from_docs(docs)
    count_nonempty = sum(1 for t in texts if t and str(t).strip())
    print(f"  Tìm thấy {len(docs)} documents, trong đó {count_nonempty} có text.", flush=True)
    if count_nonempty == 0:
        print("ERROR: Không có văn bản để lấy embedding.", file=sys.stderr)
        sys.exit(1)

    print("2) Tạo embeddings (gửi tới OpenAI)...", flush=True)
    embeddings = get_embeddings(texts)

    print("3) Build FAISS index...", flush=True)
    index = build_faiss(embeddings)

    print(f"4) Ghi index ra: {INDEX_PATH}", flush=True)
    faiss.write_index(index, INDEX_PATH)

    print("5) Ghi metadata...", flush=True)
    write_metadata(ids, texts, metas, embeddings, MODEL)

    print("DONE: index created")
    print(f"- index: {INDEX_PATH}")
    print(f"- metadata: {META_PATH}")

if __name__ == "__main__":
    main()
