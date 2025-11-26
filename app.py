# app.py – Ruby Wings Chatbot Backend (Optimized for Render + RAG + FAISS)
import os
import json
import threading
import logging
from functools import lru_cache
from typing import List, Tuple

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import numpy as np

# OpenAI client (new standard)
from openai import OpenAI
client = OpenAI()

# Try FAISS
HAS_FAISS = False
try:
    import faiss
    HAS_FAISS = True
except:
    HAS_FAISS = False


# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw")


# -----------------------------------------------------
# Config
# -----------------------------------------------------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
FALLBACK_VECTORS_PATH = os.environ.get("FALLBACK_VECTORS_PATH", "vectors.npz")
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")

EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")
TOP_K = int(os.environ.get("TOP_K", "5"))

FAISS_ENABLED = os.environ.get("FAISS_ENABLED", "true").lower() in ("1", "true", "yes")


# -----------------------------------------------------
# Flask
# -----------------------------------------------------
app = Flask(__name__)
CORS(app)


# -----------------------------------------------------
# Global State
# -----------------------------------------------------
MAPPING: List[dict] = []
FLATTENED_TEXTS: List[str] = []
INDEX = None
INDEX_LOCK = threading.Lock()


# -----------------------------------------------------
# Fallback Index (numpy)
# -----------------------------------------------------
class NumpyIndex:
    def __init__(self, mat=None):
        self.mat = mat.astype("float32") if isinstance(mat, np.ndarray) else np.empty((0, 0), dtype="float32")
        self.dim = self.mat.shape[1] if self.mat.size else None

    @property
    def ntotal(self):
        return 0 if self.mat.size == 0 else self.mat.shape[0]

    def search(self, q, k):
        if self.mat.size == 0:
            return np.zeros((1, 0)), np.zeros((1, 0), dtype=int)

        q = q.astype("float32")
        q /= (np.linalg.norm(q) + 1e-12)

        mat_norm = self.mat / (np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12)
        sims = mat_norm @ q.T

        idx = np.argsort(-sims.reshape(-1))[:k]
        return sims[idx].reshape(1, -1), idx.reshape(1, -1)

    @staticmethod
    def load(path):
        try:
            arr = np.load(path)
            return NumpyIndex(arr["mat"])
        except:
            return NumpyIndex()


# -----------------------------------------------------
# Load knowledge.json → flatten
# -----------------------------------------------------
def load_knowledge():
    global MAPPING, FLATTENED_TEXTS
    try:
        data = json.load(open(KNOWLEDGE_PATH, "r", encoding="utf-8"))
    except:
        data = {}

    MAPPING = []
    FLATTENED_TEXTS = []

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
                MAPPING.append({"path": prefix, "text": t})
                FLATTENED_TEXTS.append(t)

    scan(data, "root")
    logger.info(f"Knowledge loaded: {len(FLATTENED_TEXTS)} passages")


# -----------------------------------------------------
# Embedding
# -----------------------------------------------------
@lru_cache(maxsize=8192)
def embed(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(1536, dtype="float32")

    try:
        res = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
        vec = res.data[0].embedding
        return np.array(vec, dtype="float32")
    except:
        # fallback synthetic
        h = abs(hash(text)) % (10 ** 12)
        dim = 1536
        vec = [(float((h >> (i % 32)) & 0xFF) + (i % 7)) / 255.0 for i in range(dim)]
        return np.array(vec, dtype="float32")


# -----------------------------------------------------
# Build / Load Index
# -----------------------------------------------------
def build_index(force=False):
    global INDEX, MAPPING

    with INDEX_LOCK:

        # load existing index
        if not force:
            if FAISS_ENABLED and HAS_FAISS and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    INDEX = faiss.read_index(FAISS_INDEX_PATH)
                    MAPPING[:] = json.load(open(FAISS_MAPPING_PATH))
                    logger.info("Loaded FAISS index")
                    return True
                except:
                    logger.warning("FAISS load failed → rebuild")

            if os.path.exists(FALLBACK_VECTORS_PATH) and os.path.exists(FAISS_MAPPING_PATH):
                try:
                    INDEX = NumpyIndex.load(FALLBACK_VECTORS_PATH)
                    MAPPING[:] = json.load(open(FAISS_MAPPING_PATH))
                    logger.info("Loaded fallback numpy index")
                    return True
                except:
                    logger.warning("Fallback index load failed → rebuild")

        # rebuild
        if not FLATTENED_TEXTS:
            load_knowledge()

        if not FLATTENED_TEXTS:
            logger.error("No texts to index")
            return False

        vectors = [embed(t) for t in FLATTENED_TEXTS]
        mat = np.vstack(vectors).astype("float32")
        mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12)

        if FAISS_ENABLED and HAS_FAISS:
            index = faiss.IndexFlatIP(mat.shape[1])
            index.add(mat)
            INDEX = index
            faiss.write_index(index, FAISS_INDEX_PATH)
        else:
            INDEX = NumpyIndex(mat)
            np.savez_compressed(FALLBACK_VECTORS_PATH, mat=mat)

        json.dump(MAPPING, open(FAISS_MAPPING_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=2)

        logger.info("Index built successfully")
        return True


# -----------------------------------------------------
# Query Index
# -----------------------------------------------------
def query_index(text: str, k=TOP_K):
    if INDEX is None:
        build_index()

    q = embed(text)
    q = q.reshape(1, -1)
    scores, idxs = INDEX.search(q, k)

    results = []
    for score, i in zip(scores[0], idxs[0]):
        if 0 <= i < len(MAPPING):
            results.append((float(score), MAPPING[i]))

    return results


# -----------------------------------------------------
# System Prompt Assembly
# -----------------------------------------------------
def build_system_prompt(top):
    header = (
        "Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn tour trải nghiệm, retreat, thiền, khí công "
        "và hành trình chữa lành. Trả lời ngắn gọn, chính xác, tử tế.\n\n"
    )

    if not top:
        return header + "Không tìm thấy dữ liệu nội bộ phù hợp."

    body = "Dữ liệu nội bộ liên quan:\n"
    for i, (score, m) in enumerate(top, 1):
        body += f"\n[{i}] (score={score:.3f}) — {m['path']}\n{m['text']}\n"

    body += (
        "\n---\nHướng dẫn:\n"
        "- Luôn ưu tiên dữ liệu nội bộ.\n"
        "- Không bịa thông tin.\n"
        "- Giữ văn phong lịch sự, thân thiện.\n"
    )
    return header + body


# -----------------------------------------------------
# API: Chat (non-stream)
# -----------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    msg = data.get("message", "").strip()

    if not msg:
        return jsonify({"reply": "Bạn chưa nhập câu hỏi."})

    top = query_index(msg)
    sys_prompt = build_system_prompt(top)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": msg},
    ]

    res = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=0.2,
        max_tokens=700
    )

    reply = res.choices[0].message.content

    return jsonify({"reply": reply, "sources": [m for _, m in top]})


# -----------------------------------------------------
# API: Streaming (SSE)
# -----------------------------------------------------
@app.route("/stream", methods=["POST"])
def stream():
    data = request.json or {}
    msg = data.get("message", "").strip()

    if not msg:
        return jsonify({"error": "empty"}), 400

    top = query_index(msg)
    sys_prompt = build_system_prompt(top)

    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": msg},
    ]

    def gen():
        stream = client.chat.completions.create(
            model=CHAT_MODEL,
            messages=messages,
            temperature=0.2,
            stream=True
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield f"data: {json.dumps({'delta': delta})}\n\n"

        yield f"data: {json.dumps({'done': True, 'sources': [m for _, m in top]})}\n\n"

    return Response(stream_with_context(gen()), mimetype='text/event-stream')


# -----------------------------------------------------
# Home
# -----------------------------------------------------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "passages": len(FLATTENED_TEXTS),
        "faiss": HAS_FAISS,
        "faiss_enabled": FAISS_ENABLED,
    })


# -----------------------------------------------------
# Init on startup
# -----------------------------------------------------
load_knowledge()
build_index(force=False)


# -----------------------------------------------------
# Run
# -----------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)))
