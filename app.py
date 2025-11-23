# app.py
import os
import json
import time
import threading
import traceback
import logging
from functools import lru_cache
from typing import List, Tuple

from flask import Flask, request, jsonify, Response, stream_with_context
from flask_cors import CORS
import openai
import faiss
import numpy as np

# ---------- Logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rbw")

# ---------- Config ----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
KNOWLEDGE_PATH = os.environ.get("KNOWLEDGE_PATH", "knowledge.json")
FAISS_INDEX_PATH = os.environ.get("FAISS_INDEX_PATH", "faiss_index.bin")
FAISS_MAPPING_PATH = os.environ.get("FAISS_MAPPING_PATH", "faiss_mapping.json")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small")  # dims 1536
CHAT_MODEL = os.environ.get("CHAT_MODEL", "gpt-4o-mini")  # change as needed
TOP_K = int(os.environ.get("TOP_K", "5"))

if not OPENAI_API_KEY:
    logger.warning("OPENAI_API_KEY is missing. Set environment variable OPENAI_API_KEY.")
openai.api_key = OPENAI_API_KEY
# if custom base url is used
try:
    openai.api_base = OPENAI_BASE_URL
except Exception:
    # some OpenAI SDK versions may not accept api_base assignment; ignore if so
    pass

# ---------- Flask ----------
app = Flask(__name__)
CORS(app)

# ---------- Global state ----------
KNOWLEDGE = {}
FLATTENED_TEXTS: List[str] = []
# mapping: index -> {"source":..., "path":..., "text":...}
MAPPING: List[dict] = []
INDEX_LOCK = threading.Lock()
FAISS_INDEX = None  # will be faiss.IndexFlatIP or IndexIDMap

# ---------- Utilities ----------
def load_knowledge(path=KNOWLEDGE_PATH):
    global KNOWLEDGE, FLATTENED_TEXTS, MAPPING
    try:
        with open(path, "r", encoding="utf-8") as f:
            KNOWLEDGE = json.load(f)
        # flatten: collect small text chunks (strings) with some path label
        FLATTENED_TEXTS = []
        MAPPING = []

        def scan(obj, prefix="root"):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    scan(v, prefix + "." + k)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    scan(item, f"{prefix}[{i}]")
            elif isinstance(obj, str):
                text = obj.strip()
                if len(text) > 20:  # keep reasonably sized passages
                    FLATTENED_TEXTS.append(text)
                    MAPPING.append({"path": prefix, "text": text})
        scan(KNOWLEDGE, "root")
        logger.info("✅ knowledge loaded: %d passages", len(FLATTENED_TEXTS))
    except Exception as e:
        logger.exception("⚠️ Could not load knowledge.json:")
        KNOWLEDGE = {}
        FLATTENED_TEXTS = []
        MAPPING = []


def save_mapping(path=FAISS_MAPPING_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(MAPPING, f, ensure_ascii=False, indent=2)
        logger.info("✅ Saved mapping.")
    except Exception as e:
        logger.exception("⚠️ Save mapping failed:")


def load_mapping(path=FAISS_MAPPING_PATH):
    global MAPPING
    try:
        with open(path, "r", encoding="utf-8") as f:
            MAPPING = json.load(f)
        logger.info("✅ Loaded mapping.")
    except Exception as e:
        logger.exception("⚠️ Could not load mapping:")
        MAPPING = []


# ---------- Embedding helpers ----------
@lru_cache(maxsize=4096)
def embed_text(text: str) -> Tuple[List[float], int]:
    """
    Returns embedding vector (list) and dimension.
    Cached via lru_cache for repeated usage in same process.
    Supports both old and new OpenAI SDKs via try/except.
    """
    if not text:
        return [], 0
    txt = text if len(text) < 2000 else text[:2000]  # guard length
    # call OpenAI embeddings (try old API first, then new)
    try:
        # Old style: openai.Embedding.create(...)
        resp = None
        try:
            resp = openai.Embedding.create(model=EMBEDDING_MODEL, input=txt)
        except Exception as e_old:
            # try new-style (openai >=1.0.0)
            try:
                # new API: openai.embeddings.create(...) or openai.Embeddings.create depends on version
                resp = getattr(openai, "embeddings", None)
                if resp and hasattr(resp, "create"):
                    resp = resp.create(model=EMBEDDING_MODEL, input=txt)
                else:
                    # try top-level new client style (openai.ChatCompletion may be v2 client)
                    # Some installs expose openai.Embeddings.create
                    resp = getattr(openai, "Embeddings", None)
                    if resp and hasattr(resp, "create"):
                        resp = resp.create(model=EMBEDDING_MODEL, input=txt)
                    else:
                        raise e_old
            except Exception:
                raise e_old
        # extract embedding robustly
        if isinstance(resp, dict) and "data" in resp and len(resp["data"]) > 0:
            emb = resp["data"][0].get("embedding") or resp["data"][0].get("embedding", None)
            if emb:
                return emb, len(emb)
        # some clients return objects with .data
        if hasattr(resp, "data") and len(resp.data) > 0:
            emb = getattr(resp.data[0], "embedding", None)
            if emb:
                return emb, len(emb)
        # fallback if embedding not present
        logger.warning("Embedding response did not contain embedding (resp type %s)", type(resp))
    except Exception as e:
        logger.exception("Embedding error:")
    # fallback: deterministic synthetic vector
    try:
        h = abs(hash(txt)) % (10 ** 8)
        dim = 1536
        vec = [float((h >> (i % 32)) & 0xFF) / 255.0 for i in range(dim)]
        return vec, dim
    except Exception as ex:
        logger.exception("Fallback embedding generation failed")
        return [], 0


def build_faiss_index(force_rebuild=False):
    """
    Build FAISS index for FLATTENED_TEXTS and persist to disk.
    Safe to call multiple times; locked to avoid races.
    Returns True if index is ready.
    """
    global FAISS_INDEX, MAPPING, FLATTENED_TEXTS
    with INDEX_LOCK:
        # if index exists on disk and not forcing, try load
        if not force_rebuild and os.path.exists(FAISS_INDEX_PATH) and os.path.exists(FAISS_MAPPING_PATH):
            try:
                idx = faiss.read_index(FAISS_INDEX_PATH)
                load_mapping(FAISS_MAPPING_PATH)
                FAISS_INDEX = idx
                logger.info("✅ FAISS index loaded from disk.")
                return True
            except Exception as e:
                logger.exception("⚠️ Could not load FAISS index from disk:")
        # need to build
        if not FLATTENED_TEXTS:
            logger.warning("⚠️ No texts to index.")
            FAISS_INDEX = None
            return False
        logger.info("🔧 Building FAISS index ... (passages=%d)", len(FLATTENED_TEXTS))
        vectors = []
        dims = None
        for t in FLATTENED_TEXTS:
            emb, d = embed_text(t)
            if not emb:
                continue
            if dims is None:
                dims = d
            vectors.append(np.array(emb, dtype="float32"))
        if not vectors:
            logger.warning("⚠️ No vectors created.")
            FAISS_INDEX = None
            return False
        mat = np.vstack(vectors)
        # normalize for cos similarity
        try:
            faiss.normalize_L2(mat)
            index = faiss.IndexFlatIP(dims)  # inner product on normalized vectors => cosine
            index.add(mat)
            FAISS_INDEX = index
            # persist
            try:
                faiss.write_index(FAISS_INDEX, FAISS_INDEX_PATH)
                save_mapping()
                logger.info("✅ FAISS index built and saved.")
            except Exception as e:
                logger.exception("⚠️ Could not save FAISS index:")
            return True
        except Exception as e:
            logger.exception("⚠️ Error building faiss index:")
            FAISS_INDEX = None
            return False


def query_faiss(query: str, top_k=TOP_K) -> List[Tuple[float, dict]]:
    """
    Return list of (score, mapping) for top_k similar passages.
    Safe: if FAISS_INDEX is None, returns [].
    """
    global FAISS_INDEX
    if not query:
        return []
    if FAISS_INDEX is None:
        # attempt to build lazily, but guard afterwards
        built = build_faiss_index(force_rebuild=False)
        if not built or FAISS_INDEX is None:
            logger.warning("query_faiss: FAISS_INDEX not ready - returning empty results")
            return []
    emb, d = embed_text(query)
    if not emb:
        return []
    try:
        vec = np.array(emb, dtype="float32").reshape(1, -1)
        faiss.normalize_L2(vec)
        D, I = FAISS_INDEX.search(vec, top_k)
    except Exception as e:
        logger.exception("Error searching FAISS index:")
        return []
    results = []
    for score, idx in zip(D[0].tolist(), I[0].tolist()):
        if idx < 0 or idx >= len(MAPPING):
            continue
        results.append((float(score), MAPPING[idx]))
    return results


# ---------- Prompt composition ----------
def compose_system_prompt(top_passages: List[Tuple[float, dict]]) -> str:
    header = (
        "Bạn là trợ lý AI của Ruby Wings — chuyên tư vấn du lịch trải nghiệm, retreat, "
        "thiền, khí công và các hành trình chữa lành. Hãy trả lời ngắn gọn, thân thiện và chính xác.\n"
        "Ưu tiên trích dẫn thông tin từ nguồn nội bộ dưới đây. Nếu không đủ, hãy trả kiến thức chung chính xác.\n\n"
    )
    if not top_passages:
        return header
    content = header + "Dữ liệu nội bộ liên quan (sắp xếp theo độ liên quan):\n"
    for i, (score, m) in enumerate(top_passages, start=1):
        content += f"\n[{i}] (score={score:.3f}) nguồn: {m.get('path','?')}\n{m.get('text','')}\n"
    content += "\n---\nHướng dẫn: Dùng dữ liệu nội bộ trên nếu trả lời liên quan; trích dẫn nguồn nếu cần."
    return content


# ---------- Endpoints ----------
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "Ruby Wings FAISS backend running.",
        "knowledge_count": len(FLATTENED_TEXTS),
        "faiss_exists": FAISS_INDEX is not None
    })


@app.route("/reindex", methods=["POST"])
def reindex():
    """
    Safe endpoint to rebuild index. Use when knowledge.json updated.
    """
    try:
        secret = request.headers.get("X-RBW-ADMIN", "")
        # very simple guard: if admin header not provided, disallow in public
        if not secret and os.environ.get("RBW_ALLOW_REINDEX", "") != "1":
            return jsonify({"error": "reindex not allowed without admin header or RBW_ALLOW_REINDEX=1"}), 403
        load_knowledge(KNOWLEDGE_PATH)
        ok = build_faiss_index(force_rebuild=True)
        return jsonify({"ok": ok, "count": len(FLATTENED_TEXTS)})
    except Exception as e:
        logger.exception("Unhandled error in /reindex")
        return jsonify({"error": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    """
    Non-streaming chat: returns assistant reply (single response).
    Input JSON: { "message": "...", "max_tokens": 700, "top_k": 5 }
    """
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"reply": "Bạn chưa nhập câu hỏi."})
        top_k = int(data.get("top_k", TOP_K))
        top = query_faiss(user_message, top_k)
        system_prompt = compose_system_prompt(top)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        logger.info("CHAT request: model=%s top_k=%d msg_len=%d faiss_hits=%d", CHAT_MODEL, top_k, len(user_message), len(top))
        logger.debug("System prompt (truncated): %s", system_prompt[:1000])

        # call OpenAI ChatCompletion - try multiple client styles for compatibility
        resp = None
        try:
            resp = openai.ChatCompletion.create(
                model=CHAT_MODEL,
                messages=messages,
                temperature=0.2,
                max_tokens=int(data.get("max_tokens", 700)),
                top_p=0.95
            )
        except Exception as e_old:
            logger.warning("openai.ChatCompletion.create failed, trying alternate client styles: %s", e_old)
            try:
                # new style: openai.chat.completions.create or openai.ChatCompletion.create may differ
                # Try openai.ChatCompletion.create was already attempted; try attribute chat if exists
                if hasattr(openai, "ChatCompletion"):
                    resp = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages)
                elif hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
                    resp = openai.chat.completions.create(model=CHAT_MODEL, messages=messages)
                else:
                    raise e_old
            except Exception as e2:
                logger.exception("All attempts to call OpenAI Chat failed")
                return jsonify({"error": "OpenAI chat request failed", "detail": str(e2)}), 500

        content = ""
        try:
            # robust extraction
            if isinstance(resp, dict):
                choices = resp.get("choices")
                if choices and isinstance(choices, list) and len(choices) > 0:
                    first = choices[0]
                    # different SDKs have different nesting
                    if isinstance(first.get("message"), dict):
                        content = first["message"].get("content", "")
                    elif "text" in first:
                        content = first.get("text", "")
                    else:
                        content = str(first)
                else:
                    content = str(resp)
            else:
                # some SDK returns objects with .choices
                choices = getattr(resp, "choices", None)
                if choices and len(choices) > 0:
                    first = choices[0]
                    msg = getattr(first, "message", None)
                    if msg:
                        content = getattr(msg, "get", lambda k, d=None: "")("content", "")
                    else:
                        content = str(first)
                else:
                    content = str(resp)
        except Exception:
            logger.exception("Parsing OpenAI response failed")
            content = str(resp)

        return jsonify({
            "reply": content,
            "sources": [m for _, m in top]
        })
    except Exception as e:
        logger.exception("Unhandled error in /chat")
        return jsonify({"error": str(e)}), 500


@app.route("/stream", methods=["POST"])
def stream():
    """
    Streaming chat using Server-Sent Events (SSE).
    Client should call with Accept: text/event-stream or normal POST.
    Input JSON: { "message": "...", "top_k": 5 }
    """
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "").strip()
        if not user_message:
            return jsonify({"error": "empty message"}), 400
        top_k = int(data.get("top_k", TOP_K))
        top = query_faiss(user_message, top_k)
        system_prompt = compose_system_prompt(top)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]

        def gen():
            try:
                logger.info("STREAM request: model=%s top_k=%d msg_len=%d faiss_hits=%d", CHAT_MODEL, top_k, len(user_message), len(top))
                try:
                    stream_iter = openai.ChatCompletion.create(model=CHAT_MODEL, messages=messages, temperature=0.2, stream=True)
                except Exception as e_old:
                    logger.warning("openai.ChatCompletion.create(stream=True) failed, trying alternate call: %s", e_old)
                    try:
                        if hasattr(openai, "chat") and hasattr(openai.chat, "completions"):
                            stream_iter = openai.chat.completions.create(model=CHAT_MODEL, messages=messages, stream=True)
                        else:
                            raise e_old
                    except Exception as e2:
                        logger.exception("OpenAI streaming create failed")
                        yield f"data: {json.dumps({'error': 'openai_stream_create_failed', 'detail': str(e2)})}\n\n"
                        return

                if stream_iter is None:
                    logger.error("openai.ChatCompletion.create returned None (stream_iter is None)")
                    yield f"data: {json.dumps({'error':'openai_stream_returned_none'})}\n\n"
                    return

                for chunk in stream_iter:
                    try:
                        if not chunk:
                            continue
                        # robust extraction for chunk
                        if isinstance(chunk, dict):
                            choices = chunk.get("choices")
                            if choices and isinstance(choices, list) and len(choices) > 0:
                                delta = choices[0].get("delta", {}) if isinstance(choices[0], dict) else {}
                                content = delta.get("content", "")
                                if content:
                                    yield f"data: {json.dumps({'delta': content})}\n\n"
                        else:
                            # try object-like access
                            choices = getattr(chunk, "choices", None)
                            if choices and len(choices) > 0:
                                delta = getattr(choices[0], "delta", None)
                                if delta:
                                    content = getattr(delta, "get", lambda k, d=None: "")("content", "")
                                    if content:
                                        yield f"data: {json.dumps({'delta': content})}\n\n"
                    except Exception:
                        # log and continue
                        logger.exception("error processing stream chunk")
                        continue
                # at end, send sources
                yield f"data: {json.dumps({'done': True, 'sources': [m for _, m in top]})}\n\n"
            except Exception as ex:
                logger.exception("stream generator error")
                yield f"data: {json.dumps({'error': str(ex)})}\n\n"
        return Response(stream_with_context(gen()), mimetype="text/event-stream")
    except Exception as e:
        logger.exception("Unhandled error in /stream")
        return jsonify({"error": str(e)}), 500


# ---------- Startup ----------
if __name__ == "__main__":
    # initial load
    load_knowledge(KNOWLEDGE_PATH)
    # try to load index; if not available, try to build lazily/background
    try:
        # If mapping exists load it
        if os.path.exists(FAISS_MAPPING_PATH):
            load_mapping(FAISS_MAPPING_PATH)
        if os.path.exists(FAISS_INDEX_PATH):
            try:
                FAISS_INDEX = faiss.read_index(FAISS_INDEX_PATH)
                logger.info("✅ FAISS index loaded at startup.")
            except Exception as e:
                logger.exception("⚠️ Failed loading FAISS index at startup:")
                # attempt to build
                built = build_faiss_index(force_rebuild=True)
                logger.info("Built index: %s", built)
        else:
            # build in background thread to reduce startup latency but safe check in query_faiss prevents errors
            t = threading.Thread(target=build_faiss_index, kwargs={"force_rebuild": False}, daemon=True)
            t.start()
    except Exception as ex:
        logger.exception("Startup index error:")

    port = int(os.environ.get("PORT", 10000))
    logger.info("Server starting on port %d ...", port)
    app.run(host="0.0.0.0", port=port)
