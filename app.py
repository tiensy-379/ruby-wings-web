# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
import traceback

app = Flask(__name__)
CORS(app)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip()
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

if not OPENROUTER_API_KEY:
    app.logger.warning("OPENROUTER_API_KEY chưa được thiết lập. Backend sẽ không gọi OpenRouter được.")

@app.route('/')
def home():
    if OPENROUTER_API_KEY:
        return "✅ Backend đang chạy - OpenRouter key được thiết lập"
    else:
        return "⚠️ Backend chạy nhưng OPENROUTER_API_KEY chưa được thiết lập"

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json() or {}
        user_message = data.get("message", "")
        if not user_message:
            return jsonify({"reply": "Vui lòng gửi message trong body JSON, ví dụ: {\"message\":\"Xin chào\"}"}), 400

        if not OPENROUTER_API_KEY:
            return jsonify({"reply": "Server chưa cấu hình OPENROUTER_API_KEY."}), 500

        payload = {
            "model": "gpt-3.5-turbo",
            "messages": [
                {"role": "user", "content": user_message}
            ],
            "max_tokens": 512,
            "temperature": 0.6
        }

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
            "Content-Type": "application/json"
        }

        resp = requests.post(f"{OPENROUTER_BASE}/chat/completions", json=payload, headers=headers, timeout=60)

        if resp.status_code not in (200, 201):
            app.logger.error("OpenRouter trả lỗi %s: %s", resp.status_code, resp.text)
            try:
                body = resp.json()
                msg = body.get("error") or body.get("message") or resp.text
            except Exception:
                msg = resp.text
            return jsonify({"reply": f"Lỗi từ OpenRouter ({resp.status_code}): {msg}"}), 502

        result = resp.json()
        reply = None
        try:
            reply = result.get("choices", [])[0].get("message", {}).get("content")
        except Exception:
            reply = None

        if not reply:
            reply = result.get("reply") or result.get("message") or str(result)

        return jsonify({"reply": reply})

    except requests.exceptions.Timeout:
        app.logger.exception("Request tới OpenRouter timeout")
        return jsonify({"reply": "Hết thời gian chờ khi gọi mô hình. Vui lòng thử lại."}), 504
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error("Lỗi khi xử lý /chat: %s\n%s", str(e), tb)
        return jsonify({"reply": "Lỗi nội bộ server. Kiểm tra logs."}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
