// Ruby Wings Chat Widget (GTM version)
(function () {
  if (window.RW_CHAT_LOADED) return;
  window.RW_CHAT_LOADED = true;

  const BACKEND_URL = "https://ruby-wings-chatbot.onrender.com";

  function init() {
    // inject CSS
    const style = document.createElement("style");
    style.textContent = `
      #rw-chat-btn {
        position: fixed;
        left: 15px;
        bottom: 15px;
        width: 60px;
        height: 60px;
        border-radius: 50%;
        background: linear-gradient(135deg,#00d97e,#00b368);
        color:#fff;
        font-size:24px;
        border:none;
        cursor:pointer;
        z-index:10001;
      }
      #rw-chat-box {
        position: fixed;
        left: 15px;
        bottom: 90px;
        width: 360px;
        height: 480px;
        background:#fff;
        border-radius:12px;
        box-shadow:0 10px 40px rgba(0,0,0,.2);
        display:none;
        flex-direction:column;
        z-index:10002;
      }
      #rw-chat-box.open { display:flex; }
      #rw-chat-header {
        padding:12px;
        background:#00b368;
        color:#fff;
        font-weight:bold;
      }
      #rw-chat-body {
        flex:1;
        padding:12px;
        overflow:auto;
        font-size:14px;
      }
      #rw-chat-input {
        display:flex;
        border-top:1px solid #eee;
      }
      #rw-chat-input input {
        flex:1;
        padding:10px;
        border:none;
        outline:none;
      }
      #rw-chat-input button {
        padding:10px 14px;
        border:none;
        background:#00b368;
        color:#fff;
        cursor:pointer;
      }
    `;
    document.head.appendChild(style);

    // HTML
    const btn = document.createElement("button");
    btn.id = "rw-chat-btn";
    btn.textContent = "🤖";

    const box = document.createElement("div");
    box.id = "rw-chat-box";
    box.innerHTML = `
      <div id="rw-chat-header">Ruby Wings AI</div>
      <div id="rw-chat-body">Xin chào! Tôi có thể giúp gì cho bạn?</div>
      <div id="rw-chat-input">
        <input placeholder="Nhập tin nhắn..." />
        <button>Gửi</button>
      </div>
    `;

    document.body.appendChild(btn);
    document.body.appendChild(box);

    btn.onclick = () => box.classList.toggle("open");

    const input = box.querySelector("input");
    const sendBtn = box.querySelector("button");
    const body = box.querySelector("#rw-chat-body");

    sendBtn.onclick = send;
    input.addEventListener("keypress", e => e.key === "Enter" && send());

    function send() {
      const text = input.value.trim();
      if (!text) return;
      body.innerHTML += `<div><b>Bạn:</b> ${text}</div>`;
      input.value = "";

      fetch(BACKEND_URL + "/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text })
      })
        .then(r => r.json())
        .then(d => {
          body.innerHTML += `<div><b>AI:</b> ${d.reply || "..."}</div>`;
          body.scrollTop = body.scrollHeight;
        });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
