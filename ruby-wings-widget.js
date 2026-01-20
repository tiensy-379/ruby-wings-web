(function () {
  if (window.RW_WIDGET_LOADED) return;
  window.RW_WIDGET_LOADED = true;

  var BACKEND_URL = 'https://ruby-wings-chatbot.onrender.com';

  function init() {
    /* ========== CSS ========== */
    var style = document.createElement('style');
    style.innerHTML =
      '#rw-chat-btn{position:fixed;left:15px;bottom:15px;width:60px;height:60px;border-radius:50%;background:#00b368;color:#fff;font-size:24px;border:none;cursor:pointer;z-index:99999}' +
      '#rw-chat-box{position:fixed;left:15px;bottom:90px;width:360px;height:480px;background:#fff;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,.2);display:none;flex-direction:column;z-index:100000}' +
      '#rw-chat-box.open{display:flex}' +
      '#rw-chat-header{background:#00b368;color:#fff;padding:12px;font-weight:bold}' +
      '#rw-chat-body{flex:1;padding:12px;overflow:auto;font-size:14px;background:#f8fafc}' +
      '#rw-chat-input{display:flex;border-top:1px solid #eee}' +
      '#rw-chat-input input{flex:1;padding:10px;border:none;outline:none}' +
      '#rw-chat-input button{padding:10px 14px;border:none;background:#00b368;color:#fff;cursor:pointer}';
    document.head.appendChild(style);

    /* ========== HTML ========== */
    var btn = document.createElement('button');
    btn.id = 'rw-chat-btn';
    btn.innerHTML = '🤖';

    var box = document.createElement('div');
    box.id = 'rw-chat-box';
    box.innerHTML =
      '<div id="rw-chat-header">Trợ lý Ruby Wings AI</div>' +
      '<div id="rw-chat-body">Chào bạn! Tôi có thể hỗ trợ gì cho bạn?</div>' +
      '<div id="rw-chat-input">' +
      '<input type="text" placeholder="Nhập tin nhắn..." />' +
      '<button>Gửi</button>' +
      '</div>';

    document.body.appendChild(btn);
    document.body.appendChild(box);

    /* ========== TOGGLE ========== */
    btn.onclick = function () {
      box.className = box.className.indexOf('open') === -1 ? 'open' : '';
    };

    /* ========== CHAT LOGIC ========== */
    var input = box.querySelector('input');
    var sendBtn = box.querySelector('button');
    var body = box.querySelector('#rw-chat-body');

    function addMsg(who, text) {
      body.innerHTML += '<div><b>' + who + ':</b> ' + text + '</div>';
      body.scrollTop = body.scrollHeight;
    }

    function send() {
      var text = input.value.replace(/^\s+|\s+$/g, '');
      if (!text) return;

      addMsg('Bạn', text);
      input.value = '';

      var xhr = new XMLHttpRequest();
      xhr.open('POST', BACKEND_URL + '/chat', true);
      xhr.setRequestHeader('Content-Type', 'application/json');
      xhr.onreadystatechange = function () {
        if (xhr.readyState === 4) {
          try {
            var res = JSON.parse(xhr.responseText || '{}');
            addMsg('AI', res.reply || '...');
          } catch (e) {}
        }
      };
      xhr.send(JSON.stringify({
        message: text,
        page_url: window.location.href
      }));
    }

    sendBtn.onclick = send;
    input.onkeypress = function (e) {
      if (e.key === 'Enter') send();
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
