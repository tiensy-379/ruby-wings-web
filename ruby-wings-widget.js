(function () {
  if (window.RW_CHAT_LOADED) return;
  window.RW_CHAT_LOADED = true;

  function init() {
    // CSS
    var style = document.createElement('style');
    style.innerHTML =
      '#rw-chat-btn{position:fixed;left:15px;bottom:15px;width:60px;height:60px;border-radius:50%;border:none;background:#00b368;color:#fff;font-size:24px;cursor:pointer;z-index:99999}' +
      '#rw-chat-box{position:fixed;left:15px;bottom:90px;width:360px;height:480px;background:#fff;border-radius:12px;box-shadow:0 10px 40px rgba(0,0,0,.2);display:none;flex-direction:column;z-index:100000}' +
      '#rw-chat-box.open{display:flex}' +
      '#rw-chat-header{padding:12px;background:#00b368;color:#fff;font-weight:bold}' +
      '#rw-chat-body{flex:1;padding:12px;overflow:auto;font-size:14px}' +
      '#rw-chat-input{display:flex;border-top:1px solid #eee}' +
      '#rw-chat-input input{flex:1;padding:10px;border:none;outline:none}' +
      '#rw-chat-input button{padding:10px 14px;border:none;background:#00b368;color:#fff;cursor:pointer}';
    document.head.appendChild(style);

    // Button
    var btn = document.createElement('button');
    btn.id = 'rw-chat-btn';
    btn.innerHTML = '🤖';

    // Box
    var box = document.createElement('div');
    box.id = 'rw-chat-box';
    box.innerHTML =
      '<div id="rw-chat-header">Ruby Wings AI</div>' +
      '<div id="rw-chat-body">Chào bạn! Tôi có thể giúp gì cho bạn?</div>' +
      '<div id="rw-chat-input">' +
      '<input type="text" placeholder="Nhập tin nhắn..." />' +
      '<button>Gửi</button>' +
      '</div>';

    document.body.appendChild(btn);
    document.body.appendChild(box);

    btn.onclick = function () {
      box.className = box.className.indexOf('open') === -1 ? 'open' : '';
    };
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
