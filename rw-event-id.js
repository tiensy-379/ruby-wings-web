(function () {
  if (window.RW_EVENT_ID) return;

  try {
    window.RW_EVENT_ID = crypto.randomUUID();
  } catch (e) {
    window.RW_EVENT_ID =
      'rw-' + Date.now() + '-' + Math.random().toString(36).slice(2);
  }

  // alias để tương thích ngược (nếu code cũ còn dùng)
  window.__RW_EVENT_ID__ = window.RW_EVENT_ID;
})();
