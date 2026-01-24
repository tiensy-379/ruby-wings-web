(function () {
  if (window.__RW_EVENT_ID__) return;
  try {
    window.__RW_EVENT_ID__ = crypto.randomUUID();
  } catch (e) {
    window.__RW_EVENT_ID__ = 'rw-' + Date.now() + '-' + Math.random().toString(36).slice(2);
  }
})();
