(function () {
  // RW_EVENT_ID: CHỈ DÙNG CHO PAGEVIEW (Pixel + CAPI)
  // TUYỆT ĐỐI KHÔNG dùng cho conversion

  if (window.RW_EVENT_ID) return;

  var eid = null;

  try {
    if (window.crypto && typeof crypto.randomUUID === 'function') {
      eid = crypto.randomUUID();
    }
  } catch (e) {}

  if (!eid) {
    eid =
      'rw-pv-' +
      Date.now() +
      '-' +
      Math.random().toString(36).slice(2);
  }

  window.RW_EVENT_ID = eid;

  // Alias tương thích ngược (code cũ)
  window.__RW_EVENT_ID__ = eid;
})();
