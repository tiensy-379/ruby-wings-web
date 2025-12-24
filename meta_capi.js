import fetch from "node-fetch";

export async function sendMetaPageView(req) {
  const pixelId = process.env.META_PIXEL_ID;
  const token = process.env.META_CAPI_TOKEN;

  if (!pixelId || !token) return;

  const payload = {
    data: [
      {
        event_name: "PageView",
        event_time: Math.floor(Date.now() / 1000),
        event_source_url: req.headers.referer || "",
        action_source: "website",
        user_data: {
          client_ip_address: req.ip,
          client_user_agent: req.headers["user-agent"]
        }
      }
    ]
  };

  fetch(
    `https://graph.facebook.com/v18.0/${pixelId}/events?access_token=${token}`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    }
  ).catch(() => {});
}
