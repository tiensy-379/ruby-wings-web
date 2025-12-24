import time
import requests
import os

def send_meta_pageview(request):
    pixel_id = os.getenv("META_PIXEL_ID")
    token = os.getenv("META_CAPI_TOKEN")

    if not pixel_id or not token:
        return

    payload = {
        "data": [
            {
                "event_name": "PageView",
                "event_time": int(time.time()),
                "event_source_url": request.url,
                "action_source": "website",
                "user_data": {
                    "client_ip_address": request.remote_addr,
                    "client_user_agent": request.headers.get("User-Agent")
                }
            }
        ]
    }

    url = f"https://graph.facebook.com/v18.0/{pixel_id}/events?access_token={token}"
    try:
        requests.post(url, json=payload, timeout=2)
    except:
        pass
