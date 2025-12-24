import time
import requests
import os

def send_meta_pageview(request):
    try:
        print("META CAPI HIT")  # log kiểm tra hàm có chạy hay không

        pixel_id = os.getenv("META_PIXEL_ID")
        token = os.getenv("META_CAPI_TOKEN")

        if not pixel_id or not token:
            print("META CAPI MISSING ENV")
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
        response = requests.post(url, json=payload, timeout=2)

        print("META CAPI RESPONSE:", response.status_code)

    except Exception as e:
        print("META CAPI ERROR:", str(e))
