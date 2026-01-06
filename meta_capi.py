import time
import requests
import os
import uuid
import hashlib

# =========================
# EXISTING FUNCTION (GIỮ NGUYÊN 100%)
# =========================
def send_meta_pageview(request):
    try:
        print("=== META CAPI HIT ===")

        pixel_id = os.getenv("META_PIXEL_ID")
        token = os.getenv("META_CAPI_TOKEN")
        test_event_code = os.getenv("META_TEST_EVENT_CODE")  # BẮT BUỘC khi test

        if not pixel_id or not token:
            print("META CAPI ERROR: Missing PIXEL_ID or TOKEN")
            return

        event_id = str(uuid.uuid4())

        payload = {
            "data": [
                {
                    "event_name": "PageView",
                    "event_time": int(time.time()),
                    "event_id": event_id,
                    "event_source_url": request.url,
                    "action_source": "website",
                    "user_data": {
                        "client_ip_address": request.remote_addr,
                        "client_user_agent": request.headers.get("User-Agent")
                    }
                }
            ]
        }

        if test_event_code:
            payload["test_event_code"] = test_event_code

        url = f"https://graph.facebook.com/v18.0/{pixel_id}/events?access_token={token}"
        response = requests.post(url, json=payload, timeout=5)

        print("META CAPI STATUS:", response.status_code)
        print("META CAPI RESPONSE TEXT:", response.text)
        print("META CAPI EVENT_ID:", event_id)

    except Exception as e:
        print("META CAPI EXCEPTION:", str(e))


# =========================
# NEW CODE (ADD-ONLY, SAFE)
# =========================

def _hash(value: str) -> str:
    if not value:
        return ""
    return hashlib.sha256(value.strip().lower().encode("utf-8")).hexdigest()


def send_meta_lead(
    request,
    event_name="Lead",
    event_id=None,
    phone=None,
    value=None,
    currency="VND",
    content_name=None
):
    """
    Server-side Meta CAPI Lead / Call
    - ADD-ONLY
    - Feature-flag controlled
    - Fail-safe (không ảnh hưởng hệ thống cũ)
    """

    # Feature flag: mặc định OFF
    if os.getenv("ENABLE_META_CAPI_LEAD", "false").lower() not in ("1", "true", "yes"):
        return

    try:
        pixel_id = os.getenv("META_PIXEL_ID")
        token = os.getenv("META_CAPI_TOKEN")
        test_event_code = os.getenv("META_TEST_EVENT_CODE")

        if not pixel_id or not token:
            return

        if not event_id:
            event_id = str(uuid.uuid4())

        user_data = {
            "client_ip_address": request.remote_addr,
            "client_user_agent": request.headers.get("User-Agent")
        }

        if phone:
            user_data["ph"] = _hash(phone)

        payload_event = {
            "event_name": event_name,
            "event_time": int(time.time()),
            "event_id": event_id,
            "event_source_url": request.url,
            "action_source": "website",
            "user_data": user_data
        }

        if value is not None:
            payload_event["custom_data"] = {
                "value": value,
                "currency": currency
            }
            if content_name:
                payload_event["custom_data"]["content_name"] = content_name

        payload = {"data": [payload_event]}

        if test_event_code:
            payload["test_event_code"] = test_event_code

        url = f"https://graph.facebook.com/v18.0/{pixel_id}/events?access_token={token}"
        requests.post(url, json=payload, timeout=5)

    except Exception:
        # Fail-safe tuyệt đối: nuốt lỗi
        return
def send_meta_call_button(
    request,
    event_name="CallButtonClick",
    event_id=None,
    phone=None,
    call_type="phone",
    page_url=None,
    value=150000,
    currency="VND",
    # === THÊM THAM SỐ MỚI ===
    fbp=None,    # Facebook Browser ID
    fbc=None,    # Facebook Click ID
    pixel_id=None  # Pixel ID từ client
):
    """
    Server-side Meta CAPI cho nút gọi điện
    - Feature-flag controlled
    - Fail-safe (không ảnh hưởng hệ thống cũ)
    """
    
    # Feature flag: mặc định OFF
    if os.getenv("ENABLE_META_CAPI_CALL", "false").lower() not in ("1", "true", "yes"):
        return
    
    try:
        # Ưu tiên dùng pixel_id từ client, fallback đến biến môi trường
        target_pixel_id = pixel_id or os.getenv("META_PIXEL_ID")
        token = os.getenv("META_CAPI_TOKEN")
        test_event_code = os.getenv("META_TEST_EVENT_CODE")
        
        if not target_pixel_id or not token:
            return
        
        # Sử dụng event_id từ client nếu có, không thì tạo mới
        if not event_id:
            event_id = str(uuid.uuid4())
        
        user_data = {
            "client_ip_address": request.remote_addr,
            "client_user_agent": request.headers.get("User-Agent")
        }
        
        # THÊM fbp/fbc nếu có từ client
        if fbp:
            user_data["fbp"] = fbp
        if fbc:
            user_data["fbc"] = fbc
        
        # Hash số điện thoại nếu có
        if phone:
            user_data["ph"] = _hash(phone)
        
        payload_event = {
            "event_name": event_name,
            "event_time": int(time.time()),
            "event_id": event_id,
            "event_source_url": page_url or request.url,
            "action_source": "website",
            "user_data": user_data,
            "custom_data": {
                "value": value,
                "currency": currency,
                "call_type": call_type,
                "button_location": "fixed_bottom_left"
            }
        }
        
        payload = {"data": [payload_event]}
        
        if test_event_code:
            payload["test_event_code"] = test_event_code
        
        url = f"https://graph.facebook.com/v18.0/{target_pixel_id}/events?access_token={token}"
        response = requests.post(url, json=payload, timeout=3)
        
        # Log cho debugging (tùy chọn)
        print(f"META CAPI CALL BUTTON SENT - Event: {event_name}, Pixel: {target_pixel_id}")
        
    except Exception as e:
        # Fail-safe tuyệt đối
        print(f"META CAPI CALL BUTTON ERROR: {e}")
        return