import requests
url = 'http://127.0.0.1:8000/predict'
payload = {"text": "sản phẩm tuyệt vời giao hàng nhanh", "model": "bert"}
try:
    r = requests.post(url, json=payload, timeout=10)
    print('status', r.status_code)
    try:
        print(r.json())
    except Exception as e:
        print('raw:', r.text)
except Exception as e:
    print('request failed', e)
