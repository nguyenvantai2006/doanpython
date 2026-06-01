import requests
import json
import ast
import re
from urllib.parse import urljoin, urlparse


# =========================
# SAFE URL BUILDER
# =========================
def build_url(api_url, endpoint):
    base = api_url.rstrip("/")

    parsed = urlparse(endpoint)
    path = parsed.path.lstrip("/")

    url = urljoin(base + "/", path)

    if parsed.query:
        url += "?" + parsed.query

    return url


# =========================
# EXPRESSION NORMALIZER (IMPORTANT FIX)
# handles: "a'*200"
# =========================
def normalize_expression(value):
    if not isinstance(value, str):
        return value

    match = re.fullmatch(r"a'\*\s*(\d+)\s*", value.replace(" ", ""))
    if match:
        return "a" * int(match.group(1))

    return value


# =========================
# SAFE REQUEST PARSER
# =========================
def parse_request_body(raw):

    if raw is None or raw == "":
        return {}

    if isinstance(raw, dict):
        return raw

    if isinstance(raw, str):
        raw = raw.strip()

        # try JSON first
        try:
            return json.loads(raw)
        except Exception:
            pass

        # fallback python literal (your CSV format)
        try:
            return ast.literal_eval(raw)
        except Exception:
            return {}

    return {}


# =========================
# SEND REQUEST (SAFE)
# =========================
def send_request(api_url, endpoint, method, body=None, headers=None, timeout=10):

    url = build_url(api_url, endpoint)
    method = str(method).upper().strip()

    headers = headers or {"Content-Type": "application/json"}

    try:
        if method == "GET":
            return requests.get(url, headers=headers, timeout=timeout)

        elif method == "POST":
            return requests.post(url, json=body or {}, headers=headers, timeout=timeout)

        elif method == "PUT":
            return requests.put(url, json=body or {}, headers=headers, timeout=timeout)

        elif method == "DELETE":
            return requests.delete(url, headers=headers, timeout=timeout)

        else:
            return requests.request(method, url, json=body or {}, headers=headers, timeout=timeout)

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"[HTTP ERROR] {method} {url} -> {e}")


# =========================
# BUILD PAYLOAD (ROBUST FOR 175 CASES)
# =========================
def build_payload(case):

    endpoint = case.get("Endpoint", "")
    raw = case.get("Request", "")

    data = parse_request_body(raw)

    if not isinstance(data, dict):
        return {}

    # normalize null values
    for k, v in list(data.items()):
        if v in ("null", "None"):
            data[k] = None

    # remove None values (prevents invalid FastAPI request)
    data = {k: v for k, v in data.items() if v is not None}

    # normalize text field
    if "text" in data:
        data["text"] = normalize_expression(data["text"])

    # =========================
    # BATCH HANDLING
    # =========================
    if "/batch" in endpoint:
        texts = data.get("texts", [])

        if texts is None:
            return {"texts": []}

        if isinstance(texts, str):
            texts = [texts]

        if isinstance(texts, list):
            texts = [t for t in texts if t is not None]

        return {"texts": texts}

    return data


# =========================
# ASSERT RESULT
# =========================
def assert_result(data, expected, endpoint=None):

    if not expected:
        return True

    if not isinstance(data, dict):
        return False

    # =========================
    # SYSTEM ENDPOINTS
    # =========================
    if endpoint in ("/", "/health"):
        actual = (
            data.get("status")
            or data.get("message")
            or data.get("result")
        )

        if not actual:
            return False

        # CLEAN EMOJI + SYMBOLS
        actual = str(actual)
        actual = actual.replace("✅", "").replace("❌", "").strip().lower()

        return actual == str(expected).strip().lower()

    # =========================
    # ML MODEL RESPONSE
    # =========================
    label = (
        data.get("label")
        or data.get("sentiment")
        or data.get("result")
        or data.get("prediction")
    )

    if not label:
        return False

    return str(label).strip().upper() == str(expected).strip().upper()

    # =========================
    # ML MODEL RESPONSE
    # =========================
    label = (
        data.get("label")
        or data.get("sentiment")
        or data.get("prediction")
        or data.get("result")
    )

    if not label:
        return False

    return str(label).strip().upper() == str(expected).strip().upper()


# =========================
# STATUS CHECK
# =========================
def assert_status(status_code, expected_status):

    if expected_status in ("", None):
        return True

    try:
        return int(status_code) == int(expected_status)
    except Exception:
        return False