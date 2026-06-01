import csv
import pytest
import json
from io import StringIO

from utils import (
    send_request,
    assert_result,
    build_payload
)


# =========================
# LOAD TEST CASES (ROBUST)
# =========================
def load_cases():
    cases = []

    with open("tests/api_testcase.csv", "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            # skip comments / empty
            if not line or line.startswith("#"):
                continue

            cases.append(line)

    return list(csv.DictReader(StringIO("\n".join(cases))))


# =========================
# LABEL NORMALIZER
# =========================
def normalize_label(label):
    if not isinstance(label, str):
        return ""

    label = label.upper().strip()

    for ch in ["😊", "😞", "😐", "🙂", "🔥", "💥"]:
        label = label.replace(ch, "")

    label = label.replace("LABEL:", "").replace("RESULT:", "").strip()

    return label


# =========================
# SAFE JSON PARSER
# =========================
def safe_json(response):
    try:
        return response.json()
    except Exception:
        return {}


# =========================
# TEST
# =========================
@pytest.mark.parametrize("case", load_cases())
def test_api(case, api_url, request):

    # =========================
    # BUILD PAYLOAD
    # =========================
    body = build_payload(case)
    request.node.request_body = body

    # =========================
    # FIX: USE METHOD COLUMN (IMPORTANT)
    # =========================
    method = case.get("Method", "POST")

    # =========================
    # SEND REQUEST
    # =========================
    res = send_request(
        api_url=api_url,
        endpoint=case.get("Endpoint", "/"),
        method=method,
        body=body
    )

    # =========================
    # PARSE RESPONSE SAFELY
    # =========================
    data = safe_json(res)

    # =========================
    # METADATA FOR REPORT
    # =========================
    request.node.expected_status = case.get("Expected_Status", "")
    request.node.actual_status = str(res.status_code)
    request.node.expected_result = case.get("Expected_Result", "")

    # =========================
    # NORMALIZE LABEL
    # =========================
    actual_label = (
        data.get("label")
        or data.get("sentiment")
        or data.get("result")
        or ""
    )

    request.node.actual_result = normalize_label(actual_label)

    # =========================
    # ASSERT STATUS
    # =========================
    expected_status = case.get("Expected_Status")

    assert str(res.status_code) == str(expected_status), (
        f"TC {case.get('TC_ID')} failed: "
        f"expected {expected_status} got {res.status_code}"
    )

    # =========================
    # ASSERT RESULT
    # =========================
    expected = case.get("Expected_Result")

    if expected:
        assert assert_result(data, expected, case.get("Endpoint")), (
    f"TC {case.get('TC_ID')} failed: expected {expected} got {data}"
)