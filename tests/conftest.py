import pytest
import os
from collections import defaultdict, Counter

# =========================
# CONFIG
# =========================
BASE_URL = os.getenv("BASE_URL", "http://localhost:8000")
CI_MODE = os.getenv("CI_MODE", "1") == "1"

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORT_DIR = os.path.join(BASE, "reports")

# =========================
# GLOBAL STATS
# =========================
TEST_STATS = {
    "category_type": defaultdict(lambda: {"PASS": 0, "FAIL": 0}),
    "clusters": defaultdict(lambda: {"PASS": 0, "FAIL": 0}),
    "endpoints": defaultdict(lambda: {"PASS": 0, "FAIL": 0}),
}

# =========================
# FIXTURES
# =========================
@pytest.fixture(scope="session")
def api_url():
    return BASE_URL


@pytest.fixture(scope="session")
def headers():
    return {"Content-Type": "application/json"}


@pytest.fixture(scope="session")
def ci_mode():
    return CI_MODE


@pytest.fixture(scope="session")
def report_dir():
    os.makedirs(REPORT_DIR, exist_ok=True)
    return REPORT_DIR


# =========================
# HTML HEADER
# =========================
def pytest_html_results_table_header(cells):
    # NEVER modify existing structure
    cells.extend([
        "Cluster",
        "Category",
        "Type",
        "Method",
        "Endpoint",
        "Latency",
        "Expected",
        "Actual",
        "Status",
    ])

# =========================
# HTML ROW
# =========================
def pytest_html_results_table_row(report, cells):

    status = getattr(report, "test_status", "UNKNOWN")

    color = (
        "color:green;font-weight:bold;" if status == "PASS"
        else "color:red;font-weight:bold;"
    )

    latency = float(getattr(report, "latency", 0) or 0)

    cluster = str(getattr(report, "cluster", "UNKNOWN"))
    category = str(getattr(report, "category", "N/A"))
    ctype = str(getattr(report, "type", "N/A"))
    method = str(getattr(report, "method", "N/A"))
    endpoint = str(getattr(report, "endpoint", ""))

    expected = str(getattr(report, "expected_result", ""))
    actual = str(getattr(report, "actual_result", ""))

    def safe(v):
        return v if v not in [None, ""] else "-"

    # IMPORTANT: DO NOT reformat base structure
    extra_columns = [
        f"[{cluster}]",
        f"{category} / {ctype}",
        ctype,
        method,
        f"{method} {endpoint}".strip(),
        f"{latency:.2f} ms",
        safe(expected),
        safe(actual),
        f'<span style="{color}">{status}</span>',
    ]

    cells.extend(extra_columns)


# =========================
# TOP FAILING ENDPOINTS
# =========================
def top_failing_endpoints(n=10):
    counter = Counter()
    for ep, v in TEST_STATS["endpoints"].items():
        counter[ep] = v["FAIL"]
    return counter.most_common(n)


# =========================
# HOOK: ATTACH METADATA + AUTO CLUSTERING
# =========================
@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):

    outcome = yield
    rep = outcome.get_result()

    if rep.when != "call":
        return

    case = {}
    if hasattr(item, "callspec") and item.callspec:
        case = item.callspec.params.get("case", {}) or {}

    category = case.get("Category", "N/A")
    ctype = case.get("Type", "N/A")
    endpoint = case.get("Endpoint", "")

    category_upper = str(category).upper()

    # =========================
    # CLUSTERS
    # =========================
    if category_upper == "SYSTEM":
        cluster = "API_HEALTH"

    elif category_upper == "TFIDF":
        cluster = "MODEL_CLASSICAL"

    elif category_upper == "BERT":
        cluster = "MODEL_DEEP"

    elif category_upper in ["EDGE", "BOUNDARY", "BATCH_EDGE"]:
        cluster = "INPUT_VALIDATION"

    elif category_upper == "SECURITY":
        cluster = "SECURITY_TESTS"

    elif category_upper == "ROUTING":
        cluster = "ROUTING_TESTS"

    else:
        cluster = "SENTIMENT_BEHAVIOR"

    # =========================
    # ATTACH META
    # =========================
    rep.cluster = cluster
    rep.category = category
    rep.type = ctype
    rep.method = case.get("Method", "N/A")
    rep.endpoint = endpoint
    rep.expected_result = case.get("Expected_Result", "")
    rep.actual_result = getattr(item, "actual_result", "")
    rep.latency = getattr(item, "latency", 0)

    rep.test_status = "PASS" if rep.passed else "FAIL"

    # =========================
    # UPDATE STATS
    # =========================
    TEST_STATS["category_type"][(category, ctype)][rep.test_status] += 1
    TEST_STATS["clusters"][cluster][rep.test_status] += 1
    TEST_STATS["endpoints"][endpoint][rep.test_status] += 1


# =========================
# SUMMARY DASHBOARD
# =========================
def pytest_html_results_summary(prefix, summary, postfix):

    total = sum(v["PASS"] + v["FAIL"] for v in TEST_STATS["category_type"].values())
    total_pass = sum(v["PASS"] for v in TEST_STATS["category_type"].values())
    total_fail = sum(v["FAIL"] for v in TEST_STATS["category_type"].values())

    pass_rate = (total_pass / total * 100) if total else 0

    prefix.extend([
        "<h2>📊 Sentiment AI Test Dashboard</h2>",
        f"<p><b>Total:</b> {total} | "
        f"<b>PASS:</b> {total_pass} | "
        f"<b>FAIL:</b> {total_fail} | "
        f"<b>Pass Rate:</b> {pass_rate:.2f}%</p>",
    ])

    # CLUSTERS
    prefix.append("<h3>🧠 Test Clusters</h3><ul>")
    for cluster, v in TEST_STATS["clusters"].items():
        prefix.append(f"<li><b>{cluster}</b>: PASS={v['PASS']} | FAIL={v['FAIL']}</li>")
    prefix.append("</ul>")

    # TOP FAILURES
    prefix.append("<h3>🚨 Top Failing Endpoints</h3><ul>")
    for ep, fail_count in top_failing_endpoints():
        prefix.append(f"<li>{ep or '-'}: {fail_count} FAIL</li>")
    prefix.append("</ul>")