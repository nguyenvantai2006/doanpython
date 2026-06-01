import os
from playwright.sync_api import Page, expect

UI_URL = os.getenv("UI_URL", "http://localhost:8501")


def test_vietnamese_text(page: Page):
    # ── Open UI ─────────────────────────────────────────────
    page.goto(UI_URL, wait_until="domcontentloaded")

    # Streamlit takes time to load
    page.wait_for_timeout(5000)

    # ── Fill input box ─────────────────────────────────────
    page.get_by_placeholder(
        "e.g., chất lượng tuyệt vời giao hàng rất nhanh"
    ).fill("chất lượng tuyệt vời giao hàng rất nhanh")

    # ── Click analyze button ───────────────────────────────
    page.get_by_role("button", name="🔍 Phân tích Cảm xúc").click()

    # ── Wait for result box ────────────────────────────────
    page.wait_for_timeout(3000)

    # ── Assertions (based on your UI HTML output) ──────────
        # ✅ wait for ANY result box (any sentiment class)
    result = page.locator("div.sentiment-box")
    expect(result).to_be_visible(timeout=15000)

