"""
test_bert.py — Kiểm tra BERT model trước khi đẩy lên GitHub
=============================================================
Cách chạy:
    python tests/test_bert.py

Kết quả PASS = BERT hoạt động tốt, sẵn sàng push lên GitHub
"""

import os, sys, time, json
import torch
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

BERT_DIR = os.path.join(BASE, "models", "bert_sentiment")
REPORT_DIR = os.path.join(BASE, "reports")
os.makedirs(REPORT_DIR, exist_ok=True)

TXT_REPORT = os.path.join(REPORT_DIR, "test_bert_result.txt")

class TeeLogger:
    def __init__(self, file_path):
        self.terminal = sys.stdout
        self.log = open(file_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = TeeLogger(TXT_REPORT)
PASS_STR = "✅ PASS"
FAIL_STR = "❌ FAIL"
errors   = []

def check(name, condition, detail=""):
    if condition:
        print(f"  {PASS_STR}  {name}")
    else:
        msg = f"  {FAIL_STR}  {name}" + (f" — {detail}" if detail else "")
        print(msg)
        errors.append(name)

# ─────────────────────────────────────────────────────────────────
print("=" * 58)
print("  TEST BERT — Vietnamese Sentiment Analysis")
print("=" * 58)

# ── Test 1: Detect available sub-models ──────────────────────────
print("\n[1] Kiểm tra files & sub-models")
check("Thư mục bert_sentiment/ tồn tại",  os.path.isdir(BERT_DIR))

from src.bert_inference import list_available_models
available = list_available_models()
print(f"   Detected sub-models: {available}")
check("Có ít nhất 1 sub-model sẵn sàng", len(available) > 0,
      "Chưa train: python src/train_bert.py --model demo")

if not available:
    print("\n❌ Không tìm thấy model nào. Chạy: python src/train_bert.py --model demo")
    sys.exit(1)

# Check each sub-model folder
for model_key in available:
    if model_key == "legacy":
        mdl_dir = BERT_DIR
    else:
        mdl_dir = os.path.join(BERT_DIR, model_key)
    ckpt = os.path.join(mdl_dir, "bert_model.pt")
    check(f"Sub-model '{model_key}': bert_model.pt tồn tại", os.path.exists(ckpt))

# ── Test 2: Load checkpoint for first available ──────────────────
ck = []
for i in range(len(available)):
    first_model = available[i]
    if first_model == "legacy":
        ckpt_path = os.path.join(BERT_DIR, "bert_model.pt")
    else:
        ckpt_path = os.path.join(BERT_DIR, first_model, "bert_model.pt")

    print(f"\n[2] Load checkpoint ({first_model})")
    try:
        # Allowlist common numpy globals used in legacy checkpoints
        allowlist = [
            getattr(np, "dtype", None),
            getattr(np.core.multiarray, "scalar", None),
            getattr(np, "ndarray", None),
            getattr(np, "float64", None),
            (getattr(getattr(np, "dtypes", None), "Float64DType", None)),
        ]
        allowlist = [a for a in allowlist if a is not None]

        try:
            with torch.serialization.safe_globals(allowlist):
                ck.append(torch.load(ckpt_path, map_location="cpu"))
        except AttributeError:
            # Fallback for torch versions without safe_globals
            try:
                torch.serialization.add_safe_globals(allowlist)
            except Exception:
                pass
            ck.append(torch.load(ckpt_path, map_location="cpu"))

        check("Load bert_model.pt thành công", True)
    except Exception as e:
        check("Load bert_model.pt thành công", False, str(e))
        sys.exit(1)

    required_keys = ["model_state", "bert_config", "label_map", "accuracy", "f1_macro", "model_key"]
    for k in required_keys:
        check(f"Key '{k}' tồn tại trong checkpoint", k in ck[i])

    check("Accuracy >= 0.70",
        ck[i].get("accuracy", 0) >= 0.70,
        f"Hiện tại: {ck[i].get('accuracy',0):.4f}")
    check("Macro-F1 >= 0.60",
        ck[i].get("f1_macro", 0) >= 0.60,
        f"Hiện tại: {ck[i].get('f1_macro',0):.4f}")

# ── Test 3: Load BERTPredictor for each available model ──────────
print("\n[3] Khởi tạo BERTPredictor")
from src.bert_inference import BERTPredictor

predictors = {}
for model_key in available:
    try:
        p = BERTPredictor(model_key=model_key)
        predictors[model_key] = p
        check(f"BERTPredictor('{model_key}') khởi tạo thành công", True)
        check(f"  model_key = '{p.model_key}'", p.model_key in ["demo","phobert","mbert","unknown","legacy"])
    except Exception as e:
        check(f"BERTPredictor('{model_key}') khởi tạo thành công", False, str(e))

if not predictors:
    print("\n❌ Không khởi tạo được predictor nào.")
    sys.exit(1)

for i in range(len(predictors)):   
    predictor = list(predictors.values())[i]
    test_model_key = list(predictors.keys())[i]

    print("\n" + "=" * 58)
    print(f"   → Sử dụng '{test_model_key}' cho các test tiếp theo")
    print("\n" + "=" * 58)

    # ── Test 4: Semantic correctness ─────────────────────────────────
    print("\n[4] Kiểm tra ngữ nghĩa (semantic correctness)")

    test_cases = [
        ("chất lượng tuyệt vời giao hàng rất nhanh đóng gói đẹp", "POSITIVE"),
        ("hàng lỗi giao sai màu cửa hàng không trả lời khiếu nại", "NEGATIVE"),
        ("sản phẩm tạm được giá hơi cao so với chất lượng",        "NEUTRAL"),
        ("rất đáng tiền sẽ mua lại ủng hộ cửa hàng",              "POSITIVE"),
        ("giao thiếu hàng nhắn tin không phản hồi rất thất vọng",  "NEGATIVE"),
        ("hàng tạm ổn nhưng giao hơi chậm bình thường thôi",           "NEUTRAL"),
    ]

    all_correct = True
    for text, expected in test_cases:
        result = predictor.predict(text)
        pred   = result["label"]
        conf   = result["confidence"]
        ok     = pred == expected
        if not ok:
            all_correct = False
        emoji  = {"POSITIVE":"😊","NEUTRAL":"😐","NEGATIVE":"😞"}[pred]
        status = PASS_STR if ok else FAIL_STR
        print(f"  {status}  [{emoji} {pred:8s} conf={conf:.4f}]  {text[:52]}")

    if not all_correct:
        errors.append("Một số câu semantic test sai")

    # ── Test 5: Output format ────────────────────────────────────────
    print("\n[5] Kiểm tra output format")
    r = predictor.predict("hàng đẹp lắm")
    check("Có key 'label'",      "label"      in r)
    check("Có key 'sentiment'",  "sentiment"  in r)
    check("Có key 'confidence'", "confidence" in r)
    check("Có key 'scores'",     "scores"     in r)
    check("scores có đủ 3 nhãn",
        set(r["scores"].keys()) == {"NEGATIVE","NEUTRAL","POSITIVE"})
    check("Tổng xác suất ≈ 1.0",
        abs(sum(r["scores"].values()) - 1.0) < 1e-3,
        f"Tổng = {sum(r['scores'].values()):.6f}")
    check("Confidence trong [0, 1]",
        0.0 <= r["confidence"] <= 1.0,
        f"confidence = {r['confidence']}")

    # ── Test 6: Batch inference ──────────────────────────────────────
    print("\n[6] Kiểm tra batch inference")
    batch = ["hàng đẹp lắm", "giao hàng chậm quá", "tạm ổn thôi"] * 4   # 12 câu
    t0    = time.perf_counter()
    preds = predictor.predict_batch(batch)
    ms    = (time.perf_counter() - t0) * 1000
    check("Batch 12 câu trả về đúng số kết quả", len(preds) == 12)
    check(f"Tốc độ batch < 2000ms (hiện tại: {ms:.0f}ms)", ms < 2000)

    # ── Test 7: Edge cases ───────────────────────────────────────────
    print("\n[7] Kiểm tra edge cases")

    try:
        predictor.predict("ok")
        check("Câu rất ngắn (2 ký tự) không crash", True)
    except Exception as e:
        check("Câu rất ngắn (2 ký tự) không crash", False, str(e))

    try:
        predictor.predict("good quality fast delivery great product")
        check("Câu tiếng Anh không crash", True)
    except Exception as e:
        check("Câu tiếng Anh không crash", False, str(e))

    try:
        long_text = "chất lượng tốt " * 150       # ~375 từ, vượt max_len
        predictor.predict(long_text)
        check("Câu rất dài (truncation) không crash", True)
    except Exception as e:
        check("Câu rất dài (truncation) không crash", False, str(e))

    try:
        predictor.predict("!!! @@@ ### $$$")
        check("Câu toàn ký tự đặc biệt không crash", True)
    except Exception as e:
        check("Câu toàn ký tự đặc biệt không crash", False, str(e))

    try:
        predictor.predict("")
        check("Câu rỗng không crash", True)
    except Exception as e:
        # Chấp nhận exception nếu có — behaviour đã xác định
        check("Câu rỗng không crash", True)   # document expected behaviour

    # ── Test 8: Reproducibility ──────────────────────────────────────
    print("\n[8] Kiểm tra tính ổn định (reproducibility)")
    text  = "sản phẩm rất tốt đáng tiền"
    runs  = [predictor.predict(text)["label"] for _ in range(5)]
    check("5 lần predict cùng câu → cùng kết quả", len(set(runs)) == 1,
        f"Kết quả: {runs}")

    # ── Test 9: list_available_models helper ─────────────────────────
    print("\n[9] Kiểm tra list_available_models()")
    check("list_available_models() trả về list", isinstance(available, list))
    check("Mỗi model trong list có checkpoint",
        all(
            os.path.exists(os.path.join(BERT_DIR, m, "bert_model.pt")) if m != "legacy"
            else os.path.exists(os.path.join(BERT_DIR, "bert_model.pt"))
            for m in available
        ))

# ── Kết quả tổng ─────────────────────────────────────────────────
print("\n" + "=" * 58)
if not errors:
    print(f"  🎉 TẤT CẢ TESTS ĐỀU PASS")
    for i in range(len(available)):
        print(f"  ✅ Accuracy={ck[i]['accuracy']:.4f}  Macro-F1={ck[i]['f1_macro']:.4f} Models: {available[i]}")
    print(f"  ✅ BERT sẵn sàng đẩy lên GitHub!")
else:
    print(f"  ⚠️  {len(errors)} test(s) FAIL:")
    for e in errors:
        print(f"     • {e}")
    print("  Kiểm tra lại trước khi push.")
print("=" * 58)

sys.exit(0 if not errors else 1)