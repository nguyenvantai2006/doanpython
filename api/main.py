"""
main.py — Vietnamese Sentiment FastAPI Server
=============================================
Cách chạy:
    uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
"""

import os, time, sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from scipy.sparse import hstack
import joblib

# ── Load model ────────────────────────────────────────────────────
BASE       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

MODEL_PATH = os.getenv("MODEL_PATH",
             os.path.join(BASE, "models", "model_sentiment", "sentiment_model.joblib"))



MODEL_NAME = "tfidf"
ACCURACY = 0.0
F1_MACRO = 0.0
MODEL = None
TFIDF_WORD = None
TFIDF_CHAR = None
USE_CHAR = True
MODEL_AVAILABLE = False

if not os.path.exists(MODEL_PATH):
    MODEL_AVAILABLE = False
    print(f"⚠️  TF-IDF model không tìm thấy: {MODEL_PATH}")
else:
    try:
        obj = joblib.load(MODEL_PATH)
        MODEL = obj['model']
        TFIDF_WORD = obj['tfidf_word']
        TFIDF_CHAR = obj['tfidf_char']
        USE_CHAR = obj.get('use_char', True)
        MODEL_NAME = obj.get('model_name', 'tfidf')
        ACCURACY = obj.get('accuracy', 0.0)
        F1_MACRO = obj.get('f1_macro', 0.0)
        MODEL_AVAILABLE = True
    except Exception as e:
        raise RuntimeError(f"❌ Không thể khởi tải TF-IDF model. Chi tiết: {e}")
BERT_PREDICTORS: Dict[str, object] = {}

def get_bert_predictor(sub_model: Optional[str] = None):
    """Get or create a BERTPredictor for the given sub_model key."""
    global BERT_PREDICTORS
    from src.bert_inference import BERTPredictor, list_available_models

    # Normalize key
    key = sub_model.lower() if sub_model else "auto"

    if key in BERT_PREDICTORS:
        return BERT_PREDICTORS[key]

    try:
        predictor = BERTPredictor(model_key=sub_model if sub_model else None)
        # Cache by both the requested key and the resolved key
        BERT_PREDICTORS[key] = predictor
        resolved = predictor.model_key
        if resolved != key:
            BERT_PREDICTORS[resolved] = predictor
        return predictor
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"BERT model '{sub_model or 'auto'}' is not available. "
                   f"Please train it first. Error: {e}"
        )

def _get_available_bert_models() -> list:
    """Return list of available BERT sub-models."""
    try:
        from src.bert_inference import list_available_models
        return list_available_models()
    except Exception:
        return []

def _get_bert_models_info() -> list:
    """Return info about all available BERT sub-models."""
    models_info = []
    available = _get_available_bert_models()
    for model_key in available:
        try:
            predictor = get_bert_predictor(model_key)
            models_info.append({
                "model_key": predictor.model_key,
                "folder": model_key,
                "accuracy": round(predictor.accuracy, 4),
                "f1_macro": round(predictor.f1_macro, 4),
            })
        except Exception:
            pass
    return models_info

# Try preload silently
try:
    from src.bert_inference import BERTPredictor, list_available_models
    for _mk in list_available_models():
        try:
            _p = BERTPredictor(model_key=_mk)
            BERT_PREDICTORS[_mk] = _p
            BERT_PREDICTORS[_p.model_key] = _p
        except Exception:
            pass
    
    if BERT_PREDICTORS:
        first_key = list(BERT_PREDICTORS.keys())[0]
        BERT_PREDICTORS["auto"] = BERT_PREDICTORS[first_key]
except Exception:
    pass

app = FastAPI(
    title="🤖 SGU Sentiment Analytics Engine",
    description="Hệ thống AI phân tích sắc thái văn bản - Đồ án Nghiên cứu Khoa học",
    version="2.0.0", 
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


EMOJI = {"POSITIVE": "💚 👍", "NEUTRAL": "💛 💬", "NEGATIVE": "❤️ 👎"}


# ── Schemas ───────────────────────────────────────────────────────
# ── Schemas ───────────────────────────────────────────────────────
class SingleRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="Nội dung văn bản cần phân tích cảm xúc")
    model: str = Field("tfidf", description="Model to use: 'tfidf' or 'bert'")
    sub_model: str = Field("demo", description="BERT sub-model: 'phoBert', 'mBert', 'demo'")

    class Config:
        schema_extra = {
            "example": {
                "text": "chất lượng tuyệt vời giao hàng rất nhanh",
                "model": "tfidf",
                "sub_model": "demo"
            }
        }

class BatchRequest(BaseModel):
    texts: list = Field(..., description="Danh sách các câu văn bản cần phân tích")
    model: str = Field("tfidf", description="Model to use: 'tfidf' or 'bert'")
    sub_model: str = Field("demo", description="BERT sub-model: 'phoBert', 'mBert', 'demo'")

    class Config:
        schema_extra = {
            "example": {
                "texts": ["hàng đẹp lắm", "giao sai màu rồi"],
                "model": "tfidf",
                "sub_model": "demo"
            }
        }

class PredictResult(BaseModel):
    text: str
    sentiment: str
    label: str
    confidence: float = 1.0
    scores: dict = {}

class BatchResult(BaseModel):
    count: int
    results: list
    latency_ms: float
# ── Inference helper ──────────────────────────────────────────────
def _run(texts: List[str]):
    if MODEL is None or TFIDF_WORD is None or TFIDF_CHAR is None:
        raise RuntimeError("TF-IDF model chưa được tải. Vui lòng kiểm tra models/model_sentiment/sentiment_model.joblib")

    Xw = TFIDF_WORD.transform(texts)
    X = hstack([Xw, TFIDF_CHAR.transform(texts)]) if USE_CHAR else Xw
    labels = MODEL.predict(X).tolist()
    confs = (MODEL.predict_proba(X).max(axis=1).round(4).tolist()
             if hasattr(MODEL, "predict_proba") else [None] * len(texts))
    return labels, confs

# ── Endpoints ─────────────────────────────────────────────────────
@app.get("/", tags=["Info"])
def root():
    available_bert = _get_available_bert_models()
    bert_models_info = _get_bert_models_info()

    # For backward compat: pick first available as "default" bert info
    default_bert_info = bert_models_info[0] if bert_models_info else None

    return {
        "status": "✅ running",
        "model": MODEL_NAME,
        "accuracy": ACCURACY,
        "f1_macro": F1_MACRO,
        "bert_available": len(available_bert) > 0,
        "available_bert_models": available_bert,
        "bert_models_info": bert_models_info,
        "bert_info": default_bert_info,
        "docs": "/docs"
    }

@app.get("/health", tags=["Info"])
def health():
    return {"status": "healthy"}

@app.post("/predict", response_model=PredictResult, tags=["Inference"])
def predict(body: SingleRequest, model: Optional[str] = Query(None, description="Model to use: 'tfidf' or 'bert'")):
    """Phân tích cảm xúc **1 câu**."""
    try:
        model_choice = (body.model or model or "tfidf").lower()
        if model_choice == "bert":
            predictor = get_bert_predictor(body.sub_model)
            res = predictor.predict(body.text)
            return PredictResult(text=body.text,
                                 sentiment=res["sentiment"],
                                 label=res["label"],
                                 confidence=res["confidence"],
                                 scores=res["scores"])

        if not MODEL_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="TF-IDF model chưa có sẵn. Vui lòng chạy src/train.py để tạo models/model_sentiment/sentiment_model.joblib")
        labels, confs = _run([body.text])
        lbl = str(labels[0])
        return PredictResult(text=body.text,
                             sentiment=f"{EMOJI[lbl]} {lbl}",
                             label=lbl,
                             confidence=confs[0])
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/batch", response_model=BatchResult, tags=["Inference"])
def predict_batch(body: BatchRequest, model: Optional[str] = Query(None, description="Model to use: 'tfidf' or 'bert'")):
    """Phân tích cảm xúc **nhiều câu** (tối đa 100)."""
    try:
        model_choice = (body.model or model or "tfidf").lower()
        if model_choice == "bert":
            predictor = get_bert_predictor(body.sub_model)
            t0 = time.perf_counter()
            res_list = predictor.predict_batch(body.texts)
            ms = round((time.perf_counter()-t0)*1000, 2)
            results = [
                PredictResult(text=t,
                              sentiment=res["sentiment"],
                              label=res["label"],
                              confidence=res["confidence"],
                              scores=res["scores"])
                for t, res in zip(body.texts, res_list)
            ]
            return BatchResult(count=len(results), results=results, latency_ms=ms)

        if not MODEL_AVAILABLE:
            raise HTTPException(status_code=503,
                                detail="TF-IDF model chưa có sẵn. Vui lòng chạy src/train.py để tạo models/model_sentiment/sentiment_model.joblib")
        t0 = time.perf_counter()
        labels, confs = _run(body.texts)
        ms = round((time.perf_counter()-t0)*1000, 2)
        results = [
            PredictResult(text=t,
                          sentiment=f"{EMOJI[str(l)]} {l}",
                          label=str(l),
                          confidence=c)
            for t, l, c in zip(body.texts, labels, confs)
        ]
        return BatchResult(count=len(results), results=results, latency_ms=ms)
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, str(e))
    # DÁN ĐOẠN NÀY VÀO DƯỚI CÙNG CỦA FILE api/main.py
if __name__ == "__main__":
    import uvicorn
    print("🚀 Đang khởi chạy Server FastAPI trên cổng 8000...")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)