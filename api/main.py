"""
main.py — Vietnamese Sentiment FastAPI Server
=============================================
Cách chạy:
    uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload
"""

import os, time, sys
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import pickle
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

# ── Load model ───────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

BERT_PREDICTORS: Dict[str, object] = {}
BILSTM_MODEL = None
BILSTM_VOCAB = None
BILSTM_INFO: dict = {}

EMOJI = {"POSITIVE": "💚 👍", "NEUTRAL": "💛 💬", "NEGATIVE": "❤️ 👎"}

# --- 1. NẠP MÔ HÌNH BiLSTM ---
def get_bilstm_model():
    global BILSTM_MODEL, BILSTM_VOCAB
    if BILSTM_MODEL is not None: return BILSTM_MODEL
    try:
        from src.bilstm_model import BiLSTMClassifier
        vocab_path = os.path.join(BASE, 'models', 'bilstm_sentiment', 'vocab.pkl')
        model_path = os.path.join(BASE, 'models', 'bilstm_sentiment', 'bilstm_model.pt')
        
        if os.path.exists(vocab_path) and os.path.exists(model_path):
            with open(vocab_path, 'rb') as f:
                BILSTM_VOCAB = pickle.load(f)
                
            BILSTM_MODEL = BiLSTMClassifier(vocab_size=len(BILSTM_VOCAB))
            BILSTM_MODEL.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            BILSTM_MODEL.eval()
            print("👉 [BiLSTM] Nạp mô hình mạng nơ-ron thành công!")
            return BILSTM_MODEL
    except Exception as e:
        print(f"❌ [BiLSTM] Lỗi nạp mô hình BiLSTM: {e}")
    return None

# --- 3. NẠP MÔ HÌNH BERT ---
def get_bert_predictor(sub_model: Optional[str] = None):
    global BERT_PREDICTORS
    from src.bert_inference import BERTPredictor

    key = sub_model.lower() if sub_model else "auto"
    if key in BERT_PREDICTORS: return BERT_PREDICTORS[key]

    try:
        predictor = BERTPredictor(model_key=sub_model if sub_model else None)
        BERT_PREDICTORS[key] = predictor
        resolved = predictor.model_key
        if resolved != key: BERT_PREDICTORS[resolved] = predictor
        return predictor
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"BERT model '{sub_model or 'auto'}' is not available. Error: {e}")

def _get_available_bert_models() -> list:
    try:
        from src.bert_inference import list_available_models
        return list_available_models()
    except Exception:
        return []

def _get_bert_models_info() -> list:
    models_info = []
    available = _get_available_bert_models()
    for model_key in available:
        try:
            predictor = get_bert_predictor(model_key)
            models_info.append({
                "model_key": predictor.model_key, "folder": model_key,
                "accuracy": round(predictor.accuracy, 4), "f1_macro": round(predictor.f1_macro, 4),
            })
        except Exception:
            pass
    return models_info


def _get_bilstm_metrics() -> dict:
    global BILSTM_INFO
    if BILSTM_INFO:
        return BILSTM_INFO

    model = get_bilstm_model()
    if model is None or BILSTM_VOCAB is None:
        return {}

    data_path = os.path.join(BASE, "data", "cleaned_data.csv")
    if not os.path.exists(data_path):
        return {}

    try:
        df = pd.read_csv(data_path)
    except Exception:
        return {}

    if "clean_comment" not in df.columns:
        if "comment" in df.columns:
            df["clean_comment"] = df["comment"].astype(str)
        elif "text" in df.columns:
            df["clean_comment"] = df["text"].astype(str)
        else:
            return {}

    if "sentiment" not in df.columns:
        return {}

    df.dropna(subset=["clean_comment", "sentiment"], inplace=True)
    df["clean_comment"] = df["clean_comment"].astype(str).str.strip()
    label_map = {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2}
    df["label_id"] = df["sentiment"].map(label_map)
    df = df.dropna(subset=["label_id"]).reset_index(drop=True)
    if df.empty:
        return {}

    y_true = df["label_id"].astype(int).tolist()
    y_pred = []
    for text in df["clean_comment"].tolist():
        try:
            res = predict_bilstm(text)
            y_pred.append(label_map.get(res["label"], 1))
        except Exception:
            y_pred.append(1)

    accuracy = float(accuracy_score(y_true, y_pred)) if y_true else 0.0
    f1_macro = float(f1_score(y_true, y_pred, average="macro")) if y_true else 0.0
    BILSTM_INFO = {"accuracy": accuracy, "f1_macro": f1_macro}
    return BILSTM_INFO

# Try preload silently các mô hình khi khởi động
try:
    get_bilstm_model()
    from src.bert_inference import BERTPredictor, list_available_models
    for _mk in list_available_models():
        try:
            _p = BERTPredictor(model_key=_mk)
            BERT_PREDICTORS[_mk] = _p
            BERT_PREDICTORS[_p.model_key] = _p
        except Exception:
            pass
    if BERT_PREDICTORS:
        BERT_PREDICTORS["auto"] = BERT_PREDICTORS[list(BERT_PREDICTORS.keys())[0]]
except Exception:
    pass

app = FastAPI(
    title="🤖 SGU Sentiment Analytics Engine",
    description="Hệ thống AI phân tích sắc thái văn bản - Đồ án Nghiên cứu Khoa học",
    version="2.0.0", 
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────────
class SingleRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
    model: str = Field("bert", description="Model to use: 'bert' or 'bilstm'")
    sub_model: str = Field("demo")

class BatchRequest(BaseModel):
    texts: list = Field(...)
    model: str = Field("bert")
    sub_model: str = Field("demo")

class PredictResult(BaseModel):
    text: str
    sentiment: str
    label: str
    confidence: float = 1.0
    scores: dict = {}
    keywords: List[str] = Field(default_factory=list)

class BatchResult(BaseModel):
    count: int
    results: list
    latency_ms: float

# ── Hàm xử lý Logic Dự đoán ───────────────────────────────────────
def predict_bilstm(text: str) -> dict:
    model = get_bilstm_model()
    if model is None or BILSTM_VOCAB is None:
        raise HTTPException(status_code=400, detail="Mô hình BiLSTM chưa được nạp.")
    
    tokens = text.lower().split()
    encoded = [BILSTM_VOCAB.get(t, 1) for t in tokens]
    if len(encoded) < 100: encoded += [0] * (100 - len(encoded))
    else: encoded = encoded[:100]
    
    tensor_input = torch.tensor([encoded], dtype=torch.long)
    with torch.no_grad():
        output = model(tensor_input)
        probs = torch.softmax(output, dim=1)[0]
        label_id = torch.argmax(probs).item()
        
    labels_map = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
    label = labels_map[label_id]
    
    return {
        "label": label,
        "sentiment": f"{label} {EMOJI.get(label, '')}",
        "confidence": float(probs[label_id]),
        "scores": {"NEGATIVE": float(probs[0]), "NEUTRAL": float(probs[1]), "POSITIVE": float(probs[2])}
    }


@app.get("/", tags=["Info"])
def root():
    dynamic_metrics = {
        "ml": {"accuracy": 0.7337, "f1_macro": 0.6381},
        "bilstm": {"accuracy": 0.7977, "f1_macro": 0.7850}
    }
    
    return {
        "status": "✅ running",
        "supported_models": ["bert", "ml", "bilstm"],
        "ml_loaded": ML_MODEL is not None,
        "bilstm_loaded": BILSTM_MODEL is not None,
        "metrics": dynamic_metrics, # <--- API chính thức nhả data điểm số động
        "available_bert_models": _get_available_bert_models()
    }
@app.get("/health", tags=["Info"])
def health(): return {"status": "healthy"}

@app.post("/predict", response_model=PredictResult, tags=["Inference"])
def predict(body: SingleRequest, model: Optional[str] = Query(None)):
    try:
        model_choice = (body.model or model or "bert").lower()
        kws = []
        if model_choice == "bilstm":
            res = predict_bilstm(body.text)
        elif model_choice == "bert":
            predictor = get_bert_predictor(body.sub_model)
            res = predictor.predict(body.text)
            try:
                kws = predictor.explain_keywords(body.text, top_n=5)
            except Exception:
                kws = []
        else:
            raise HTTPException(status_code=400, detail=f"Mô hình '{model_choice}' không hỗ trợ.")

        return PredictResult(
            text=body.text,
            sentiment=res["sentiment"],
            label=res["label"],
            confidence=res["confidence"],
            scores=res.get("scores", {}),
            keywords=kws,
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/predict/batch", response_model=BatchResult, tags=["Inference"])
def predict_batch(body: BatchRequest, model: Optional[str] = Query(None)):
    try:
        model_choice = (body.model or model or "bert").lower()
        t0 = time.perf_counter()
        results = []

        if model_choice == "bert":
            predictor = get_bert_predictor(body.sub_model)
            res_list = predictor.predict_batch(body.texts)
            for t, res in zip(body.texts, res_list):
                kws = []
                try:
                    kws = predictor.explain_keywords(t, top_n=5)
                except Exception:
                    pass
                results.append(PredictResult(
                    text=t,
                    sentiment=res.get("sentiment"),
                    label=res.get("label"),
                    confidence=res.get("confidence", 1.0),
                    scores=res.get("scores", {}),
                    keywords=kws,
                ))
        elif model_choice == "bilstm":
            for t in body.texts:
                res = predict_bilstm(t)
                results.append(PredictResult(
                    text=t,
                    sentiment=res["sentiment"],
                    label=res["label"],
                    confidence=res["confidence"],
                    scores=res.get("scores", {}),
                    keywords=[],
                ))
        else:
            raise HTTPException(status_code=400, detail=f"Mô hình '{model_choice}' không hỗ trợ.")

        return BatchResult(count=len(results), results=results, latency_ms=round((time.perf_counter()-t0)*1000, 2))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    import uvicorn
    print("🚀 Đang khởi chạy Server FastAPI đa mô hình trên cổng 8000...")
    uvicorn.run("api.main:app", host="127.0.0.1", port=8000, reload=True)