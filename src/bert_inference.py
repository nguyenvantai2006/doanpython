"""
bert_inference.py — Load và chạy BERT model
============================================
Hỗ trợ nhiều sub-model: phoBert, mBert, demo.

Cách dùng:
    from src.bert_inference import BERTPredictor

    # Tự động chọn model có sẵn (ưu tiên phoBert > mBert > demo > legacy)
    p = BERTPredictor()

    # Chọn sub-model cụ thể
    p = BERTPredictor(model_key="phoBert")
    p = BERTPredictor(model_key="mBert")
    p = BERTPredictor(model_key="demo")

    print(p.predict("giao hàng nhanh chất lượng tốt"))
"""
import os, json
import torch
import torch.nn as nn
from typing import Optional

BASE     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BERT_DIR = os.path.join(BASE, "models", "bert_sentiment")

# Mapping from model_key (case-insensitive) to subfolder name
SUBFOLDER_MAP = {
    "phobert": "phobert",
    "mbert":   "mbert",
    "demo":    "demo",
}

# Priority order when auto-detecting available models
AUTO_PRIORITY = ["phobert", "mbert", "demo"]

LABEL_MAP = {0: "NEGATIVE", 1: "NEUTRAL", 2: "POSITIVE"}
EMOJI_MAP = {"NEGATIVE": "😞", "NEUTRAL": "😐", "POSITIVE": "😊"}


# ── Helpers ───────────────────────────────────────────────────────
def list_available_models() -> list[str]:
    """Return list of available sub-model folder names that contain bert_model.pt."""
    available = []
    for folder in AUTO_PRIORITY:
        ckpt = os.path.join(BERT_DIR, folder, "bert_model.pt")
        if os.path.exists(ckpt):
            available.append(folder)
    # Also check legacy root path
    legacy_ckpt = os.path.join(BERT_DIR, "bert_model.pt")
    if os.path.exists(legacy_ckpt) and "legacy" not in available:
        available.append("legacy")
    return available


def _resolve_model_dir(model_key: Optional[str] = None) -> tuple[str, str]:
    """Resolve model_key to (model_dir, resolved_key).

    Priority:
      1. If model_key is given, look up subfolder directly.
      2. If not found, auto-detect first available in priority order.
      3. Fallback to legacy root path.
    """
    if model_key:
        # Normalize key
        folder = SUBFOLDER_MAP.get(model_key.lower(), model_key)
        candidate = os.path.join(BERT_DIR, folder)
        if os.path.exists(os.path.join(candidate, "bert_model.pt")):
            return candidate, folder

    # Auto-detect
    for folder in AUTO_PRIORITY:
        candidate = os.path.join(BERT_DIR, folder)
        if os.path.exists(os.path.join(candidate, "bert_model.pt")):
            return candidate, folder

    # Legacy fallback
    legacy = os.path.join(BERT_DIR, "bert_model.pt")
    if os.path.exists(legacy):
        return BERT_DIR, "legacy"

    raise FileNotFoundError(
        f"Không tìm thấy BERT model nào trong {BERT_DIR}\n"
        "Chạy: python src/train_bert.py --model demo"
    )


class SimpleViTokenizer:
    def __init__(self, word2id, max_len=128):
        self.word2id = word2id
        self.max_len = max_len

    def __call__(self, text, max_length=128, padding="max_length",
                 truncation=True, return_tensors=None):
        tokens = [self.word2id.get("[CLS]", 2)]
        for w in text.lower().split():
            tokens.append(self.word2id.get(w, self.word2id.get("[UNK]", 1)))
        tokens.append(self.word2id.get("[SEP]", 3))
        if truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        mask = [1] * len(tokens)
        while len(tokens) < max_length:
            tokens.append(0); mask.append(0)
        input_ids      = torch.tensor([tokens], dtype=torch.long)
        attention_mask = torch.tensor([mask],   dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _BERTClf(nn.Module):
    def __init__(self, bert_model, num_labels=3, dropout=0.3):
        super().__init__()
        self.bert   = bert_model
        hidden      = bert_model.config.hidden_size
        self.drop   = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        return self.linear(self.drop(pooled))


class BERTPredictor:
    def __init__(self, model_key: Optional[str] = None,
                 ckpt_path: Optional[str] = None, device=None):
        """
        Args:
            model_key: Sub-model to load: "phoBert", "mBert", "demo", or None (auto).
            ckpt_path: Explicit checkpoint path. Overrides model_key resolution.
            device:    Torch device.
        """
        # Resolve paths
        if ckpt_path and os.path.exists(ckpt_path):
            model_dir = os.path.dirname(ckpt_path)
            resolved_key = model_key or "unknown"
        else:
            model_dir, resolved_key = _resolve_model_dir(model_key)
            ckpt_path = os.path.join(model_dir, "bert_model.pt")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(
                f"Chưa tìm thấy model: {ckpt_path}\n"
                "Chạy: python src/train_bert.py --model demo"
            )

        self.device = device or torch.device("cpu")
        ck = torch.load(ckpt_path, map_location=self.device)

        # Build model
        model_type = ck["bert_config"].get("model_type", "bert")
        if model_type == "roberta":
            from transformers import RobertaConfig, RobertaModel
            cfg  = RobertaConfig(**ck["bert_config"])
            bert = RobertaModel(cfg)
        else:
            from transformers import BertConfig, BertModel
            cfg  = BertConfig(**ck["bert_config"])
            bert = BertModel(cfg)
        clf   = _BERTClf(bert)
        clf.load_state_dict(ck["model_state"])
        clf.to(self.device).eval()
        self.model = clf

        # Load tokenizer
        vocab_path = os.path.join(model_dir, "tokenizer_vocab.json")
        if os.path.exists(vocab_path):
            with open(vocab_path, encoding="utf-8") as f:
                word2id = json.load(f)
            self.tokenizer = SimpleViTokenizer(word2id)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_dir)

        self.label_map = LABEL_MAP
        self.accuracy  = ck.get("accuracy", 0)
        self.f1_macro  = ck.get("f1_macro",  0)
        self.model_key = ck.get("model_key", resolved_key)
        self.model_dir = model_dir

    @torch.no_grad()
    def predict(self, text: str) -> dict:
        enc    = self.tokenizer(text, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
        ids    = enc["input_ids"].to(self.device)
        mask   = enc["attention_mask"].to(self.device)
        logits = self.model(ids, mask)
        probs  = torch.softmax(logits, dim=1)
        idx    = probs.argmax(dim=1).item()
        label  = LABEL_MAP[idx]
        return {
            "label":      label,
            "sentiment":  f"{EMOJI_MAP[label]} {label}",
            "confidence": round(probs[0, idx].item(), 4),
            "scores": {
                "NEGATIVE": round(probs[0, 0].item(), 4),
                "NEUTRAL":  round(probs[0, 1].item(), 4),
                "POSITIVE": round(probs[0, 2].item(), 4),
            }
        }

    @torch.no_grad()
    def predict_batch(self, texts: list) -> list:
        return [self.predict(t) for t in texts]