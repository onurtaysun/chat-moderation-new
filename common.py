"""Shared code: text normalization, model definitions, and predictor loading.

train.py and app.py both import from here, so the models that are measured
are exactly the models that run in the application.
"""
import os
import re

import numpy as np
import torch
import torch.nn as nn

ART = os.environ.get("ART_DIR", "artifacts")
BERT_NAME = "distilbert-base-uncased"

NAMES = [
    "TF-IDF + LogReg (classical baseline)",
    "word2vec + MLP (own neural net)",
    "DistilBERT FP32 (Transformer)",
    "DistilBERT INT8 (quantized Transformer)",
]

# ---------------------------------------------------------------- Area B: regex normalization
LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s"})
URL = re.compile(r"https?://\S+|www\.\S+")
REPEAT = re.compile(r"(.)\1{2,}")
NONALPHA = re.compile(r"[^a-z\s]")


def normalize(text: str) -> str:
    """Lowercase, drop URLs, undo leetspeak, squeeze repeated letters, keep a-z only."""
    text = URL.sub(" ", str(text).lower())
    words = [w.translate(LEET) if re.search(r"[a-z]", w) else w for w in text.split()]
    text = REPEAT.sub(r"\1\1", " ".join(words))
    text = NONALPHA.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------- Area A: word2vec + feedforward net
class MLP(nn.Module):
    def __init__(self, d: int = 100):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 2))

    def forward(self, x):
        return self.net(x)


def avg_vec(text: str, kv) -> np.ndarray:
    vs = [kv[w] for w in normalize(text).split() if w in kv]
    return np.mean(vs, axis=0).astype(np.float32) if vs else np.zeros(kv.vector_size, np.float32)


# ---------------------------------------------------------------- Area C / D: DistilBERT + INT8
@torch.no_grad()
def bert_proba(model, tok, texts, device="cpu", bs=64, max_len=128):
    model.eval()
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([normalize(t) for t in texts[i:i + bs]], truncation=True,
                  max_length=max_len, padding=True, return_tensors="pt").to(device)
        probs = torch.softmax(model(**enc).logits.float(), dim=-1)[:, 1]
        out.append(probs.cpu().numpy())
    return np.concatenate(out)


def quantize(model):
    """Post-training dynamic INT8 quantization of all Linear layers (CPU)."""
    return torch.ao.quantization.quantize_dynamic(model.cpu().eval(), {nn.Linear}, dtype=torch.qint8)


def load_predictors(art: str = ART):
    """Returns ({name: fn(text) -> P(offensive)}, {'qbert': quantized model})."""
    import joblib
    from gensim.models import KeyedVectors
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tf = joblib.load(f"{art}/tfidf_lr.joblib")
    kv = KeyedVectors.load(f"{art}/w2v.kv")
    mlp = MLP(kv.vector_size)
    mlp.load_state_dict(torch.load(f"{art}/w2v_mlp.pt", map_location="cpu"))
    mlp.eval()
    tok = AutoTokenizer.from_pretrained(f"{art}/bert")
    bert = AutoModelForSequenceClassification.from_pretrained(f"{art}/bert").eval()
    qbert = quantize(bert)  # returns a quantized copy; `bert` stays FP32 (moved to CPU)

    @torch.no_grad()
    def p_mlp(t):
        x = torch.from_numpy(avg_vec(t, kv))[None]
        return float(torch.softmax(mlp(x), -1)[0, 1])

    preds = {
        NAMES[0]: lambda t: float(tf["clf"].predict_proba(tf["vec"].transform([normalize(t)]))[0, 1]),
        NAMES[1]: p_mlp,
        NAMES[2]: lambda t: float(bert_proba(bert, tok, [t])[0]),
        NAMES[3]: lambda t: float(bert_proba(qbert, tok, [t])[0]),
    }
    return preds, {"qbert": qbert}
