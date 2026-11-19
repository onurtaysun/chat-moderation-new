"""Train the 4-model ladder, then measure F1 / size / latency on ONE shared test set.

Usage:
    python train.py                       # downloads a Jigsaw-style toxicity dataset from the HF Hub
    python train.py --csv train.csv       # or use the Kaggle Jigsaw file (columns: comment_text, toxic)

Outputs (in ./artifacts): trained models, results.json, results.md
"""
import argparse
import json
import math
import os
import platform
import time

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gensim.models import KeyedVectors, Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          get_linear_schedule_with_warmup)

from common import BERT_NAME, MLP, NAMES, avg_vec, load_predictors, normalize

SEED = 42


def load_data(csv, n):
    if csv:
        df = pd.read_csv(csv)
    else:
        from datasets import load_dataset
        df = load_dataset("tcapelle/jigsaw-toxic-comment-classification-challenge", split="train").to_pandas()
    tcol = next(c for c in ("comment_text", "text") if c in df.columns)
    lcol = next(c for c in ("toxic", "label") if c in df.columns)
    df = df.rename(columns={tcol: "text", lcol: "label"})[["text", "label"]].dropna()
    df["label"] = (df["label"] > 0).astype(int)
    df = df[df["text"].map(normalize).str.len() > 0]
    return df.sample(n=min(n, len(df)), random_state=SEED).reset_index(drop=True)


def size_mb(*paths):
    total = 0
    for p in paths:
        if os.path.isdir(p):
            total += sum(os.path.getsize(os.path.join(p, f)) for f in os.listdir(p))
        elif os.path.exists(p):
            total += os.path.getsize(p)
    return total / 1e6


def cpu_name():
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--n", type=int, default=24000, help="total examples sampled")
    ap.add_argument("--test", type=int, default=4000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--art", default="artifacts")
    ap.add_argument("--skip_train", action="store_true", help="reuse saved models in --art, only evaluate")
    a = ap.parse_args()
    os.makedirs(a.art, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ------------------------------------------------------------------ data
    df = load_data(a.csv, a.n)
    tr, te = train_test_split(df, test_size=a.test, stratify=df["label"], random_state=SEED)
    tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
    print(f"train={len(tr)} test={len(te)} toxic_rate_train={tr.label.mean():.3f} toxic_rate_test={te.label.mean():.3f}")
    tr_norm = [normalize(t) for t in tr.text]
    y = tr.label.values

    if not a.skip_train:
        # ------------------------------------------------------------------ 1) classical baseline
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(vec.fit_transform(tr_norm), y)
        joblib.dump({"vec": vec, "clf": clf}, f"{a.art}/tfidf_lr.joblib")

        # ------------------------------------------------------------------ 2) own word2vec + feedforward net (SGD / backprop)
        w2v = Word2Vec([t.split() for t in tr_norm], vector_size=100, window=5, min_count=2, workers=4, epochs=10, seed=SEED)
        w2v.wv.save(f"{a.art}/w2v.kv")
        X = torch.tensor(np.stack([avg_vec(t, w2v.wv) for t in tr.text]))
        yt = torch.tensor(y)
        mlp = MLP(100)
        weight = torch.tensor([1.0, float((y == 0).sum() / max((y == 1).sum(), 1))])
        loss_fn = nn.CrossEntropyLoss(weight=weight)
        opt = torch.optim.SGD(mlp.parameters(), lr=0.05, momentum=0.9)
        for ep in range(30):
            mlp.train()
            perm = torch.randperm(len(X))
            for i in range(0, len(X), 256):
                idx = perm[i:i + 256]
                opt.zero_grad()
                loss_fn(mlp(X[idx]), yt[idx]).backward()
                opt.step()
        torch.save(mlp.state_dict(), f"{a.art}/w2v_mlp.pt")

        # ------------------------------------------------------------------ 3) Transformer: fine-tune DistilBERT
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        if dev == "cpu":
            print("WARNING: no GPU found, BERT fine-tuning will be very slow. Use Colab (T4) or lower --n.")
        tok = AutoTokenizer.from_pretrained(BERT_NAME)
        bert = AutoModelForSequenceClassification.from_pretrained(BERT_NAME, num_labels=2).to(dev)
        bs = 32
        steps = a.epochs * math.ceil(len(tr_norm) / bs)
        opt = torch.optim.AdamW(bert.parameters(), lr=5e-5, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
        scaler = torch.cuda.amp.GradScaler(enabled=dev == "cuda")
        bert_loss = nn.CrossEntropyLoss(weight=weight.to(dev))
        for ep in range(a.epochs):
            bert.train()
            perm = np.random.permutation(len(tr_norm))
            for k, i in enumerate(range(0, len(perm), bs)):
                idx = perm[i:i + bs]
                enc = tok([tr_norm[j] for j in idx], truncation=True, max_length=128,
                          padding=True, return_tensors="pt").to(dev)
                labels = torch.tensor(y[idx]).to(dev)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=dev == "cuda"):
                    logits = bert(**enc).logits
                loss = bert_loss(logits.float(), labels)
                opt.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                sched.step()
                if k % 100 == 0:
                    print(f"epoch {ep + 1} step {k} loss {loss.item():.4f}")
        bert.save_pretrained(f"{a.art}/bert")
        tok.save_pretrained(f"{a.art}/bert")
        del bert, opt
        if dev == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------ 4) quantize + evaluate everything on CPU, same test set
    preds, parts = load_predictors(a.art)
    texts, truth = te.text.tolist(), te.label.values
    results, failures = {}, []
    for name in NAMES:
        fn = preds[name]
        for t in texts[:10]:  # warm-up
            fn(t)
        probs, lat = [], []
        for t in texts:       # one message at a time, like a chat server; averaged over the whole test set
            t0 = time.perf_counter()
            probs.append(fn(t))
            lat.append((time.perf_counter() - t0) * 1000)
        pred = (np.array(probs) >= 0.5).astype(int)
        results[name] = {"macro_f1": round(float(f1_score(truth, pred, average="macro")), 4),
                         "latency_ms": round(float(np.mean(lat)), 2), "n_runs": len(lat)}
        print(name, results[name])
        if name == NAMES[3]:
            wrong = np.where(pred != truth)[0][:6]
            failures = [{"text": texts[i][:300], "true": int(truth[i]), "p_offensive": round(float(probs[i]), 3)} for i in wrong]

    torch.save(parts["qbert"].state_dict(), f"{a.art}/_qbert_tmp.pt")
    sizes = {
        NAMES[0]: size_mb(f"{a.art}/tfidf_lr.joblib"),
        NAMES[1]: size_mb(f"{a.art}/w2v.kv", f"{a.art}/w2v.kv.vectors.npy", f"{a.art}/w2v_mlp.pt"),
        NAMES[2]: size_mb(f"{a.art}/bert/model.safetensors") if os.path.exists(f"{a.art}/bert/model.safetensors")
        else size_mb(f"{a.art}/bert/pytorch_model.bin"),
        NAMES[3]: size_mb(f"{a.art}/_qbert_tmp.pt"),
    }
    os.remove(f"{a.art}/_qbert_tmp.pt")
    for n in NAMES:
        results[n]["size_mb"] = round(sizes[n], 1)

    hw = f"{cpu_name()}, {os.cpu_count()} logical CPUs, torch {torch.__version__}, {torch.get_num_threads()} threads, CPU inference"
    out = {"hardware": hw, "n_train": len(tr), "n_test": len(te), "results": results, "failure_examples": failures}
    json.dump(out, open(f"{a.art}/results.json", "w"), indent=2)
    md = ["| Model | Macro-F1 | Size (MB) | Latency (ms/msg) |", "|---|---|---|---|"]
    md += [f"| {n} | {results[n]['macro_f1']} | {results[n]['size_mb']} | {results[n]['latency_ms']} |" for n in NAMES]
    md.append(f"\nTest set: {len(te)} messages (same for all models). Latency: mean over {len(te)} single-message calls. Hardware: {hw}")
    open(f"{a.art}/results.md", "w").write("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
