# Real-Time Chat Moderation (SWE015 Homework, Use Case 4)

Author: Onur Tayşun, 2309111040
Demo video: [video link]

Flags offensive chat messages live (clean / offensive + confidence), masks flagged messages, and logs the
response time of every message. Four models of increasing sophistication are compared on the same test set;
the application runs the **quantized Transformer** by default.

## Model ladder

| # | Model | Techniques | Area |
|---|-------|-----------|------|
| 1 | TF-IDF + logistic regression | regex normalization (leetspeak, repeated letters, URLs), tf-idf | B |
| 2 | Own word2vec + feedforward net | word2vec trained on the training messages, MLP trained with SGD + backprop | A |
| 3 | DistilBERT fine-tuned | WordPiece tokenizer, attention, classification head | C |
| 4 | DistilBERT INT8 | post-training dynamic quantization, size/latency benchmark | D |

## Quick start (no training needed)

```bash
git clone https://github.com/onurtaysun/chat-moderation-new
cd chat-moderation-new
pip install -r requirements.txt
pip install gdown
gdown 1S6fp7uzVwHrkhtDUwf0bP3uNY5gsiM2P -O artifacts.zip
unzip -q artifacts.zip      # run in the repo root; creates the artifacts/ folder
python app.py               # add --share for a public link (Colab)
```

The trained weights are not in this repo (too large). The `gdown` command above downloads them from
[Google Drive](https://drive.google.com/file/d/1S6fp7uzVwHrkhtDUwf0bP3uNY5gsiM2P/view?usp=drive_link).

## Train + evaluate from scratch (GPU recommended, e.g. a free Colab GPU; ~30-40 min)

```bash
python train.py                  # downloads the dataset from the Hugging Face Hub
python train.py --csv train.csv  # or use the Kaggle Jigsaw file (columns: comment_text, toxic)
python train.py --skip_train     # only re-evaluate the models already saved in artifacts/
```

This writes `artifacts/` (models, `results.json`, `results.md`).

## Data

Jigsaw-style toxic comment classification (binary: offensive vs clean). 24,000 random messages are sampled
(seed 42): 20,000 train / 4,000 test, stratified. All four models are evaluated on the same 4,000 test messages.
Source: Jigsaw Toxic Comment Classification Challenge (Kaggle / Conversation AI), via the Hugging Face mirror
`tcapelle/jigsaw-toxic-comment-classification-challenge`.

## Architecture

```
user message
  -> regex normalization (common.normalize)
  -> one of 4 models: TF-IDF+LogReg | word2vec+MLP | DistilBERT FP32 | DistilBERT INT8 (default)
  -> P(offensive) >= 0.5 ? mask message with * : show as is
  -> Gradio UI: masked message, confidence, side-by-side model table, response-time log
```

Everything runs on CPU in one Python process (`app.py`). Code: `common.py` (normalization, models),
`train.py` (training + evaluation), `app.py` (Gradio app).

## Results (my own run)

| Model | Macro-F1 | Size (MB) | Latency (ms/msg) |
|---|---|---|---|
| TF-IDF + LogReg (baseline) | 0.8221 | 3.9 | 0.97 |
| word2vec + MLP | 0.7526 | 10.7 | 0.66 |
| DistilBERT FP32 | 0.8743 | 267.8 | 88.2 |
| DistilBERT INT8 (quantized) | 0.8875 | 138.7 | 46.12 |

Same 4,000-message test set for all models. Latency: mean of 4,000 single-message calls on CPU
(Intel Xeon @ 2.00 GHz, 2 logical CPUs, 1 torch thread).

## AI usage

Claude (Anthropic) drafted the first version of the code and the slides and helped me debug; I ran the training and measurements myself.
