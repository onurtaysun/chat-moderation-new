# Real-Time Chat Moderation (SWE015 Homework, Use Case 4)

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

## Setup (fresh environment)

```bash
pip install -r requirements.txt
```

## Train + evaluate (GPU recommended, e.g. a free Colab T4; ~30-40 min)

```bash
python train.py            # downloads the dataset from the Hugging Face Hub
# or, with the Kaggle Jigsaw file (columns: comment_text, toxic):
python train.py --csv train.csv
```

This writes `artifacts/` (models, `results.json`, `results.md`). Weights are **not** in this repo:
download them from LINK_TO_HF_HUB_OR_DRIVE and unzip as `artifacts/`, or just re-run `train.py`.

## Run the app (single command)

```bash
python app.py              # add --share for a public link (Colab)
```

## Data

Jigsaw-style toxic comment classification (binary: offensive vs clean). 24,000 random messages are sampled
(seed 42): 20,000 train / 4,000 test, stratified. All four models are evaluated on the same 4,000 test messages.
Source: Jigsaw Toxic Comment Classification Challenge (Kaggle / Conversation AI), as mirrored on the HF Hub.

## Architecture

```
user message
  -> regex normalization (common.normalize)
  -> one of 4 models: TF-IDF+LogReg | word2vec+MLP | DistilBERT FP32 | DistilBERT INT8 (default)
  -> P(offensive) >= 0.5 ? mask message with * : show as is
  -> Gradio UI: masked message, confidence, side-by-side model table, response-time log
```
Everything runs on CPU in one Python process (`app.py`).

## Results

See `artifacts/results.md` (produced by `train.py`; every number comes from your own run).

## AI usage

Code scaffolding was drafted with Claude; I ran, checked and take responsibility for all of it.
