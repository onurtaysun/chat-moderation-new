"""Live chat moderation demo. Default model = quantized DistilBERT (INT8).

    python app.py            # local
    python app.py --share    # public link (e.g. from Colab)
"""
import argparse
import time

import gradio as gr

from common import NAMES, load_predictors

P, _ = load_predictors()
DEFAULT = NAMES[3]  # quantized Transformer (requirement R5)
LOG_HEADERS = ["message shown", "label", "P(offensive)", "model", "latency (ms)"]


def moderate(msg, model_name, log):
    log = log or []
    if not msg or not msg.strip():
        return "", "", [], log, log
    rows = []
    for name, fn in P.items():
        t0 = time.perf_counter()
        p = fn(msg)
        ms = (time.perf_counter() - t0) * 1000
        rows.append([name, "offensive" if p >= 0.5 else "clean", round(p, 3), round(ms, 1)])
    chosen = next(r for r in rows if r[0] == model_name)
    toxic = chosen[1] == "offensive"
    shown = "*" * len(msg) if toxic else msg
    label = f"**{'🚫 OFFENSIVE (masked)' if toxic else '✅ clean'}** — confidence {max(chosen[2], 1 - chosen[2]):.0%} · {chosen[3]} ms · {model_name}"
    log = [[shown, chosen[1], chosen[2], model_name, chosen[3]]] + log
    return shown, label, rows, log, log


with gr.Blocks(title="Chat moderation") as demo:
    gr.Markdown("# Real-time chat moderation\nType a message. It is labelled live; offensive messages are masked.")
    state = gr.State([])
    with gr.Row():
        with gr.Column(scale=2):
            msg = gr.Textbox(label="Chat message", placeholder="Type a message and press Enter…")
            model = gr.Radio(NAMES, value=DEFAULT, label="Model")
            shown = gr.Textbox(label="What other players see", interactive=False)
            verdict = gr.Markdown()
        with gr.Column(scale=3):
            compare = gr.Dataframe(headers=["model", "label", "P(offensive)", "latency (ms)"],
                                   label="All four models, side by side", interactive=False)
            logdf = gr.Dataframe(headers=LOG_HEADERS, label="Response-time log", interactive=False)
    msg.submit(moderate, [msg, model, state], [shown, verdict, compare, logdf, state])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true")
    demo.launch(share=ap.parse_args().share)
