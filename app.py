"""
app.py — Smriti Web Interface
==============================
Run:
    pip install flask sentencepiece torch
"""

import os
import json
import torch
import sentencepiece as spm
from flask import Flask, request, Response, render_template, jsonify

from model import HindiLM

CHECKPOINT = "./checkpoints/best.pt"
TOKENIZER  = "./tokenizer/hindi_spm.model"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
HOST       = "0.0.0.0"
PORT       = 5000

print("Loading tokenizer...")
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER)

print(f"Loading model from {CHECKPOINT}...")
device = torch.device(DEVICE)
state  = torch.load(CHECKPOINT, map_location=device)
cfg    = state["cfg"]

model = HindiLM(
    vocab_size    = cfg["vocab_size"],
    embed_dim     = cfg["embed_dim"],
    hidden_dim    = cfg["hidden_dim"],
    num_layers    = cfg["num_layers"],
    dropout_emb   = cfg["dropout_emb"],
    dropout_inter = cfg["dropout_inter"],
    dropout_out   = cfg["dropout_out"],
    use_grad_ckpt = False,
).to(device)

model.load_state_dict(state["model_state"])
model.eval()

MODEL_INFO = {
    "parameters"  : f"{cfg['vocab_size'] // 1000}k vocab · ~49.7M params",
    "step"        : f"{state['step']:,} steps",
    "val_ppl"     : f"{state['val_perplexity']:.2f}",
    "architecture": "3-layer LSTM · hidden 1024 · weight-tied",
    "trained_on"  : "Sangraha Verified · 1B Hindi tokens",
}

print(f"Model ready on {DEVICE}. Val ppl: {state['val_perplexity']:.2f}")

app = Flask(__name__, template_folder=".")


@app.route("/")
def index():
    return render_template("index.html", info=MODEL_INFO)


@app.route("/generate", methods=["POST"])
def generate():
    data        = request.get_json()
    prompt      = data.get("prompt", "").strip()
    temperature = float(data.get("temperature", 1.0))
    max_new     = int(data.get("max_new", 150))
    top_k       = int(data.get("top_k", 50))
    top_p       = float(data.get("top_p", 0.95))

    if not prompt:
        return jsonify({"error": "Empty prompt"}), 400

    temperature = max(0.1, min(temperature, 2.0))
    max_new     = max(20,  min(max_new, 400))

    def stream_tokens():
        with torch.no_grad():
            ids        = sp.encode(prompt, out_type=int)
            prompt_ids = torch.tensor(
                [[sp.bos_id()] + ids], dtype=torch.long, device=device
            )

            hidden = model.init_hidden(1, device)

            if prompt_ids.shape[1] > 1:
                _, hidden = model(prompt_ids[:, :-1], hidden)
                hidden = model.detach_hidden(hidden)

            input_id  = prompt_ids[:, -1:]
            generated = [] 

            for _ in range(max_new):
                logits, hidden = model(input_id, hidden)
                hidden = model.detach_hidden(hidden)
                logits = logits[:, -1, :]

                if temperature != 1.0:
                    logits = logits / temperature

                if top_k > 0:
                    values, _ = torch.topk(logits, top_k)
                    logits = logits.masked_fill(
                        logits < values[:, -1:], float("-inf")
                    )

                if top_p < 1.0:
                    sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                    cumprobs = torch.cumsum(
                        torch.softmax(sorted_logits, dim=-1), dim=-1
                    )
                    sorted_logits[
                        cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p
                    ] = float("-inf")
                    logits = torch.zeros_like(logits).scatter_(
                        1, sorted_idx, sorted_logits
                    )

                probs   = torch.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)
                token   = next_id.item()

                if token == sp.eos_id():
                    break

                generated.append(token)
                input_id = next_id
              
                prev_text = sp.decode(generated[:-1]) if len(generated) > 1 else ""
                curr_text = sp.decode(generated)
                new_text  = curr_text[len(prev_text):]

                if new_text:
                    yield f"data: {json.dumps({'token': new_text})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

    return Response(stream_tokens(), mimetype="text/event-stream")


if __name__ == "__main__":
    print(f"\nSmriti running at http://localhost:{PORT}\n")
    app.run(host=HOST, port=PORT, debug=False, threaded=True)
