"""
inference.py — Interactive Hindi Text Generation
==================================================
Loads a checkpoint and runs interactive generation.
Training can continue in parallel — this script is read-only.

Usage:
    python3 inference.py
    python3 inference.py --checkpoint ./checkpoints/best.pt
    python3 inference.py --checkpoint ./checkpoints/best.pt --temp 0.8 --max_new 150
"""

import argparse
import os
import torch
import sentencepiece as spm

from model import HindiLM

# ── Args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Hindi LM Inference")
parser.add_argument("--checkpoint",  default="./checkpoints/best.pt",    type=str)
parser.add_argument("--tokenizer",   default="./tokenizer/hindi_spm.model", type=str)
parser.add_argument("--max_new",     default=200,   type=int,   help="Max new tokens to generate")
parser.add_argument("--temp",        default=1.0,   type=float, help="Sampling temperature")
parser.add_argument("--top_k",       default=50,    type=int,   help="Top-k filtering")
parser.add_argument("--top_p",       default=0.95,  type=float, help="Nucleus sampling threshold")
parser.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--compare",     action="store_true", help="Generate at all 3 temperatures for comparison")
args = parser.parse_args()

# ── Load tokenizer ────────────────────────────────────────────────────────────

print(f"Loading tokenizer from {args.tokenizer}...")
sp = spm.SentencePieceProcessor()
sp.load(args.tokenizer)
print(f"  Vocab size: {sp.get_piece_size()}")

# ── Load checkpoint ───────────────────────────────────────────────────────────

print(f"\nLoading checkpoint from {args.checkpoint}...")
device = torch.device(args.device)

state = torch.load(args.checkpoint, map_location=device)
cfg   = state["cfg"]

print(f"  Trained for  : {state['step']:,} steps")
print(f"  Epoch        : {state['epoch']}")
print(f"  Val ppl      : {state['val_perplexity']:.2f}")

# ── Build model ───────────────────────────────────────────────────────────────

model = HindiLM(
    vocab_size    = cfg["vocab_size"],
    embed_dim     = cfg["embed_dim"],
    hidden_dim    = cfg["hidden_dim"],
    num_layers    = cfg["num_layers"],
    dropout_emb   = cfg["dropout_emb"],
    dropout_inter = cfg["dropout_inter"],
    dropout_out   = cfg["dropout_out"],
    use_grad_ckpt = False,    # not needed for inference
).to(device)

model.load_state_dict(state["model_state"])
model.eval()

param_count = model.count_parameters()["total"]
print(f"  Parameters   : {param_count:,}")
print(f"  Device       : {device}")

# ── Generation helper ─────────────────────────────────────────────────────────

def generate(prompt: str, temp: float, max_new: int, top_k: int, top_p: float) -> str:
    ids        = sp.encode(prompt, out_type=int)
    prompt_ids = torch.tensor([[sp.bos_id()] + ids], dtype=torch.long, device=device)

    out_ids = model.generate(
        prompt_ids,
        max_new     = max_new,
        temperature = temp,
        top_k       = top_k,
        top_p       = top_p,
        eos_id      = sp.eos_id(),
    )

    # Decode — strip BOS
    return sp.decode(out_ids[1:])


def print_separator():
    print("\n" + "─" * 70)


# ── Interactive loop ──────────────────────────────────────────────────────────

print("\n" + "═" * 70)
print("  Hindi LM — Interactive Inference")
print(f"  Checkpoint : step {state['step']:,}  |  val ppl {state['val_perplexity']:.2f}")
print(f"  Temperature: {args.temp}  |  top_k: {args.top_k}  |  top_p: {args.top_p}")
print("  Type a Hindi prompt and press Enter.")
print("  Commands:")
print("    :temp 0.8     — change temperature")
print("    :max 300      — change max new tokens")
print("    :compare      — generate at temps 0.7, 1.0, 1.2 for comparison")
print("    :quit         — exit")
print("═" * 70 + "\n")

current_temp    = args.temp
current_max_new = args.max_new

while True:
    try:
        prompt = input(">>> ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
        break

    if not prompt:
        continue

    # ── Commands ──────────────────────────────────────────────────────────────
    if prompt.startswith(":quit"):
        print("Exiting.")
        break

    elif prompt.startswith(":temp"):
        try:
            current_temp = float(prompt.split()[1])
            print(f"Temperature set to {current_temp}")
        except (IndexError, ValueError):
            print("Usage: :temp 0.8")
        continue

    elif prompt.startswith(":max"):
        try:
            current_max_new = int(prompt.split()[1])
            print(f"Max new tokens set to {current_max_new}")
        except (IndexError, ValueError):
            print("Usage: :max 300")
        continue

    elif prompt.startswith(":compare"):
        # Use next non-command input as the prompt
        print("Enter prompt for comparison:")
        try:
            prompt = input(">>> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt:
            continue

        for temp in [0.7, 1.0, 1.2]:
            print_separator()
            print(f"  Temperature: {temp}")
            print_separator()
            out = generate(prompt, temp, current_max_new, args.top_k, args.top_p)
            print(out)
        print_separator()
        continue

    # ── Normal generation ─────────────────────────────────────────────────────
    print_separator()
    out = generate(prompt, current_temp, current_max_new, args.top_k, args.top_p)
    print(out)
    print_separator()
