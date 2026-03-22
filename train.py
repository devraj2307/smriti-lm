"""
train.py — Training Script for Hindi LSTM Language Model
==========================================================
- Reads  : ./corpus/train.bin  and  ./corpus/val.bin
- Model  : HindiLM (model.py)
- Epochs : 2
- Logs   : ./runs/  (loss, perplexity, lr per step)
- Ckpts  : ./checkpoints/  (every 10k steps + best val)
"""

import os
import math
import time
import logging
import numpy as np
import sentencepiece as spm
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from model import HindiLM

# ── Logging ───────────────────────────────────────────────────────────────────
os.makedirs("./runs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./runs/train.log"),
    ],
)
log = logging.getLogger(__name__)

# ═════════════════════════════════════════════════════════════════════════════
# CONFIG — all hyperparameters in one place
# ═════════════════════════════════════════════════════════════════════════════

CFG = {
    # Data
    "train_bin"       : "./corpus/train.bin",
    "val_bin"         : "./corpus/val.bin",
    "tokenizer_model" : "./tokenizer/hindi_spm.model",

    # Model
    "vocab_size"      : 24_000,
    "embed_dim"       : 1_024,
    "hidden_dim"      : 1_024,
    "num_layers"      : 3,
    "dropout_emb"     : 0.3,
    "dropout_inter"   : 0.4,
    "dropout_out"     : 0.3,
    "use_grad_ckpt"   : True,

    # Training
    "epochs"          : 2,
    "seq_len"         : 256,
    "batch_size"      : 256,           # tune down if OOM
    "grad_accum_steps": 1,            # effective batch = 32 × 8 = 256
    "learning_rate"   : 3e-4,
    "lr_min"          : 3e-5,
    "warmup_steps"    : 2_000,
    "weight_decay"    : 0.01,
    "grad_clip"       : 0.5,
    "label_smoothing" : 0.1,

    # Checkpointing
    "ckpt_dir"        : "./checkpoints",
    "ckpt_every"      : 10_000,       # steps
    "keep_last_n"     : 5,

    # Evaluation
    "val_every"       : 10_000,       # also evaluate at ckpt points
    "val_batches"     : 200,          # how many val batches per eval
    "gen_every"       : 20_000,       # generate sample text every N steps

    # Misc
    "seed"            : 42,
    "device"          : "cuda" if torch.cuda.is_available() else "cpu",
    "dtype"           : torch.float32,
}

# ═════════════════════════════════════════════════════════════════════════════
# DATA LOADER
# ═════════════════════════════════════════════════════════════════════════════

class BinDataset:
    """
    Memory-mapped loader for tokenized .bin files.

    The file is one flat uint16 array of token IDs.
    We carve it into (input, target) pairs of length seq_len.
    - input  = tokens[i   : i + seq_len]
    - target = tokens[i+1 : i + seq_len + 1]

    Within an epoch we iterate sequentially so that hidden states
    remain valid across batches.
    """

    def __init__(self, path: str, seq_len: int, batch_size: int):
        self.seq_len    = seq_len
        self.batch_size = batch_size

        data = np.memmap(path, dtype=np.uint16, mode="r")
        self.data = torch.from_numpy(data.astype(np.int64))

        # Total number of full windows we can extract
        n_windows       = (len(self.data) - 1) // seq_len
        # Trim to a multiple of batch_size so every batch is full
        n_windows       = (n_windows // batch_size) * batch_size
        self.n_windows  = n_windows
        self.n_batches  = n_windows // batch_size

        log.info(f"  Dataset: {path}")
        log.info(f"    Tokens    : {len(self.data):,}")
        log.info(f"    Windows   : {n_windows:,}")
        log.info(f"    Batches   : {self.n_batches:,}")

    def __len__(self):
        return self.n_batches

    def get_batch(self, batch_idx: int, device: torch.device):
        """Return (input, target) tensors of shape [batch_size, seq_len]."""
        start  = batch_idx * self.batch_size
        inputs  = []
        targets = []

        for i in range(start, start + self.batch_size):
            pos = i * self.seq_len
            x   = self.data[pos     : pos + self.seq_len]
            y   = self.data[pos + 1 : pos + self.seq_len + 1]
            inputs.append(x)
            targets.append(y)

        inputs  = torch.stack(inputs).to(device)
        targets = torch.stack(targets).to(device)
        return inputs, targets


# ═════════════════════════════════════════════════════════════════════════════
# LR SCHEDULE
# ═════════════════════════════════════════════════════════════════════════════

def build_scheduler(optimizer, warmup_steps: int, total_steps: int, lr_min: float, lr_max: float):
    """
    Linear warmup from 0 to lr_max over warmup_steps,
    then cosine decay from lr_max down to lr_min over remaining steps.
    """
    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        # Scale so floor is lr_min / lr_max
        floor    = lr_min / lr_max
        return floor + (1.0 - floor) * cosine

    return LambdaLR(optimizer, lr_lambda)


# ═════════════════════════════════════════════════════════════════════════════
# CHECKPOINTING
# ═════════════════════════════════════════════════════════════════════════════

def save_checkpoint(model, optimizer, scheduler, step, epoch, val_ppl, is_best, cfg):
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)

    state = {
        "step"          : step,
        "epoch"         : epoch,
        "val_perplexity": val_ppl,
        "model_state"   : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "cfg"           : cfg,
    }

    path = os.path.join(cfg["ckpt_dir"], f"ckpt_step_{step:08d}.pt")
    torch.save(state, path)
    log.info(f"  Checkpoint saved: {path}")

    if is_best:
        best_path = os.path.join(cfg["ckpt_dir"], "best.pt")
        torch.save(state, best_path)
        log.info(f"  Best checkpoint updated: {best_path}  (ppl={val_ppl:.2f})")

    # Prune old checkpoints — keep only last N (excluding best.pt)
    all_ckpts = sorted([
        f for f in os.listdir(cfg["ckpt_dir"])
        if f.startswith("ckpt_step_") and f.endswith(".pt")
    ])
    while len(all_ckpts) > cfg["keep_last_n"]:
        old = os.path.join(cfg["ckpt_dir"], all_ckpts.pop(0))
        os.remove(old)
        log.info(f"  Pruned old checkpoint: {old}")


def load_checkpoint(path, model, optimizer, scheduler, device):
    log.info(f"Resuming from checkpoint: {path}")
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    scheduler.load_state_dict(state["scheduler_state"])
    return state["step"], state["epoch"], state["val_perplexity"]


# ═════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, val_dataset, criterion, device, max_batches: int) -> float:
    """Returns validation perplexity."""
    model.eval()
    total_loss  = 0.0
    total_tokens = 0
    hidden      = model.init_hidden(CFG["batch_size"], device)

    n = min(max_batches, len(val_dataset))

    for i in range(n):
        inputs, targets = val_dataset.get_batch(i, device)
        logits, hidden  = model(inputs, hidden)
        hidden          = model.detach_hidden(hidden)

        # Flatten for loss
        logits_flat  = logits.reshape(-1, CFG["vocab_size"])
        targets_flat = targets.reshape(-1)

        loss         = criterion(logits_flat, targets_flat)
        total_loss  += loss.item() * targets_flat.numel()
        total_tokens += targets_flat.numel()

    avg_loss   = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    model.train()
    return perplexity


# ═════════════════════════════════════════════════════════════════════════════
# GENERATION SAMPLE
# ═════════════════════════════════════════════════════════════════════════════

def generate_samples(model, sp, device, step: int):
    """Generate from fixed prompts at multiple temperatures. Logged only."""
    prompts = [
        "भारत एक विशाल देश है जहाँ",
        "सरकार ने घोषणा की कि",
        "बारिश होने की वजह से",
        "वह लड़की जो कल आई थी",
    ]

    log.info(f"  ── Generation samples at step {step} ──")
    model.eval()

    for prompt in prompts:
        ids        = sp.encode(prompt, out_type=int)
        prompt_ids = torch.tensor([[sp.bos_id()] + ids], dtype=torch.long, device=device)

        for temp in [0.7, 1.0, 1.2]:
            out_ids  = model.generate(
                prompt_ids,
                max_new     = 80,
                temperature = temp,
                top_k       = 50,
                top_p       = 0.95,
                eos_id      = sp.eos_id(),
            )
            out_text = sp.decode(out_ids)
            log.info(f"    [temp={temp}] {out_text}")

        log.info("")

    model.train()


# ═════════════════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═════════════════════════════════════════════════════════════════════════════

def train():
    torch.manual_seed(CFG["seed"])
    device = torch.device(CFG["device"])
    log.info(f"Device: {device}")
    if device.type == "cuda":
        log.info(f"GPU   : {torch.cuda.get_device_name(0)}")
        log.info(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # ── Load tokenizer ────────────────────────────────────────────────────────
    sp = spm.SentencePieceProcessor()
    sp.load(CFG["tokenizer_model"])
    log.info(f"Tokenizer loaded. Vocab size: {sp.get_piece_size()}")

    # ── Datasets ──────────────────────────────────────────────────────────────
    log.info("Loading datasets...")
    train_data = BinDataset(CFG["train_bin"], CFG["seq_len"], CFG["batch_size"])
    val_data   = BinDataset(CFG["val_bin"],   CFG["seq_len"], CFG["batch_size"])

    # ── Model ─────────────────────────────────────────────────────────────────
    model = HindiLM(
        vocab_size    = CFG["vocab_size"],
        embed_dim     = CFG["embed_dim"],
        hidden_dim    = CFG["hidden_dim"],
        num_layers    = CFG["num_layers"],
        dropout_emb   = CFG["dropout_emb"],
        dropout_inter = CFG["dropout_inter"],
        dropout_out   = CFG["dropout_out"],
        use_grad_ckpt = CFG["use_grad_ckpt"],
    ).to(device)

    param_info = model.count_parameters()
    log.info("Model parameter counts:")
    for k, v in param_info.items():
        log.info(f"  {k:15s}: {v:>12,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # ignore_index=0 means pad tokens don't contribute to loss
    criterion = nn.CrossEntropyLoss(
        label_smoothing = CFG["label_smoothing"],
        ignore_index    = 0,
    )

    # ── Optimizer ─────────────────────────────────────────────────────────────
    # Weight decay only on weight matrices, not biases or layernorms
    decay_params    = [p for n, p in model.named_parameters() if "bias" not in n]
    no_decay_params = [p for n, p in model.named_parameters() if "bias" in n]

    optimizer = AdamW(
        [
            {"params": decay_params,    "weight_decay": CFG["weight_decay"]},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr   = CFG["learning_rate"],
        betas= (0.9, 0.999),
        eps  = 1e-8,
    )

    # ── Scheduler ─────────────────────────────────────────────────────────────
    steps_per_epoch = len(train_data)
    total_steps     = steps_per_epoch * CFG["epochs"]
    # Divide by grad_accum because scheduler steps once per optimizer step
    total_opt_steps = total_steps // CFG["grad_accum_steps"]

    scheduler = build_scheduler(
        optimizer,
        warmup_steps = CFG["warmup_steps"],
        total_steps  = total_opt_steps,
        lr_min       = CFG["lr_min"],
        lr_max       = CFG["learning_rate"],
    )

    log.info(f"Total batches per epoch : {steps_per_epoch:,}")
    log.info(f"Total optimizer steps   : {total_opt_steps:,}")
    log.info(f"Warmup steps            : {CFG['warmup_steps']:,}")

    # ── Resume from checkpoint if one exists ──────────────────────────────────
    global_step  = 0
    start_epoch  = 0
    best_val_ppl = float("inf")

    latest_ckpt = None
    if os.path.isdir(CFG["ckpt_dir"]):
        ckpts = sorted([
            f for f in os.listdir(CFG["ckpt_dir"])
            if f.startswith("ckpt_step_") and f.endswith(".pt")
        ])
        if ckpts:
            latest_ckpt = os.path.join(CFG["ckpt_dir"], ckpts[-1])

    if latest_ckpt:
        global_step, start_epoch, best_val_ppl = load_checkpoint(
            latest_ckpt, model, optimizer, scheduler, device
        )
        log.info(f"Resumed at step {global_step}, epoch {start_epoch}, best ppl {best_val_ppl:.2f}")
    else:
        log.info("No checkpoint found. Starting fresh.")

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()

    for epoch in range(start_epoch, CFG["epochs"]):
        log.info("=" * 70)
        log.info(f"EPOCH {epoch + 1} / {CFG['epochs']}")
        log.info("=" * 70)

        # Fresh hidden state at the start of each epoch
        hidden = model.init_hidden(CFG["batch_size"], device)

        optimizer.zero_grad()
        accum_loss   = 0.0
        accum_tokens = 0
        t0           = time.time()

        for batch_idx in range(len(train_data)):
            inputs, targets = train_data.get_batch(batch_idx, device)

            # Forward
            logits, hidden = model(inputs, hidden)

            # Detach hidden state — preserve values, cut gradient chain
            hidden = model.detach_hidden(hidden)

            # Loss
            logits_flat  = logits.reshape(-1, CFG["vocab_size"])
            targets_flat = targets.reshape(-1)
            loss         = criterion(logits_flat, targets_flat)

            # Scale loss for gradient accumulation
            loss_scaled  = loss / CFG["grad_accum_steps"]
            loss_scaled.backward()

            accum_loss   += loss.item()
            accum_tokens += targets_flat.numel()
            global_step  += 1

            # ── Optimizer step (every grad_accum_steps batches) ───────────────
            if global_step % CFG["grad_accum_steps"] == 0:
                # Gradient clipping — essential for LSTMs
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            # ── Logging (every 500 steps) ─────────────────────────────────────
            if global_step % 500 == 0:
                elapsed      = time.time() - t0
                avg_loss     = accum_loss / 500
                perplexity   = math.exp(avg_loss)
                tok_per_sec  = accum_tokens / elapsed
                current_lr   = scheduler.get_last_lr()[0]
                vram_gb      = torch.cuda.memory_allocated() / 1e9 if device.type == "cuda" else 0

                log.info(
                    f"  step {global_step:>8,} | "
                    f"epoch {epoch+1} | "
                    f"loss {avg_loss:.4f} | "
                    f"ppl {perplexity:>8.2f} | "
                    f"lr {current_lr:.2e} | "
                    f"tok/s {tok_per_sec:>8,.0f} | "
                    f"vram {vram_gb:.1f}GB"
                )
                accum_loss   = 0.0
                accum_tokens = 0
                t0           = time.time()

            # ── Validation + checkpoint ───────────────────────────────────────
            if global_step % CFG["ckpt_every"] == 0:
                log.info(f"  Running validation at step {global_step}...")
                val_ppl  = evaluate(model, val_data, criterion, device, CFG["val_batches"])
                is_best  = val_ppl < best_val_ppl

                log.info(
                    f"  ── Val perplexity: {val_ppl:.2f}"
                    f"  {'  ← new best' if is_best else f'  (best: {best_val_ppl:.2f})'}"
                )

                if is_best:
                    best_val_ppl = val_ppl

                save_checkpoint(
                    model, optimizer, scheduler,
                    global_step, epoch, val_ppl, is_best, CFG
                )

                model.train()

            # ── Generation sample ─────────────────────────────────────────────
            if global_step % CFG["gen_every"] == 0:
                generate_samples(model, sp, device, global_step)
                model.train()

        # ── End of epoch ──────────────────────────────────────────────────────
        log.info(f"Epoch {epoch + 1} complete. Running full validation...")
        val_ppl = evaluate(model, val_data, criterion, device, len(val_data))
        is_best = val_ppl < best_val_ppl

        log.info(f"  End-of-epoch val perplexity: {val_ppl:.2f}")
        if is_best:
            best_val_ppl = val_ppl
            log.info("  New best perplexity!")

        save_checkpoint(
            model, optimizer, scheduler,
            global_step, epoch + 1, val_ppl, is_best, CFG
        )

        model.train()

    # ── Done ──────────────────────────────────────────────────────────────────
    log.info("=" * 70)
    log.info("Training complete.")
    log.info(f"  Best validation perplexity : {best_val_ppl:.2f}")
    log.info(f"  Best checkpoint            : {CFG['ckpt_dir']}/best.pt")
    log.info("=" * 70)


if __name__ == "__main__":
    train()