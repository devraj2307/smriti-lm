"""
Step 2: Corpus Preparation
===========================
- Loads full Sangraha Verified hin split
- Tokenizes with trained SentencePiece model
- Document-level shuffle
- Writes 990M token train split + 10M token val split
- Output: ./corpus/train.bin and ./corpus/val.bin  (numpy uint16, memory-mapped)

Token layout inside .bin files:
    Each document is stored as:
    [BOS] token_1 token_2 ... token_n [EOS]
    All documents concatenated sequentially.
    Training will slice this into windows of seq_len=256.
"""

import os
import random
import logging
import numpy as np
import sentencepiece as spm
from datasets import load_dataset

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TOKENIZER_MODEL     = "./tokenizer/hindi_spm.model"
OUTPUT_DIR          = "./corpus"

TOTAL_TOKENS        = 1_000_000_000    # 1B
VAL_TOKENS          = 10_000_000       # 10M  (1% held out)
TRAIN_TOKENS        = TOTAL_TOKENS - VAL_TOKENS  # 990M

MIN_DOC_LENGTH      = 100              # characters — skip very short documents
RANDOM_SEED         = 42

# Sangraha config
DATASET_NAME        = "ai4bharat/sangraha"
DATASET_SUBSET      = "verified"
LANGUAGE_SPLIT      = "hin"
TEXT_COLUMN         = "text"

# ── Setup ─────────────────────────────────────────────────────────────────────
os.makedirs(OUTPUT_DIR, exist_ok=True)
random.seed(RANDOM_SEED)

# ── Load tokenizer ────────────────────────────────────────────────────────────
log.info(f"Loading tokenizer from {TOKENIZER_MODEL}")
sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_MODEL)

BOS_ID = sp.bos_id()   # 2
EOS_ID = sp.eos_id()   # 3

log.info(f"  Vocab size : {sp.get_piece_size()}")
log.info(f"  BOS id     : {BOS_ID}")
log.info(f"  EOS id     : {EOS_ID}")


# ── Step 1: Load full dataset into memory ─────────────────────────────────────
# Not streaming this time — we need to shuffle across the full corpus.
# With 480GB RAM this is safe. The raw Hindi text will be a few GB at most.

log.info("=" * 60)
log.info("Loading full Sangraha Verified hin split...")
log.info("This will download the dataset if not cached.")
log.info("Expected download size: 3–8 GB. May take 30–90 minutes.")
log.info("=" * 60)

dataset = load_dataset(
    "parquet",
    data_files=f"hf://datasets/{DATASET_NAME}/{DATASET_SUBSET}/{LANGUAGE_SPLIT}/*.parquet",
    split="train",
)

log.info(f"Dataset loaded. Total documents: {len(dataset):,}")


# ── Step 2: Filter short documents ───────────────────────────────────────────

log.info("Filtering short documents...")
before = len(dataset)
dataset = dataset.filter(
    lambda x: len(x[TEXT_COLUMN].strip()) >= MIN_DOC_LENGTH,
    num_proc=os.cpu_count(),
)
after = len(dataset)
log.info(f"  Removed {before - after:,} short documents ({100*(before-after)/before:.1f}%)")
log.info(f"  Remaining: {after:,} documents")


# ── Step 3: Shuffle at document level ─────────────────────────────────────────
# This is critical — prevents the model from seeing topically clustered text
# which would bias the hidden state distribution

log.info("Shuffling documents...")
dataset = dataset.shuffle(seed=RANDOM_SEED)
log.info("Shuffle complete.")


# ── Step 4: Tokenize and write .bin files ─────────────────────────────────────
# uint16 covers vocab sizes up to 65535 — our 24k vocab fits comfortably
# Each token takes 2 bytes. 1B tokens = ~2 GB on disk.

train_out = os.path.join(OUTPUT_DIR, "train.bin")
val_out   = os.path.join(OUTPUT_DIR, "val.bin")

log.info("=" * 60)
log.info("Tokenizing and writing binary files...")
log.info(f"  Train target : {TRAIN_TOKENS:,} tokens  → {train_out}")
log.info(f"  Val target   : {VAL_TOKENS:,} tokens   → {val_out}")
log.info("=" * 60)

train_tokens_written = 0
val_tokens_written   = 0
docs_processed       = 0
skipped              = 0

# We write val first (from the tail of the shuffled corpus)
# then train from the rest.
# Simpler approach: stream through, fill val first then train.
# Since corpus is shuffled, order doesn't matter.

val_buffer   = []
train_buffer = []
val_done     = False
train_done   = False

BUFFER_FLUSH_SIZE = 5_000_000   # flush to disk every 5M tokens to avoid RAM buildup

def flush_buffer(buf, filepath, mode="ab"):
    arr = np.array(buf, dtype=np.uint16)
    with open(filepath, mode) as f:
        f.write(arr.tobytes())
    buf.clear()

# Initialize files (overwrite if they exist)
open(train_out, "wb").close()
open(val_out,   "wb").close()

for example in dataset:
    if train_done and val_done:
        break

    text = example[TEXT_COLUMN].strip()
    if not text:
        skipped += 1
        continue

    # Tokenize document — prepend BOS, append EOS
    token_ids = [BOS_ID] + sp.encode(text, out_type=int) + [EOS_ID]

    # Very short tokenized documents are noise — skip
    if len(token_ids) < 5:
        skipped += 1
        continue

    # Fill val split first, then train
    if not val_done:
        val_buffer.extend(token_ids)
        val_tokens_written += len(token_ids)

        if len(val_buffer) >= BUFFER_FLUSH_SIZE:
            flush_buffer(val_buffer, val_out)

        if val_tokens_written >= VAL_TOKENS:
            # Flush remaining
            if val_buffer:
                flush_buffer(val_buffer, val_out)
            val_done = True
            log.info(f"  Val split complete: {val_tokens_written:,} tokens written.")

    elif not train_done:
        train_buffer.extend(token_ids)
        train_tokens_written += len(token_ids)

        if len(train_buffer) >= BUFFER_FLUSH_SIZE:
            flush_buffer(train_buffer, train_out)

        if train_tokens_written >= TRAIN_TOKENS:
            if train_buffer:
                flush_buffer(train_buffer, train_out)
            train_done = True
            log.info(f"  Train split complete: {train_tokens_written:,} tokens written.")

    docs_processed += 1

    if docs_processed % 50_000 == 0:
        log.info(
            f"  Docs: {docs_processed:,} | "
            f"Train: {train_tokens_written:,} / {TRAIN_TOKENS:,} "
            f"({100*train_tokens_written/TRAIN_TOKENS:.1f}%) | "
            f"Val: {val_tokens_written:,} / {VAL_TOKENS:,} "
            f"({'done' if val_done else f'{100*val_tokens_written/VAL_TOKENS:.1f}%'})"
        )

# Flush any remaining buffers
if val_buffer:
    flush_buffer(val_buffer, val_out)
if train_buffer:
    flush_buffer(train_buffer, train_out)


# ── Step 5: Verify output files ───────────────────────────────────────────────

log.info("=" * 60)
log.info("Verifying output files...")

for label, path, expected in [
    ("train", train_out, TRAIN_TOKENS),
    ("val",   val_out,   VAL_TOKENS),
]:
    size_bytes = os.path.getsize(path)
    n_tokens   = size_bytes // 2   # uint16 = 2 bytes each

    log.info(f"  {label}.bin")
    log.info(f"    File size : {size_bytes / 1e9:.3f} GB")
    log.info(f"    Tokens    : {n_tokens:,}")
    log.info(f"    Expected  : ~{expected:,}")

    # Quick sanity — load first 20 tokens and decode
    data   = np.memmap(path, dtype=np.uint16, mode="r")
    sample = data[:20].tolist()
    decoded = sp.decode(sample)
    log.info(f"    First 20 tokens decoded: {decoded}")

log.info("=" * 60)
log.info("Step 2 complete. Ready for Step 3: Model training.")
log.info(f"  Train : {train_out}")
log.info(f"  Val   : {val_out}")
log.info(f"  Docs skipped (too short / empty): {skipped:,}")
log.info("=" * 60)