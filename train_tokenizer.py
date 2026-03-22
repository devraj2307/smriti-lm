"""
Step 1: Train SentencePiece BPE Tokenizer
==========================================
- Source  : Sangraha Verified, hin_Deva split
- Vocab   : 24,000
- Target  : ~10M tokens of raw Hindi text for tokenizer training
- Output  : hindi_spm.model + hindi_spm.vocab in ./tokenizer/
"""

import os
import sentencepiece as spm
from datasets import load_dataset
import logging

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_TOKENS       = 10_000_000   # approx tokens to collect for tokenizer training
AVG_TOKENS_PER_CHAR = 0.6          # rough estimate: 1 Hindi char ≈ 0.6 BPE tokens
TARGET_CHARS        = int(TARGET_TOKENS / AVG_TOKENS_PER_CHAR)   # ~16.6M chars

VOCAB_SIZE          = 24_000
CHARACTER_COVERAGE  = 0.9995       # captures rare Devanagari conjuncts
OUTPUT_DIR          = "./tokenizer"
SAMPLE_FILE         = os.path.join(OUTPUT_DIR, "spm_train_sample.txt")
MODEL_PREFIX        = os.path.join(OUTPUT_DIR, "hindi_spm")

# ── Sangraha dataset config ───────────────────────────────────────────────────
# Sangraha Verified is at ai4bharat/sangraha on HuggingFace
# The verified subset is split by language — we want hin_Deva
DATASET_NAME        = "ai4bharat/sangraha"
DATASET_SUBSET      = "verified"
LANGUAGE_FILTER     = "hin"       # split name in Sangraha Verified
TEXT_COLUMN         = "text"       # column name in the dataset

# ── Step 1: Collect raw text sample ──────────────────────────────────────────

def collect_sample():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log.info("Loading Sangraha Verified from HuggingFace (streaming mode)...")
    log.info("This avoids downloading the full dataset upfront.")

    # Streaming=True means documents arrive one by one — no full download needed
    # for tokenizer training. For the main corpus prep (Step 2), we will
    # download fully. Here, streaming is fine.
    # The verified subset uses language codes as split names directly
    # "hin" is the Hindi split — no separate language column filter needed
    dataset = load_dataset(
        DATASET_NAME,
        DATASET_SUBSET,
        split=LANGUAGE_FILTER,
        streaming=True,
        trust_remote_code=True,
    )

    log.info(f"Collecting ~{TARGET_TOKENS:,} tokens worth of text...")
    log.info(f"Target characters to collect: ~{TARGET_CHARS:,}")

    chars_collected = 0
    docs_collected  = 0

    with open(SAMPLE_FILE, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example.get(TEXT_COLUMN, "").strip()

            # Basic quality filter — skip very short documents
            if len(text) < 100:
                continue

            # Write one document per line (SentencePiece reads line by line)
            f.write(text.replace("\n", " ") + "\n")

            chars_collected += len(text)
            docs_collected  += 1

            if docs_collected % 10_000 == 0:
                log.info(
                    f"  Docs: {docs_collected:,} | "
                    f"Chars: {chars_collected:,} / {TARGET_CHARS:,} "
                    f"({100 * chars_collected / TARGET_CHARS:.1f}%)"
                )

            if chars_collected >= TARGET_CHARS:
                break

    log.info(f"Sample collection done.")
    log.info(f"  Total documents : {docs_collected:,}")
    log.info(f"  Total characters: {chars_collected:,}")
    log.info(f"  Saved to        : {SAMPLE_FILE}")
    return docs_collected, chars_collected


# ── Step 2: Train SentencePiece BPE ──────────────────────────────────────────

def train_tokenizer():
    log.info("Starting SentencePiece BPE training...")
    log.info(f"  Vocab size        : {VOCAB_SIZE:,}")
    log.info(f"  Character coverage: {CHARACTER_COVERAGE}")
    log.info(f"  Input file        : {SAMPLE_FILE}")

    spm.SentencePieceTrainer.train(
        input                   = SAMPLE_FILE,
        model_prefix            = MODEL_PREFIX,
        model_type              = "bpe",               # BPE, not unigram
        vocab_size              = VOCAB_SIZE,
        character_coverage      = CHARACTER_COVERAGE,
        pad_id                  = 0,                   # <pad>
        unk_id                  = 1,                   # <unk>
        bos_id                  = 2,                   # <s>  (begin of sequence)
        eos_id                  = 3,                   # </s> (end of sequence)
        pad_piece               = "<pad>",
        unk_piece               = "<unk>",
        bos_piece               = "<s>",
        eos_piece               = "</s>",
        # Allow whitespace normalization but preserve Devanagari properly
        normalization_rule_name = "nmt_nfkc",
        # SentencePiece adds a space prefix to tokens — keep this on for BPE
        add_dummy_prefix        = True,
        # Prevents splitting numbers aggressively
        split_digits            = True,
        # Use all available threads
        num_threads             = os.cpu_count(),
        input_sentence_size     = 5_000_000,           # max lines to read
        shuffle_input_sentence  = True,
    )

    log.info("Tokenizer training complete.")
    log.info(f"  Model saved to : {MODEL_PREFIX}.model")
    log.info(f"  Vocab saved to : {MODEL_PREFIX}.vocab")


# ── Step 3: Sanity check the trained tokenizer ────────────────────────────────

def sanity_check():
    log.info("Running sanity checks on trained tokenizer...")

    sp = spm.SentencePieceProcessor()
    sp.load(f"{MODEL_PREFIX}.model")

    log.info(f"  Vocabulary size : {sp.get_piece_size()}")

    test_sentences = [
        "भारत एक विशाल देश है।",                           # simple sentence
        "वह लड़की जो कल आई थी वह आज नहीं आई।",             # gender agreement
        "सरकार ने नई नीति की घोषणा की।",                   # formal register
        "बारिश होने की वजह से सड़कें भीग गई थीं।",          # causal construction
        "खाना खाकर वह सो गया।",                             # verb chaining
    ]

    log.info("  Sample tokenizations:")
    for sentence in test_sentences:
        tokens  = sp.encode_as_pieces(sentence)
        ids     = sp.encode_as_ids(sentence)
        decoded = sp.decode_pieces(tokens)
        log.info(f"    Input   : {sentence}")
        log.info(f"    Tokens  : {tokens}")
        log.info(f"    IDs     : {ids}")
        log.info(f"    Decoded : {decoded}")
        log.info(f"    Roundtrip OK: {decoded.replace(' ', '') == sentence.replace(' ', '')}")
        log.info("")

    # Check that key Hindi morphemes appear as clean tokens
    important_pieces = ["▁में", "▁की", "▁के", "▁का", "▁है", "▁से", "▁को", "▁ने"]
    log.info("  Checking key Hindi morphemes in vocabulary:")
    for piece in important_pieces:
        piece_id = sp.piece_to_id(piece)
        found    = piece_id != sp.unk_id()
        log.info(f"    {piece:10s}  id={piece_id:6d}  {'✓ found' if found else '✗ NOT FOUND — check coverage'}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    log.info("=" * 60)
    log.info("  Hindi LM — Step 1: Tokenizer Training")
    log.info("=" * 60)

    # 1. Collect sample text
    docs, chars = collect_sample()

    # 2. Train tokenizer
    train_tokenizer()

    # 3. Sanity check
    sanity_check()

    log.info("=" * 60)
    log.info("  Step 1 complete. Ready for Step 2: Corpus preparation.")
    log.info("=" * 60)