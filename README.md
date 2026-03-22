# Hindi LSTM Language Model

A from-scratch generative language model for Hindi, trained on the [Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) verified corpus. Built with a 3-layer LSTM architecture and SentencePiece BPE tokenization.

---

## Model Details

| Property | Value |
|---|---|
| Architecture | 3-layer unidirectional LSTM |
| Parameters | ~49.77M |
| Vocabulary | 24,000 (SentencePiece BPE) |
| Embedding dimension | 1,024 |
| Hidden dimension | 1,024 |
| Sequence length | 256 tokens |
| Weight tying | Yes (input embedding = output projection) |
| Dropout | 0.3 (input/output), 0.4 (inter-layer) |
| Training data | Sangraha Verified `hin` split |
| Training tokens | ~1B |
| Epochs | 2 |
| Optimizer | AdamW |
| Learning rate | 3e-4 → 3e-5 (cosine decay, 2k warmup steps) |
| Gradient clipping | 0.5 |
| Label smoothing | 0.1 |

---

## Repository Structure

```
.
├── model.py              # Model architecture (HindiLM)
├── train_tokenizer.py    # Step 1 — train SentencePiece tokenizer
├── prepare_corpus.py     # Step 2 — tokenize and write binary corpus
├── train.py              # Step 3 — training loop
├── inference.py          # Interactive generation
├── tokenizer/            # Created by train_tokenizer.py
│   ├── hindi_spm.model
│   └── hindi_spm.vocab
├── corpus/               # Created by prepare_corpus.py
│   ├── train.bin         # ~1.85 GB, 990M tokens
│   └── val.bin           # ~0.02 GB, 10M tokens
├── checkpoints/          # Created by train.py
│   ├── best.pt
│   └── ckpt_step_XXXXXXXX.pt
└── runs/
    └── train.log
```

---

## Requirements

```bash
pip install torch sentencepiece datasets numpy
```

Python 3.10+. CUDA GPU recommended. Tested on NVIDIA H100 NVL.

Minimum hardware for training:
- 8GB VRAM (reduce `batch_size` to 16 in `train.py`)
- 32GB RAM
- 10GB disk space for corpus + checkpoints

---

## Training — Step by Step

### Step 1 — Train Tokenizer

Downloads a small sample from Sangraha Verified (streaming, ~16M chars) and trains a SentencePiece BPE tokenizer.

```bash
python3 train_tokenizer.py
```

Output: `tokenizer/hindi_spm.model`, `tokenizer/hindi_spm.vocab`
Runtime: ~20–30 minutes

---

### Step 2 — Prepare Corpus

Downloads the full Hindi split of Sangraha Verified, tokenizes it, and writes binary files for training. Requires ~12–15GB of disk for the raw download and ~2GB for the final `.bin` files.

```bash
python3 prepare_corpus.py
```

Output: `corpus/train.bin` (990M tokens), `corpus/val.bin` (10M tokens)
Runtime: ~1–2 hours (dominated by download speed)

---

### Step 3 — Train

```bash
python3 train.py
```

Logs loss, perplexity, learning rate, and tokens/sec every 500 steps. Saves checkpoints every 10,000 steps and always keeps the best validation perplexity checkpoint at `checkpoints/best.pt`.

**Key parameters to tune in `train.py`:**

```python
"batch_size"       : 32,     # increase if you have more VRAM
"grad_accum_steps" : 1,
"epochs"           : 2,
"seq_len"          : 512,
```

Training resumes automatically from the latest checkpoint if interrupted — just re-run `python3 train.py`.

Runtime: ~14–18 hours on a shared H100 for 2 epochs over 1B tokens.

**Expected perplexity milestones:**

| Milestone | Expected Val Perplexity |
|---|---|
| Step 10,000 | 150–180 |
| End of epoch 1 | 60–90 |
| End of epoch 2 | 35–55 |

---

## Inference

```bash
python3 inference.py
```

Loads `checkpoints/best.pt` by default and opens an interactive prompt.

**Options:**

```bash
python3 inference.py --checkpoint ./checkpoints/best.pt
python3 inference.py --temp 0.8 --max_new 200
python3 inference.py --top_k 50 --top_p 0.95
```

**In-session commands:**

| Command | Effect |
|---|---|
| `:temp 0.8` | Change sampling temperature |
| `:max 300` | Change max new tokens |
| `:compare` | Generate same prompt at temps 0.7, 1.0, 1.2 |
| `:quit` | Exit |

**Example prompts:**

```
>>> भारत एक विशाल देश है जहाँ
>>> सरकार ने घोषणा की कि देश में
>>> बारिश होने की वजह से सड़कें
>>> वह लड़की जो कल आई थी
```

Inference can run in a separate terminal while training continues — it loads the checkpoint read-only and does not affect training.

---

## Notes

- The model is a **base language model** — it generates continuations, not instruction-following responses. It has not been fine-tuned for chat or QA.
- Training data is formal/written Hindi (news, web). Colloquial Hindi and code-switching (Hindi-English) are underrepresented.
- Gradient checkpointing is enabled by default during training to reduce VRAM usage. It adds ~20% compute overhead but halves activation memory.
- The output projection layer shares weights with the embedding layer (weight tying), saving 24.57M parameters with no loss in quality.

---

## Dataset

[Sangraha](https://huggingface.co/datasets/ai4bharat/sangraha) by AI4Bharat / IIT Madras. Verified subset, `hin` split.
License: CC-BY 4.0.
