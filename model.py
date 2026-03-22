"""
model.py — Hindi LSTM Language Model
======================================
Architecture:
    Embedding [24000 × 1024]
    → Dropout 0.3
    → LSTM Layer 1 [1024 hidden]
    → Dropout 0.4
    → LSTM Layer 2 [1024 hidden]
    → Dropout 0.4
    → LSTM Layer 3 [1024 hidden]
    → Dropout 0.3
    → Linear [1024 → 24000]  (weight-tied to embedding)

Total parameters: ~49.75M
"""

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


class HindiLM(nn.Module):

    def __init__(
        self,
        vocab_size   : int   = 24_000,
        embed_dim    : int   = 1_024,
        hidden_dim   : int   = 1_024,
        num_layers   : int   = 3,
        dropout_emb  : float = 0.3,    # input dropout (after embedding)
        dropout_inter: float = 0.4,    # inter-layer dropout (between LSTMs)
        dropout_out  : float = 0.3,    # output dropout (before projection)
        use_grad_ckpt: bool  = True,   # gradient checkpointing to save VRAM
    ):
        super().__init__()

        self.vocab_size    = vocab_size
        self.embed_dim     = embed_dim
        self.hidden_dim    = hidden_dim
        self.num_layers    = num_layers
        self.use_grad_ckpt = use_grad_ckpt

        # ── Embedding ─────────────────────────────────────────────────────────
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # ── Dropouts ──────────────────────────────────────────────────────────
        self.drop_emb   = nn.Dropout(dropout_emb)
        self.drop_inter = nn.Dropout(dropout_inter)
        self.drop_out   = nn.Dropout(dropout_out)

        # ── LSTM Layers (individual modules, not nn.LSTM multi-layer) ─────────
        # Using separate nn.LSTM modules per layer gives us clean dropout
        # control between layers and makes gradient checkpointing straightforward.
        self.lstm1 = nn.LSTM(
            input_size  = embed_dim,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
        )
        self.lstm2 = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
        )
        self.lstm3 = nn.LSTM(
            input_size  = hidden_dim,
            hidden_size = hidden_dim,
            num_layers  = 1,
            batch_first = True,
        )

        # ── Output projection (weight-tied) ───────────────────────────────────
        # Linear without bias — weight will be set to embedding.weight
        self.decoder = nn.Linear(hidden_dim, vocab_size, bias=False)
        self.decoder.weight = self.embedding.weight   # weight tying

        # ── Weight initialisation ─────────────────────────────────────────────
        self._init_weights()

    # ── Init ──────────────────────────────────────────────────────────────────

    def _init_weights(self):
        # Embedding: small uniform (standard practice)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        nn.init.constant_(self.embedding.weight[0], 0)   # pad token → zero

        for lstm in [self.lstm1, self.lstm2, self.lstm3]:
            for name, param in lstm.named_parameters():
                if "weight_ih" in name:
                    # Input-hidden weights: Xavier uniform
                    nn.init.xavier_uniform_(param)
                elif "weight_hh" in name:
                    # Hidden-hidden weights: orthogonal (helps with gradients)
                    nn.init.orthogonal_(param)
                elif "bias" in name:
                    # All biases zero, except forget gate bias set to 1.0
                    # This helps the LSTM remember by default early in training
                    nn.init.zeros_(param)
                    hidden = param.size(0) // 4
                    param.data[hidden : 2 * hidden].fill_(1.0)

    # ── Hidden state helpers ──────────────────────────────────────────────────

    def init_hidden(self, batch_size: int, device: torch.device):
        """Return zero hidden states for all three LSTM layers."""
        z = lambda: torch.zeros(1, batch_size, self.hidden_dim, device=device)
        return [
            (z(), z()),   # layer 1: (h, c)
            (z(), z()),   # layer 2: (h, c)
            (z(), z()),   # layer 3: (h, c)
        ]

    @staticmethod
    def detach_hidden(hidden):
        """Detach hidden states from the computation graph.

        Called between batches to prevent backprop through the entire epoch.
        We keep the hidden state values (continuity across windows) but cut
        the gradient chain (memory and stability).
        """
        return [(h.detach(), c.detach()) for (h, c) in hidden]

    # ── Forward ───────────────────────────────────────────────────────────────

    def _lstm1_forward(self, x, h):
        return self.lstm1(x, h)

    def _lstm2_forward(self, x, h):
        return self.lstm2(x, h)

    def _lstm3_forward(self, x, h):
        return self.lstm3(x, h)

    def forward(
        self,
        input_ids: torch.Tensor,           # [batch, seq_len]
        hidden   : list | None = None,     # list of 3 (h, c) tuples
    ):
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        if hidden is None:
            hidden = self.init_hidden(batch_size, device)

        # ── Embedding + dropout ───────────────────────────────────────────────
        x = self.embedding(input_ids)      # [batch, seq_len, embed_dim]
        x = self.drop_emb(x)

        # ── LSTM Layer 1 ──────────────────────────────────────────────────────
        if self.use_grad_ckpt and self.training:
            # gradient checkpointing: recompute activations on backward pass
            # instead of storing them — halves activation memory
            x, h1 = checkpoint(self._lstm1_forward, x, hidden[0], use_reentrant=False)
        else:
            x, h1 = self.lstm1(x, hidden[0])
        x = self.drop_inter(x)

        # ── LSTM Layer 2 ──────────────────────────────────────────────────────
        if self.use_grad_ckpt and self.training:
            x, h2 = checkpoint(self._lstm2_forward, x, hidden[1], use_reentrant=False)
        else:
            x, h2 = self.lstm2(x, hidden[1])
        x = self.drop_inter(x)

        # ── LSTM Layer 3 ──────────────────────────────────────────────────────
        if self.use_grad_ckpt and self.training:
            x, h3 = checkpoint(self._lstm3_forward, x, hidden[2], use_reentrant=False)
        else:
            x, h3 = self.lstm3(x, hidden[2])
        x = self.drop_out(x)

        # ── Project to vocabulary ─────────────────────────────────────────────
        logits = self.decoder(x)           # [batch, seq_len, vocab_size]

        new_hidden = [h1, h2, h3]
        return logits, new_hidden

    # ── Inference helper ──────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        prompt_ids  : torch.Tensor,    # [1, prompt_len]  — single sequence
        max_new     : int   = 200,
        temperature : float = 1.0,
        top_k       : int   = 50,
        top_p       : float = 0.95,
        eos_id      : int   = 3,
    ) -> list[int]:
        """Nucleus (top-p) sampling with optional top-k and temperature."""
        self.eval()
        device = prompt_ids.device
        hidden = self.init_hidden(1, device)

        generated = prompt_ids[0].tolist()

        # Prime the hidden state on the prompt without sampling
        if prompt_ids.shape[1] > 1:
            _, hidden = self.forward(prompt_ids[:, :-1], hidden)
            hidden = self.detach_hidden(hidden)

        input_id = prompt_ids[:, -1:]     # [1, 1] — last token of prompt

        for _ in range(max_new):
            logits, hidden = self.forward(input_id, hidden)
            hidden = self.detach_hidden(hidden)

            logits = logits[:, -1, :]      # [1, vocab_size]

            # Temperature scaling
            if temperature != 1.0:
                logits = logits / temperature

            # Top-k filtering
            if top_k > 0:
                values, _ = torch.topk(logits, top_k)
                min_val    = values[:, -1].unsqueeze(-1)
                logits     = logits.masked_fill(logits < min_val, float("-inf"))

            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cumprobs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative prob above threshold
                sorted_logits[cumprobs - torch.softmax(sorted_logits, dim=-1) > top_p] = float("-inf")
                logits = torch.zeros_like(logits).scatter_(1, sorted_idx, sorted_logits)

            probs   = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)

            token = next_id.item()
            generated.append(token)

            if token == eos_id:
                break

            input_id = next_id

        return generated

    # ── Parameter count ───────────────────────────────────────────────────────

    def count_parameters(self) -> dict:
        embedding_params = self.embedding.weight.numel()
        lstm1_params     = sum(p.numel() for p in self.lstm1.parameters())
        lstm2_params     = sum(p.numel() for p in self.lstm2.parameters())
        lstm3_params     = sum(p.numel() for p in self.lstm3.parameters())
        # decoder shares weights with embedding — don't double count
        total = embedding_params + lstm1_params + lstm2_params + lstm3_params

        return {
            "embedding"  : embedding_params,
            "lstm_layer1": lstm1_params,
            "lstm_layer2": lstm2_params,
            "lstm_layer3": lstm3_params,
            "decoder"    : 0,   # weight-tied
            "total"      : total,
        }