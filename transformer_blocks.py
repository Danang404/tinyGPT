"""
transformer_blocks.py
Modul arsitektur Transformer untuk TinyGPT
Topik: Bitcoin & Blockchain
"""

import torch
import torch.nn as nn
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Self-Attention dengan masking kausal (decoder-only).
    Setiap head belajar pola perhatian berbeda dalam teks.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model harus habis dibagi n_heads"
        self.d_model  = d_model
        self.n_heads  = n_heads
        self.d_k      = d_model // n_heads

        # Proyeksi Q, K, V sekaligus agar lebih efisien
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        # Hitung Q, K, V sekaligus
        qkv = self.qkv_proj(x).split(self.d_model, dim=2)
        q, k, v = [t.view(B, T, self.n_heads, self.d_k).transpose(1, 2)
                   for t in qkv]

        # Scaled Dot-Product Attention
        scale  = math.sqrt(self.d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale

        # Causal mask — token hanya boleh melihat token sebelumnya
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(mask, float('-inf'))

        attn   = torch.softmax(scores, dim=-1)
        attn   = self.dropout(attn)
        out    = torch.matmul(attn, v)
        out    = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network.
    Memperluas dimensi 4x lalu menciut kembali (pola standar Transformer).
    """
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),                          # GELU lebih smooth dari ReLU
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """
    Satu blok decoder Transformer:
      x -> LayerNorm -> MultiHeadAttention -> residual
        -> LayerNorm -> FeedForward        -> residual
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2  = nn.LayerNorm(d_model)
        self.ff   = FeedForward(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))   # Pre-norm (lebih stabil saat training)
        x = x + self.ff(self.ln2(x))
        return x


class TinyGPTModel(nn.Module):
    """
    Model TinyGPT — Decoder-only Transformer untuk text generation.
    
    Args:
        vocab_size : ukuran kosakata
        d_model    : dimensi embedding
        n_heads    : jumlah attention head
        n_layers   : jumlah blok Transformer
        max_len    : panjang konteks maksimum
        dropout    : probabilitas dropout
    """
    def __init__(self, vocab_size: int, d_model: int = 128,
                 n_heads: int = 4, n_layers: int = 3,
                 max_len: int = 128, dropout: float = 0.1):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(max_len, d_model)
        self.drop      = nn.Dropout(dropout)
        self.blocks    = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.ln_final  = nn.LayerNorm(d_model)
        self.head      = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len   = max_len

        # Weight tying: embedding & output head berbagi bobot
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        """Inisialisasi bobot agar training lebih stabil."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.max_len, f"Urutan terlalu panjang ({T} > {self.max_len})"

        # Token + Positional Embedding
        pos   = torch.arange(T, device=idx.device).unsqueeze(0)
        x     = self.drop(self.token_emb(idx) + self.pos_emb(pos))

        # Lewatkan melalui semua blok Transformer
        for block in self.blocks:
            x = block(x)

        x    = self.ln_final(x)
        logits = self.head(x)     # (B, T, vocab_size)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int = 40) -> torch.Tensor:
        """
        Generasi teks autoregresif dengan temperature & top-k sampling.
        
        Args:
            idx           : token awal (B, T)
            max_new_tokens: jumlah token baru yang dihasilkan
            temperature   : > 1 lebih acak, < 1 lebih deterministik
            top_k         : hanya sampel dari k token teratas
        """
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_len:]
            logits   = self(idx_cond)
            logits   = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, -1:]] = float('-inf')

            probs    = torch.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, num_samples=1)
            idx      = torch.cat([idx, next_tok], dim=1)
        return idx

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
