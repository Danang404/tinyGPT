"""
tinygpt.py
==========
TinyGPT — Bitcoin & Blockchain Corpus
Proyek Data Mining | Universitas Amikom Yogyakarta

Alur kerja:
  1. Load corpus (corpus.txt)
  2. Fit masing-masing tokenizer
  3. Buat dataset & dataloader
  4. Latih TinyGPTModel (transformer_blocks.py)
  5. Generate teks & tampilkan analisis perbandingan
"""

import os
import time
import math
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler  # <-- TAMBAHAN BUAT SPEED UP

# ── modul lokal ──
from tokenizers import CharTokenizer, WordTokenizer, BPETokenizer
from transformer_blocks import TinyGPTModel

# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ═══════════════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════════════

class TextDataset(Dataset):
    """
    Dataset sederhana untuk language modeling.
    Input  x : token[i   : i+block_size]
    Target y : token[i+1 : i+block_size+1]  (geser 1 posisi)
    """
    def __init__(self, token_ids: list, block_size: int):
        self.data       = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return max(0, len(self.data) - self.block_size)

    def __getitem__(self, idx: int):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_model(model: TinyGPTModel, loader: DataLoader,
                epochs: int, lr: float = 3e-4) -> dict:
    """
    Latih model dengan AdamW + CosineAnnealing scheduler + Mixed Precision.
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    
    # <-- TAMBAHAN SCALER UNTUK MENCEGAH UNDERFLOW DI MIXED PRECISION -->
    scaler = GradScaler(DEVICE)
    
    model.to(DEVICE).train()

    history = {"loss": [], "perplexity": [], "epoch_time": []}

    for epoch in range(1, epochs + 1):
        t0         = time.time()
        total_loss = 0.0
        n_batches  = 0

        for x, y in loader:
            x, y   = x.to(DEVICE), y.to(DEVICE)
            
            # <-- TAMBAHAN AUTOCAST UNTUK MEMPERCEPAT KOMPUTASI -->
            with autocast(device_type=DEVICE):
                logits = model(x)                        # (B, T, V)
                B, T, V = logits.shape
                loss   = criterion(logits.view(B * T, V), y.view(B * T))

            optimizer.zero_grad()
            
            # <-- UPDATE BACKWARD PASS PAKAI SCALER -->
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg  = total_loss / max(n_batches, 1)
        ppl  = math.exp(min(avg, 20))
        ela  = time.time() - t0

        history["loss"].append(round(avg, 4))
        history["perplexity"].append(round(ppl, 2))
        history["epoch_time"].append(round(ela, 3))

        # Print setiap 10 epoch
        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs} | "
                  f"Loss: {avg:.4f} | PPL: {ppl:.2f} | {ela:.2f}s")

    return history


# ═══════════════════════════════════════════════════════════════════
# TEXT GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate_text(model: TinyGPTModel, tokenizer,
                  prompt: str, max_new: int = 80,
                  temperature: float = 0.8, top_k: int = 40) -> str:
    """
    Generate teks autoregresif dari sebuah prompt.
    """
    ids = tokenizer.encode(prompt)
    if not ids:
        return "[tidak ada token valid dari prompt]"
    idx = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    out = model.generate(idx, max_new_tokens=max_new,
                         temperature=temperature, top_k=top_k)
    return tokenizer.decode(out[0].tolist())


# ═══════════════════════════════════════════════════════════════════
# SATU EKSPERIMEN
# ═══════════════════════════════════════════════════════════════════

def run_experiment(tokenizer, corpus: str, config: dict) -> dict:
    """
    Jalankan satu eksperimen end-to-end:
      fit tokenizer → buat dataset → latih model → generate → return hasil
    """
    print(f"\n{'='*58}")
    print(f"  TOKENIZER : {tokenizer.name}")
    print(f"{'='*58}")

    # 1. Fit tokenizer & encode corpus
    tokenizer.fit(corpus)
    ids = tokenizer.encode(corpus)

    vocab_sz = tokenizer.vocab_size
    n_tokens = len(ids)
    ratio    = round(len(corpus) / max(n_tokens, 1), 3)

    print(f"  Vocab size      : {vocab_sz:,}")
    print(f"  Total tokens    : {n_tokens:,}")
    print(f"  Char/token ratio: {ratio}")
    print(tokenizer.info())

    # 2. Dataset & DataLoader
    dataset = TextDataset(ids, config["block_size"])
    if len(dataset) == 0:
        print("  [SKIP] Dataset terlalu kecil.")
        return {"tokenizer": tokenizer.name, "skipped": True}

    loader = DataLoader(dataset, batch_size=config["batch_size"],
                        shuffle=True, drop_last=True)

    # 3. Buat model
    model = TinyGPTModel(
        vocab_size = vocab_sz,
        d_model    = config["d_model"],
        n_heads    = config["n_heads"],
        n_layers   = config["n_layers"],
        max_len    = config["block_size"],
        dropout    = config["dropout"],
    ).to(DEVICE)
    
    # <-- TAMBAHAN TORCH.COMPILE UNTUK BOOST GPU -->
    # if DEVICE == "cuda":
      #    print("  [INFO] torch.compile aktif! Training bakal lebih ngebut.")
       # except Exception:
        #    print("  [INFO] torch.compile dilewati (skip).")

    n_params = model.count_parameters()
    print(f"  Model params    : {n_params:,}")
    print(f"  Dataset samples : {len(dataset):,}")
    print(f"  Batches/epoch   : {len(loader)}")
    print()

    # 4. Training
    t_start = time.time()
    history = train_model(model, loader,
                          epochs=config["epochs"], lr=config["lr"])
    total_time = round(time.time() - t_start, 2)

    # 5. Generate teks dari beberapa prompt
    prompts = [
        "Bitcoin adalah",
        "blockchain teknologi",
        "smart contract",
        "proof of work",
    ]
    generations = []
    print("\n  ── Hasil Generasi Teks ──")
    for p in prompts:
        g = generate_text(model, tokenizer, p,
                          max_new     = config["max_new"],
                          temperature = config["temperature"],
                          top_k       = config["top_k"])
        generations.append({"prompt": p, "output": g})
        print(f"  Prompt : '{p}'")
        print(f"  Output : {g[:180]}")
        print()

    # 6. Ringkasan
    final_loss = history["loss"][-1]
    final_ppl  = history["perplexity"][-1]
    best_ppl   = min(history["perplexity"])

    print(f"  ✓ Final Loss  : {final_loss:.4f}")
    print(f"  ✓ Final PPL   : {final_ppl:.2f}")
    print(f"  ✓ Best PPL    : {best_ppl:.2f}")
    print(f"  ✓ Total waktu : {total_time}s")

    return {
        "tokenizer"      : tokenizer.name,
        "vocab_size"     : vocab_sz,
        "total_tokens"   : n_tokens,
        "char_per_token" : ratio,
        "model_params"   : n_params,
        "epochs"         : config["epochs"],
        "final_loss"     : final_loss,
        "final_ppl"      : final_ppl,
        "best_ppl"       : best_ppl,
        "total_time_s"   : total_time,
        "history"        : history,
        "generations"    : generations,
    }


# ═══════════════════════════════════════════════════════════════════
# ANALISIS KOMPARATIF
# ═══════════════════════════════════════════════════════════════════

def print_comparison(results: list):
    """Cetak tabel perbandingan performa semua tokenizer."""
    valid = [r for r in results if not r.get("skipped")]
    if not valid:
        print("Tidak ada hasil valid.")
        return

    print(f"\n\n{'#'*65}")
    print("  ANALISIS KOMPARATIF PERFORMA MODEL")
    print(f"{'#'*65}")
    print(
        f"{'Tokenizer':<24} {'Vocab':>6} {'Tokens':>8} "
        f"{'Params':>8} {'Loss':>7} {'PPL':>8} {'BestPPL':>9} {'Waktu':>7}"
    )
    print("-" * 80)
    for r in valid:
        print(
            f"{r['tokenizer']:<24} "
            f"{r['vocab_size']:>6,} "
            f"{r['total_tokens']:>8,} "
            f"{r['model_params']:>8,} "
            f"{r['final_loss']:>7.4f} "
            f"{r['final_ppl']:>8.2f} "
            f"{r['best_ppl']:>9.2f} "
            f"{r['total_time_s']:>6.1f}s"
        )

    best = min(valid, key=lambda r: r["best_ppl"])
    worst = max(valid, key=lambda r: r["best_ppl"])

    print(f"\n  ★ Model TERBAIK  : {best['tokenizer']}  (Best PPL = {best['best_ppl']:.2f})")
    print(f"  ✗ Model TERBURUK : {worst['tokenizer']} (Best PPL = {worst['best_ppl']:.2f})")

    print(f"\n  ANALISIS:\n")
    print(
        "  • Character-Level memiliki vocab paling kecil tetapi urutan token\n"
        "    paling panjang. Model harus belajar menyusun kata dari karakter,\n"
        "    sehingga membutuhkan lebih banyak epoch untuk konvergensi.\n"
    )
    print(
        "  • Word-Level memiliki urutan token paling pendek dan setiap token\n"
        "    sudah bermakna semantis. PPL biasanya lebih rendah pada corpus\n"
        "    kecil karena model langsung belajar pola kata.\n"
    )
    print(
        "  • BPE-Like menawarkan keseimbangan: vocab lebih kecil dari word-level\n"
        "    tetapi lebih ekspresif dari char-level. Cocok untuk corpus dengan\n"
        "    banyak kata teknis seperti 'blockchain', 'cryptocurrency', dll.\n"
    )


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    # ── Load corpus ──
    corpus_path = os.path.join(os.path.dirname(__file__), "corpus.txt")
    with open(corpus_path, "r", encoding="utf-8") as f:
        corpus = f.read()

    print(f"\n{'#'*58}")
    print(f"  TinyGPT — Bitcoin & Blockchain")
    print(f"  Corpus  : {len(corpus.split()):,} kata | {len(corpus):,} karakter")
    print(f"  Device  : {DEVICE.upper()}")
    print(f"{'#'*58}")

    # ── Hyperparameter ──
    config = {
        "block_size" : 64,    # panjang konteks (dalam token)
        "batch_size" : 16,    # jumlah sampel per batch
        "d_model"    : 128,   # dimensi embedding
        "n_heads"    : 4,     # jumlah attention head
        "n_layers"   : 3,     # jumlah blok Transformer
        "dropout"    : 0.1,   # probabilitas dropout
        "epochs"     : 20,    # <-- UBAH KE 20 SEMENTARA BIAR CEPET NGETEST
        "lr"         : 3e-4,  # learning rate awal
        "max_new"    : 80,    # token baru saat generate
        "temperature": 0.8,   # kreativitas generasi
        "top_k"      : 40,    # top-k sampling
    }

    print(f"\n  HYPERPARAMETER:")
    for k, v in config.items():
        print(f"    {k:<14} : {v}")

    # ── 3 Tokenizer ──
    tokenizers = [
        CharTokenizer(),
        WordTokenizer(min_freq=1),
        BPETokenizer(num_merges=200),
    ]

    all_results = []
    for tok in tokenizers:
        result = run_experiment(tok, corpus, config)
        all_results.append(result)

    # ── Analisis ──
    print_comparison(all_results)

    # ── Simpan hasil ──
    out_path = os.path.join(os.path.dirname(__file__), "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\n  ✓ Hasil lengkap disimpan ke: {out_path}")

    return all_results


if __name__ == "__main__":
    main()